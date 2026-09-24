"""
Merge many 10x/Xenium sample directories into ONE anndata-format stacked.zarr
holding RAW COUNTS, streaming, at O(chunk) memory.

WHY THIS WORKS AT LOW MEMORY
10x's cell_feature_matrix.h5 stores
    shape  = [n_features, n_cells]
    indptr = length n_cells + 1        <- indptr indexes CELLS
i.e. CSC with one column per cell. So a contiguous block of cells is a
contiguous slice of data/indices, AND that block -- with features as the inner
index -- is already a valid CSR block of the (cells x genes) matrix we want:

    lo, hi = indptr[a], indptr[b]
    csr = csr_matrix((data[lo:hi], indices[lo:hi], indptr[a:b+1] - lo),
                     shape=(b - a, n_features))

No transpose and no re-sorting. Peak memory is one chunk: at 50,000 cells and
~2,200 nnz/cell that is ~110M nnz ~ 1.3 GB, versus ~27 GB for a single
sc.read_10x_h5 of a 2.2-billion-nonzero sample.

Because each sample's cells become a contiguous row block, samples are appended
straight into one output store -- there are no per-sample h5ads and no
concat_on_disk step. Row order is the order of --samples, which is what makes
banksy_pca_multi.build_W_block's block_diag valid downstream.

WHAT IT WRITES  (anndata zarr v2, readable with anndata.io.read_elem)
    stacked.zarr/X             CSR raw counts, float32, int64 indices
    stacked.zarr/obs           cells.csv.gz columns + 'sample'
    stacked.zarr/var           gene names
    stacked.zarr/obsm/spatial  x_centroid, y_centroid
    samples.txt, merge_summary.csv

RAW COUNTS ONLY, BY DESIGN. X is counts, there is no layers/counts and no
log1p. The downstream pipeline expects normalized+log1p X plus layers/counts,
so run the normalisation pass over this store before feeding it to
sc_dask_pipeline_post_pca_*. Normalising afterwards keeps the choice of
target_sum out of the merge, which matters because a per-sample median makes
samples non-comparable.

int64 indices are kept deliberately: one Atera sample alone is 2.23e9
nonzeros, already past the int32 max of 2,147,483,647, so a merged store
cannot use int32.

Usage:
    python merge_10x_to_zarr_streaming.py \
        --sample_root /path/to/output_10x/data_0921 \
        --samples     50174428_OTSP_12A8 50174428_OTSP_4F17 \
        --output_dir  /path/to/merged/data_0921_raw \
        --chunk_size  50000
"""

import argparse
import os
import sys
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata._io.specs import write_elem

pd.options.mode.string_storage = "python"
pd.options.future.infer_string = False

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)


def fmt(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"


def decode(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x)
                     for x in a], dtype=object)


def h5_group(f):
    """10x writes the matrix under a single top-level group ('matrix')."""
    keys = list(f.keys())
    if 'matrix' in keys:
        return f['matrix']
    if len(keys) == 1:
        return f[keys[0]]
    sys.exit(f"ERROR: cannot identify the matrix group among {keys}")


def scan_sample(path):
    """Read only metadata: shape, feature names/types. No matrix data."""
    with h5py.File(path, 'r') as f:
        g = h5_group(f)
        n_feat, n_cells = (int(x) for x in g['shape'][:])
        names = decode(g['features/name'][:])
        ftype = decode(g['features/feature_type'][:])
        nnz = int(g['data'].shape[0])
        if g['indptr'].shape[0] != n_cells + 1:
            sys.exit(f"ERROR: {path} indptr length {g['indptr'].shape[0]} != "
                     f"n_cells+1 ({n_cells+1}); this reader assumes 10x's "
                     f"cell-indexed layout")
    return dict(n_feat=n_feat, n_cells=n_cells, nnz=nnz,
                names=names, ftype=ftype)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--sample_root', required=True,
                   help='directory containing the per-sample 10x bundles')
    p.add_argument('--samples', nargs='+', required=True,
                   help="sample subdirectory names, in the order they should "
                        "appear in the output, or 'all'")
    p.add_argument('--output_dir', required=True)
    p.add_argument('--chunk_size', type=int, default=50_000,
                   help='cells per streamed block. Peak memory scales with '
                        'this; 50000 ~ 1.3 GB on the Atera panel.')
    p.add_argument('--all_features', action='store_true',
                   help='keep every feature. Default keeps only '
                        "'Gene Expression', dropping Negative Control "
                        'Codeword/Probe, Genomic Control, Unassigned and '
                        'Deprecated -- 6,624 of 24,674 on the Atera panel.')
    p.add_argument('--index_unique', default='_',
                   help="separator joining barcode and sample in obs_names")
    args = p.parse_args()

    root = args.sample_root
    if len(args.samples) == 1 and args.samples[0] == 'all':
        samples = sorted(d for d in os.listdir(root)
                         if os.path.isfile(f"{root}/{d}/cell_feature_matrix.h5"))
        print(f"--samples all -> {len(samples)}: {samples}")
    else:
        samples = args.samples

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # ── pass 1: metadata only, and verify the panel is shared ───────────────
    print(f"\n[scan] {len(samples)} samples")
    meta, ref = {}, None
    for s in samples:
        h5 = f"{root}/{s}/cell_feature_matrix.h5"
        cells_csv = f"{root}/{s}/cells.csv.gz"
        for f_ in (h5, cells_csv):
            if not os.path.isfile(f_):
                sys.exit(f"ERROR: missing {f_}")
        m = scan_sample(h5)
        meta[s] = m
        print(f"  {s:<28} {m['n_cells']:>10,} cells  "
              f"{m['n_feat']:>7,} features  nnz={m['nnz']:>15,}")
        if ref is None:
            ref = m
        elif not np.array_equal(m['names'], ref['names']):
            # A plain append requires identical gene axes. Intersecting would
            # silently change what every sample contributes, so refuse instead.
            sys.exit(f"ERROR: {s} has a different feature list from "
                     f"{samples[0]}. This merger appends and cannot reconcile "
                     f"different panels.")

    # gene mask, applied identically to every sample
    if args.all_features:
        keep = np.ones(ref['n_feat'], dtype=bool)
    else:
        keep = ref['ftype'] == 'Gene Expression'
    n_var = int(keep.sum())
    # remap: old feature index -> new index, or -1 when dropped
    remap = np.full(ref['n_feat'], -1, dtype=np.int64)
    remap[keep] = np.arange(n_var, dtype=np.int64)
    var_names = ref['names'][keep]

    n_obs_total = sum(meta[s]['n_cells'] for s in samples)
    print(f"\n  features kept: {n_var:,} of {ref['n_feat']:,}"
          f"{'' if args.all_features else ' (Gene Expression only)'}")
    print(f"  total cells  : {n_obs_total:,}")
    print(f"  upper-bound nnz: {sum(meta[s]['nnz'] for s in samples):,} "
          f"(before the feature filter)")

    # ── output store ────────────────────────────────────────────────────────
    zpath = f"{args.output_dir}/stacked.zarr"
    if os.path.exists(zpath):
        sys.exit(f"ERROR: {zpath} already exists -- refusing to overwrite")
    zg = zarr.open_group(zpath, mode='w', zarr_format=2)

    # X as an anndata csr_matrix group, filled by appending. Resizable arrays
    # mean the filtered nnz does not have to be known up front, which would
    # otherwise need a full extra read of every indices array.
    xg = zg.create_group('X')
    xg.attrs['encoding-type'] = 'csr_matrix'
    xg.attrs['encoding-version'] = '0.1.0'
    xg.attrs['shape'] = [n_obs_total, n_var]
    z_data = xg.create_dataset('data', shape=(0,), chunks=(1 << 20,),
                               dtype='<f4')
    z_ind = xg.create_dataset('indices', shape=(0,), chunks=(1 << 20,),
                              dtype='<i8')
    z_ptr = xg.create_dataset('indptr', shape=(1,), chunks=(1 << 20,),
                              dtype='<i8')
    z_ptr[0] = 0

    # ── pass 2: stream ──────────────────────────────────────────────────────
    obs_frames, rows = [], []
    nnz_written = 0
    print(f"\n[stream] chunk_size={args.chunk_size:,}")

    for s in samples:
        m = meta[s]
        h5 = f"{root}/{s}/cell_feature_matrix.h5"
        ts = time.time()
        kept_here = 0

        with h5py.File(h5, 'r') as f:
            g = h5_group(f)
            indptr_all = np.asarray(g['indptr'][:], dtype=np.int64)
            barcodes = decode(g['barcodes'][:])

            n_chunks = int(np.ceil(m['n_cells'] / args.chunk_size))
            for ch in range(n_chunks):
                a = ch * args.chunk_size
                b = min(a + args.chunk_size, m['n_cells'])
                lo, hi = int(indptr_all[a]), int(indptr_all[b])

                d = np.asarray(g['data'][lo:hi], dtype=np.float32)
                idx = np.asarray(g['indices'][lo:hi], dtype=np.int64)
                ptr = (indptr_all[a:b + 1] - lo).astype(np.int64)

                if not args.all_features:
                    # Drop filtered features and renumber. Row boundaries have
                    # to be rebuilt from per-row surviving counts, which is why
                    # this goes through a csr_matrix rather than editing ptr.
                    blk = sp.csr_matrix((d, idx, ptr), shape=(b - a, ref['n_feat']))
                    blk = blk[:, keep]
                    blk.sort_indices()
                    d, idx = blk.data, blk.indices.astype(np.int64)
                    ptr = blk.indptr.astype(np.int64)
                    d = d.astype(np.float32, copy=False)

                z_data.append(d)
                z_ind.append(idx)
                # int64 explicitly: scipy hands back an int32 indptr for
                # any block under 2**31 nonzeros, so adding the running total
                # overflows once the CUMULATIVE nnz passes 2**31. That killed
                # job 994314 at 2,213,914,150.
                z_ptr.append(ptr[1:].astype(np.int64) + np.int64(nnz_written))
                nnz_written += int(ptr[-1])
                kept_here += int(ptr[-1])

                if (ch + 1) % 10 == 0 or ch == n_chunks - 1:
                    print(f"  {s:<28} chunk {ch+1}/{n_chunks}  "
                          f"cells {a:,}-{b:,}  nnz so far {nnz_written:,}  "
                          f"({fmt(time.time()-ts)})")

        # obs for this sample
        cdf = pd.read_csv(f"{root}/{s}/cells.csv.gz", compression='gzip')
        cdf = cdf.set_index('cell_id')
        cdf = cdf.reindex(barcodes)          # align to matrix column order
        if cdf.isnull().all(axis=1).any():
            sys.exit(f"ERROR: {s} has barcodes absent from cells.csv.gz")
        cdf['sample'] = s
        cdf.index = pd.Index([f"{bc}{args.index_unique}{s}" for bc in barcodes])
        obs_frames.append(cdf)
        rows.append(dict(sample=s, n_cells=m['n_cells'], nnz_kept=kept_here,
                         nnz_raw=m['nnz'], seconds=round(time.time()-ts, 1)))
        print(f"  {s:<28} done  nnz_kept={kept_here:,}  ({fmt(time.time()-ts)})")

    if z_ptr.shape[0] != n_obs_total + 1:
        sys.exit(f"ERROR: indptr length {z_ptr.shape[0]} != n_obs+1 "
                 f"({n_obs_total+1})")

    # ── obs / var / obsm ────────────────────────────────────────────────────
    print("\n[obs/var]")
    obs = pd.concat(obs_frames, axis=0)
    obs['sample'] = pd.Categorical(obs['sample'], categories=samples)
    for c in obs.columns:
        if obs[c].dtype == object and c != 'sample':
            obs[c] = pd.Categorical(obs[c])
    if {'x_centroid', 'y_centroid'}.issubset(obs.columns):
        obs['spatial_x'] = obs['x_centroid'].astype(np.float32)
        obs['spatial_y'] = obs['y_centroid'].astype(np.float32)
        spatial = obs[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float32)
    else:
        spatial = None

    write_elem(zg, 'obs', obs)
    write_elem(zg, 'var', pd.DataFrame(index=pd.Index(var_names, name=None)))
    if spatial is not None:
        zg.create_group('obsm')
        write_elem(zg['obsm'], 'spatial', spatial)
        print(f"  obsm/spatial {spatial.shape}")
    print(f"  obs {obs.shape}  var {(n_var,)}")

    with open(f"{args.output_dir}/samples.txt", 'w') as f:
        f.write('\n'.join(samples) + '\n')
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{args.output_dir}/merge_summary.csv", index=False)

    print(f"\n[done] {zpath}")
    print(f"  X: ({n_obs_total:,}, {n_var:,})  nnz={nnz_written:,}  "
          f"dtype=float32  indices=int64")
    print(summary.to_string(index=False))
    print(f"\n  RAW COUNTS -- normalise before running the post_pca pipeline.")
    print(f"\nTotal: {fmt(time.time()-t0)}")
