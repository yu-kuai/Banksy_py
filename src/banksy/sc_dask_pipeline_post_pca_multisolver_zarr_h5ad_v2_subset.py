"""
MULTI-SOLVER variant of sc_dask_pipeline_post_pca_linop_zarr_h5ad_v2_subset.py.

Identical in every respect except that the PCA backend is selectable.
run_pca_asymmetric already accepted `solver`, `dtype` and `zarr_dir`; the
linop-only version hardcoded solver='linop_cpu' at its call site, so this
version just exposes them:

  --solver {linop_cpu, linop_gpu, linop_gpu_propack, dask_cpu, dask_gpu}
           default linop_cpu -- identical behaviour to the original
  --dtype  {float32, float64}      GPU solvers only (ignored by CPU solvers)
  --banksy_zarr_dir <dir>          dask_cpu only; where the lazy BANKSY matrix
                                   is staged

Solver notes, from the measured benchmarks (banksy_pca_solver_benchmarks.md):
  linop_cpu          matrix-free svds(propack). Lowest memory, the default.
                     Needs float64 at scale -- see --fp64.
  linop_gpu          matrix-free on GPU; VRAM-bound.
  linop_gpu_propack  as linop_gpu but forces the propack path.
  dask_cpu           materialises a LAZY dask BANKSY matrix, then
                     sc.pp.pca(covariance_eigh). Does NOT scale flat in
                     practice: 194 GB at 3 samples vs 28 GB for linop at
                     12.6M cells. Wants --banksy_zarr_dir.
  dask_gpu           as dask_cpu on GPU.

PRECISION: --fp64 (default on) casts X to float64 and governs the CPU propack
path, where single precision cannot meet svds' machine-epsilon tolerance at
millions of cells. The GPU solvers take their precision from --dtype instead,
so on those two the useful knob is --dtype float64, and --fp64 only costs
memory. The script warns when that combination is requested.

sc_dask_pipeline_post_pca_linop_zarr_h5ad_v2_subset.py

Like v2 but supports subsetting cells AFTER W (spatial graph) is built,
so the neighborhood context is always computed on all cells.

New args vs v2:
  --input_dir        source stacked.zarr (all cells); defaults to --output_dir
  --subset_indices   .npy file with row indices (into input_dir order) to keep
  --n_hvg_self       HVGs for the SELF half only, all genes for the nbr half
  --hvg_flavor       seurat_v3 (default, raw counts) / seurat / cell_ranger
  --hvg_counts_layer zarr layer with raw counts, for seurat_v3
  --hvg_subsample    compute HVGs on N cells instead of all
  --fp64             cast X to float64 before PCA (double-precision PROPACK).
                     DEFAULT ON since 2026-09-18; pass --fp32 to opt out.
  --save_h5ad        also write save_dir/stacked.h5ad; OFF by default
  --h5ad_name        filename for --save_h5ad (default stacked.h5ad)

Phase 0 (new): create output_dir/stacked.zarr with subset cells
Phase 2: build W on ALL cells, then subset adata_sorted + W before linop PCA
Phase 6b (new, optional): write stacked.h5ad
Phases 3-7 otherwise unchanged from v2

--n_hvg_self drives the asymmetric BANKSY case documented at
banksy_pca_asym.py:20 — genes_self=HVGs, genes_nbr=None. The self half then
carries only variable genes while H0 (= W @ X_nbr) still averages the full
panel, so neighbourhood context is not thinned by the HVG cut. With
n_hvg_self=2000 on a 5084-gene panel the BANKSY matrix is 2000 + 5084 = 7084
features instead of the symmetric 10168.

seurat_v3 is defined on RAW counts, so it is read from
layers/<--hvg_counts_layer> in the output zarr (Phase 0 copies that layer
across) rather than from X, which is log-normalised. Note
benchmark_banksy_pca_3197a3.py runs seurat_v3 on log1p X and scanpy warns
"expects raw count data" — that ranking is computed on the wrong scale.

Every new arg defaults to off/None, so the existing callers
(submit_linop_epi_subset, _epi_after_qc_subset, _epi_after_qc3_subset,
_epi_tumoronly_subset, _total_counts_25_subset, _tc25_epi_subset,
_tc25_epi2_subset) keep producing exactly what they produced before.

The h5ad written by --save_h5ad is the same object
sc_dask_pipeline_post_pca_linop_zarr_h5ad_v2_rapids.py writes: obs + var +
obsm + obsp + uns, and no X (expression stays in stacked.zarr). So downstream
scripts that take an h5ad for labels/embeddings and read expression from the
zarr — plot_umap_metagenes_stacked_zarr.py, run_pydeseq2_from_zarr.py,
sc_plot_metagene_scatter.py — work against it directly, without having to run
the rapids stage first.

--save_h5ad defaults to OFF so that the existing callers of this script
(submit_linop_epi_subset, _epi_after_qc_subset, _epi_after_qc3_subset,
_epi_tumoronly_subset) keep producing exactly what they produced before.
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import anndata as ad
import harmonypy as hm
import scanpy as sc
import zarr

from anndata._io.specs import read_elem, write_elem

pd.options.mode.string_storage = "python"
pd.options.future.infer_string = False

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(_SCRIPTS_DIR))

from banksy_pca_multi import prepare_multi_adata
from banksy_pca_asym import run_pca_asymmetric


def format_elapsed(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def write_to_zarr(zarr_path, key, value):
    z = zarr.open_group(zarr_path, mode='a', zarr_format=2)
    write_elem(z, key, value)
    print(f"  → {os.path.basename(zarr_path)}/{key}")


def _fix_string_dtypes(adata):
    """Make obs writable by h5ad.

    obs comes back from read_elem() on the zarr, which under numpy 2 / pandas 3
    can hand back Arrow-backed strings that h5ad cannot serialize. Convert those
    and the index to plain str. Same fix as the rapids script.
    """
    for col in adata.obs.columns:
        if hasattr(adata.obs[col], 'array') and isinstance(
            adata.obs[col].array, pd.arrays.ArrowStringArray
        ):
            adata.obs[col] = adata.obs[col].astype(str)
    if hasattr(adata.obs.index, 'dtype'):
        adata.obs.index = adata.obs.index.astype(str)


def write_h5ad_checkpoint(adata, h5ad_path):
    """Write adata to h5ad — obs + var + obsm + obsp + uns, no X.

    Mirrors write_h5ad_checkpoint() in
    sc_dask_pipeline_post_pca_linop_zarr_h5ad_v2_rapids.py, minus the cupy->numpy
    conversion, since this script never puts anything on the GPU.
    """
    _fix_string_dtypes(adata)
    adata.write_h5ad(h5ad_path)
    size_gb = os.path.getsize(h5ad_path) / 1e9
    print(f"  → {os.path.basename(h5ad_path)}  ({size_gb:.2f} GB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('--input_dir', default=None,
                        help='Source stacked.zarr dir (all cells). Defaults to --output_dir.')
    parser.add_argument('--subset_indices', default=None,
                        help='.npy of row indices (into input_dir order) to keep after W build')
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--harmony_cols', default='sample')
    parser.add_argument('--no_harmony', action='store_true')
    parser.add_argument('--harmony_theta', type=float, default=2.0)
    parser.add_argument('--n_neighbors', type=int, default=30)
    parser.add_argument('--metric', default='cosine')
    parser.add_argument('--min_dist', type=float, default=0.3)
    parser.add_argument('--cluster_n_neighbors', type=int, default=20)
    parser.add_argument('--cluster_metric', default='euclidean')
    parser.add_argument('--cluster_n_pcs', type=int, default=10)
    parser.add_argument('--n_comps', type=int, default=50)
    parser.add_argument('--solver', default='linop_cpu',
                        choices=['linop_cpu', 'linop_gpu',
                                 'linop_gpu_propack', 'dask_cpu', 'dask_gpu'],
                        help='PCA backend. Default linop_cpu, which reproduces '
                             'sc_dask_pipeline_post_pca_linop_zarr_h5ad_'
                             'v2_subset.py exactly. See the module docstring '
                             'for the memory characteristics of each.')
    parser.add_argument('--dtype', default='float32',
                        choices=['float32', 'float64'],
                        help='working precision for the GPU solvers only; the '
                             'CPU solvers take theirs from X (see --fp64). '
                             'Default float32.')
    parser.add_argument('--banksy_zarr_dir', default=None,
                        help='where dask_cpu stages its lazy BANKSY matrix. '
                             'Ignored by every other solver.')
    parser.add_argument('--leiden_resolution', type=float, default=0.6)
    parser.add_argument('--leiden_graph', default='umap',
                        choices=['umap', 'cluster'],
                        help="which neighbour graph Leiden clusters on. DEFAULT CHANGED 2026-09-17 from 'cluster' to 'umap'. 'umap' is the k=30 / 20-PC / cosine graph that also positions the UMAP, so clusters look coherent in the embedding and in the density grid. 'cluster' is the k=20 / 10-PC / euclidean graph, which every run before that date used -- pass it explicitly to reproduce an older result. Each run's params_post_pca*.json records the value used.")
    parser.add_argument('--num_neighbours', type=int, default=15)
    parser.add_argument('--nbr_weight_decay', default='scaled_gaussian')
    parser.add_argument('--zscore', default='per_sample',
                        choices=['global', 'per_sample'])
    parser.add_argument('--lambda_val', type=float, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--n_hvg_self', type=int, default=None,
                        help='Use only this many HVGs for the SELF half of the '
                             'BANKSY matrix, keeping ALL genes for the '
                             'neighbourhood half. This is the '
                             'genes_self=HVGs / genes_nbr=None case documented '
                             'at banksy_pca_asym.py:20. Default None = all '
                             'genes on both halves (symmetric), i.e. unchanged '
                             'behaviour.')
    parser.add_argument('--hvg_flavor', default='seurat_v3',
                        choices=['seurat_v3', 'seurat', 'cell_ranger'],
                        help="HVG flavour. seurat_v3 (default) needs RAW "
                             "counts and is read from --hvg_counts_layer; "
                             "seurat/cell_ranger expect log data and use X.")
    parser.add_argument('--hvg_counts_layer', default='counts',
                        help='zarr layer holding raw counts, for seurat_v3')
    parser.add_argument('--hvg_subsample', type=int, default=0,
                        help='compute HVGs on this many randomly chosen cells '
                             '(seed 0) instead of all of them. 0 = all cells.')
    parser.add_argument('--fp64', action='store_true', default=True,
                        help='cast X to float64 before the linop PCA, forcing '
                             'double-precision PROPACK. DEFAULT ON since '
                             '2026-09-18: float32 PROPACK segfaulted at 11.9M '
                             'cells (job 913949) and failed to converge at '
                             '4.3M x 10,168 (job 971750), so single precision '
                             'is not a safe default at this scale. Costs one '
                             'reallocation of X.data and doubles Lanczos '
                             'vector memory.')
    parser.add_argument('--fp32', dest='fp64', action='store_false',
                        help='opt out of --fp64 and keep X in float32. Only for '
                             'runs where Lanczos memory is the binding '
                             'constraint (the 100M-cell test), since float32 '
                             'PROPACK is unreliable above ~10M cells.')
    parser.add_argument('--save_h5ad', action='store_true',
                        help='Also write save_dir/stacked.h5ad (obs + var + obsm '
                             '+ obsp + uns, no X), the same object the _rapids '
                             'script writes. Off by default so existing callers '
                             'keep producing only stacked.zarr.')
    parser.add_argument('--h5ad_name', default='stacked.h5ad',
                        help='Filename for --save_h5ad, written into --save_dir.')
    args = parser.parse_args()

    input_dir  = args.input_dir if args.input_dir is not None else args.output_dir
    output_dir = args.output_dir
    save_dir   = args.save_dir if args.save_dir is not None else output_dir
    src_zarr_path = f"{input_dir}/stacked.zarr"
    out_zarr_path = f"{output_dir}/stacked.zarr"
    harmony_cols  = [c.strip() for c in args.harmony_cols.split(',')]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir,   exist_ok=True)

    params = {
        "script": "sc_dask_pipeline_post_pca_linop_zarr_h5ad_v2_subset.py",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "args": {**vars(args), "input_dir_resolved": input_dir,
                 "save_dir_resolved": save_dir, "harmony_cols_resolved": harmony_cols},
    }
    with open(f"{save_dir}/params_post_pca_linop_v2_subset.json", 'w') as _f:
        json.dump(params, _f, indent=2)
    print(f"Saved params_post_pca_linop_v2_subset.json")

    script_start = time.time()

    # ============================================================================
    # Phase 0: Create output stacked.zarr with subset cells (if subsetting)
    # ============================================================================
    subset_indices = None
    if args.subset_indices is not None:
        subset_indices = np.load(args.subset_indices)
        print(f"\n{'='*60}")
        print(f"[Phase 0] Creating subset stacked.zarr ({len(subset_indices):,} cells)")
        print(f"{'='*60}")
        phase0_start = time.time()

        z_src = zarr.open_group(src_zarr_path, mode='r', zarr_format=2)

        obs_all = read_elem(z_src['obs'])
        var_all = read_elem(z_src['var'])
        obs_sub = obs_all.iloc[subset_indices].copy()

        print(f"  Subsetting X (normalized) ...")
        X_all = read_elem(z_src['X'])
        X_sub = X_all[subset_indices]
        del X_all

        spatial_sub = None
        if 'obsm' in z_src and 'spatial' in z_src['obsm']:
            spatial_all = read_elem(z_src['obsm/spatial'])
            spatial_sub = spatial_all[subset_indices]
            del spatial_all

        counts_sub = None
        if 'layers' in z_src and 'counts' in z_src['layers']:
            print(f"  Subsetting counts layer ...")
            indptr  = np.asarray(z_src['layers']['counts']['indptr'])
            indices = np.asarray(z_src['layers']['counts']['indices'])
            data    = np.asarray(z_src['layers']['counts']['data'])
            n_cols  = obs_all.shape[1] if False else X_sub.shape[1]
            csr_all = sp.csr_matrix((data, indices, indptr),
                                    shape=(len(obs_all), X_sub.shape[1]))
            counts_sub = csr_all[subset_indices]
            del csr_all, indptr, indices, data

        del obs_all, z_src

        print(f"  Writing output stacked.zarr ...")
        z_out = zarr.open_group(out_zarr_path, mode='w', zarr_format=2)
        write_elem(z_out, 'obs', obs_sub)
        write_elem(z_out, 'var', var_all)
        write_elem(z_out, 'X',   X_sub)
        if spatial_sub is not None:
            write_elem(z_out, 'obsm/spatial', spatial_sub)
        if counts_sub is not None:
            write_elem(z_out, 'layers/counts', counts_sub)
        del X_sub, spatial_sub, counts_sub

        phase0_elapsed = time.time() - phase0_start
        print(f"[Phase 0 complete] {format_elapsed(phase0_elapsed)}")
    else:
        phase0_elapsed = 0.0

    stacked_zarr_path = out_zarr_path

    # ============================================================================
    # Phase 1: Load stacked.zarr (all cells, from input_dir)
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"[Phase 1] Loading stacked.zarr (all cells from {input_dir})")
    print(f"{'='*60}")
    phase1_start = time.time()

    z_load   = zarr.open_group(src_zarr_path, mode='r', zarr_format=2)
    obs_load = read_elem(z_load['obs'])
    var_load = read_elem(z_load['var'])
    X_load   = read_elem(z_load['X'])
    adata    = ad.AnnData(X=X_load, obs=obs_load, var=var_load)
    if 'obsm' in z_load and 'spatial' in z_load['obsm']:
        adata.obsm['spatial'] = read_elem(z_load['obsm/spatial'])
    del z_load, obs_load, var_load, X_load
    print(f"  shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)}")

    if 'sample' not in adata.obs.columns:
        adata.obs['sample'] = adata.obs_names.str.rsplit('_', n=1).str[-1]
        print(f"  Derived 'sample' from obs_names")

    if args.lambda_val is not None:
        lambda_val = args.lambda_val
        print(f"  lambda_val: {lambda_val}  (from --lambda_val)")
    else:
        lam_csv = pd.read_csv(f"{input_dir}/sample_lambda_vals.csv")
        unique_lambdas = lam_csv['lambda'].unique()
        assert len(unique_lambdas) == 1, \
            f"Multiple lambda values found: {unique_lambdas}. Use --lambda_val."
        lambda_val = float(unique_lambdas[0])
        print(f"  lambda_val: {lambda_val}  (from sample_lambda_vals.csv)")

    obs_names_original = adata.obs_names.tolist()

    phase1_elapsed = time.time() - phase1_start
    print(f"[Phase 1 complete] {format_elapsed(phase1_elapsed)}")

    # ============================================================================
    # Phase 2: Build W (all cells) + subset + linop PCA
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"[Phase 2] Build W (all cells) + linop PCA"
          f"  n_comps={args.n_comps}  zscore={args.zscore}  lambda={lambda_val}")
    print(f"{'='*60}")
    phase2_start = time.time()

    z_stacked  = zarr.open_group(stacked_zarr_path, mode='r', zarr_format=2)
    pca_exists = args.resume and 'obsm' in z_stacked and 'X_pca' in z_stacked['obsm']

    if pca_exists:
        print(f"  [RESUME] Loading X_pca from stacked.zarr")
        X_pca        = np.asarray(read_elem(z_stacked['obsm/X_pca']))
        obs_metadata = read_elem(z_stacked['obs'])
        var_metadata = read_elem(z_stacked['var'])
        print(f"  X_pca: {X_pca.shape}")
    else:
        print(f"  Building spatial graph on ALL {adata.n_obs:,} cells "
              f"(k={args.num_neighbours}, decay={args.nbr_weight_decay}) ...")
        t_w = time.time()
        adata_sorted, W = prepare_multi_adata(
            adata,
            normalize=False,
            n_hvg=None,
            batch_key='sample',
            num_neighbours=args.num_neighbours,
            nbr_weight_decay=args.nbr_weight_decay,
        )
        print(f"  W built  ({format_elapsed(time.time() - t_w)})"
              f"  shape={W.shape}  nnz={W.nnz}")

        name_to_orig = {name: i for i, name in enumerate(obs_names_original)}
        sorted_to_orig = np.array([name_to_orig[n] for n in adata_sorted.obs_names])

        # ── subset to target cells after W is built ───────────────────────────
        if subset_indices is not None:
            print(f"  Subsetting to {len(subset_indices):,} target cells "
                  f"(W computed on all cells — neighborhood preserved) ...")
            orig_to_sorted = np.empty(len(obs_names_original), dtype=np.intp)
            orig_to_sorted[sorted_to_orig] = np.arange(len(sorted_to_orig))

            epi_sorted_idx  = orig_to_sorted[subset_indices]
            epi_sorted_mask = np.zeros(len(adata_sorted), dtype=bool)
            epi_sorted_mask[epi_sorted_idx] = True

            adata_sorted = adata_sorted[epi_sorted_mask].copy()
            W            = W[epi_sorted_mask, :][:, epi_sorted_mask]
            sorted_to_orig_sub = sorted_to_orig[epi_sorted_idx]

            # Map sorted-epi position → output position (subset_indices order)
            orig_to_out = {int(orig): out for out, orig in enumerate(subset_indices)}
            sorted_to_out = np.array([orig_to_out[int(sorted_to_orig_sub[i])]
                                      for i in range(len(epi_sorted_idx))])
            n_out = len(subset_indices)
            print(f"  Subset done — adata_sorted: {adata_sorted.shape}  W: {W.shape}")
        else:
            sorted_to_out = sorted_to_orig
            n_out         = len(obs_names_original)

        del adata

        if args.fp64:
            # Force the double-precision PROPACK path.
            #
            # banksy_pca_asym.py:686 picks fp_dtype from X's dtype:
            #   _fp = np.float32 if X_csr_self.dtype == np.float32 else np.float64
            # so a float32 zarr (test_16s_linop/stacked.zarr X is <f4) makes the
            # LinearOperator float32 and runs svds(solver='propack') in SINGLE
            # precision. That combination segfaulted at 11.9M cells (job 913949,
            # Exit 139, only 124 GB of 500 G used, so not an OOM). The same
            # solver at 12.6M cells succeeded on 2026-08-13 (job 863401), before
            # fp_dtype existed, i.e. in double precision.
            #
            # Casting only .data leaves indices/indptr alone, so this costs one
            # reallocation of the values array rather than a full sparse copy.
            # NOT a fix for the 100M-cell test: float32 exists there to halve
            # Lanczos memory, so that case needs a different solver instead.
            import scipy.sparse as _sp
            if _sp.issparse(adata_sorted.X) and adata_sorted.X.dtype != np.float64:
                print(f"  --fp64: casting X {adata_sorted.X.dtype} -> float64 "
                      f"(nnz={adata_sorted.X.nnz:,})")
                adata_sorted.X = adata_sorted.X.tocsr()
                adata_sorted.X.data = adata_sorted.X.data.astype(np.float64)
                print(f"    X dtype now {adata_sorted.X.dtype}")

        # ── optional: HVG subset for the SELF half only ──────────────────────
        # genes_self=HVGs, genes_nbr=None — the asymmetric case documented at
        # banksy_pca_asym.py:20. The self half then carries only variable
        # genes while H0 (= W @ X_nbr) still averages the full panel, so
        # neighbourhood context is not thinned by the HVG cut.
        genes_self = None
        if args.n_hvg_self:
            print(f"\n  Selecting {args.n_hvg_self} HVGs for the self half "
                  f"(flavor={args.hvg_flavor}) ...")
            t_hvg = time.time()
            if args.hvg_flavor == 'seurat_v3':
                # seurat_v3 is defined on RAW counts. adata_sorted.X here is
                # log-normalised, so counts are read from the zarr layer
                # instead — Phase 0 copies layers/counts into the subset zarr.
                # (benchmark_banksy_pca_3197a3.py runs seurat_v3 on log1p X and
                # scanpy warns "expects raw count data"; that ranking is on the
                # wrong scale.)
                z_hvg = zarr.open_group(stacked_zarr_path, mode='r',
                                        zarr_format=2)
                if 'layers' not in z_hvg or args.hvg_counts_layer not in z_hvg['layers']:
                    raise RuntimeError(
                        f"layers/{args.hvg_counts_layer} not in "
                        f"{stacked_zarr_path}; needed for "
                        f"--hvg_flavor seurat_v3")
                X_hvg = read_elem(z_hvg[f'layers/{args.hvg_counts_layer}'])
                print(f"    read layers/{args.hvg_counts_layer}: "
                      f"{X_hvg.shape}  nnz={X_hvg.nnz:,}  dtype={X_hvg.dtype}")
            else:
                X_hvg = adata_sorted.X

            # cell order is irrelevant here — only a gene list comes out
            ad_hvg = ad.AnnData(X=X_hvg, var=adata_sorted.var.copy())
            if args.hvg_subsample and args.hvg_subsample < ad_hvg.n_obs:
                rs = np.random.default_rng(0).choice(
                    ad_hvg.n_obs, size=args.hvg_subsample, replace=False)
                ad_hvg = ad_hvg[np.sort(rs)].copy()
                print(f"    subsampled to {ad_hvg.n_obs:,} cells for HVG")
            sc.pp.highly_variable_genes(ad_hvg, n_top_genes=args.n_hvg_self,
                                        flavor=args.hvg_flavor)
            genes_self = ad_hvg.var_names[ad_hvg.var.highly_variable].tolist()
            del ad_hvg, X_hvg
            gc.collect()

            hvg_path = f"{save_dir}/hvg_self_genes.txt"
            with open(hvg_path, 'w') as _f:
                _f.write('\n'.join(genes_self) + '\n')
            print(f"    {len(genes_self)} HVGs selected "
                  f"({format_elapsed(time.time() - t_hvg)})")
            print(f"    first 15: {genes_self[:15]}")
            print(f"    → {hvg_path}")
            print(f"    BANKSY matrix will be "
                  f"{len(genes_self)} self + {adata_sorted.n_vars} nbr = "
                  f"{len(genes_self) + adata_sorted.n_vars} features "
                  f"(symmetric would be {2 * adata_sorted.n_vars})")

        print(f"  Running linop PCA ...")
        t_pca = time.time()
        # Solver/precision interaction. --fp64 casts X, which is what the CPU
        # propack path needs; the GPU paths read --dtype instead, so there
        # --fp64 buys nothing and doubles the matrix in host memory.
        if args.solver in ('linop_gpu', 'linop_gpu_propack', 'dask_gpu'):
            if args.fp64:
                print(f"  NOTE: solver={args.solver} takes precision from "
                      f"--dtype ({args.dtype}); --fp64 only costs host memory "
                      f"here. Pass --fp32 --dtype float64 for GPU double "
                      f"precision.")
        elif args.dtype != 'float32':
            print(f"  NOTE: --dtype {args.dtype} is ignored by "
                  f"solver={args.solver}; CPU precision comes from --fp64.")
        if args.solver == 'dask_cpu' and not args.banksy_zarr_dir:
            print("  NOTE: solver=dask_cpu without --banksy_zarr_dir; the "
                  "BANKSY matrix will be held in memory rather than staged.")
        print(f"  solver={args.solver}  dtype={args.dtype}  fp64={args.fp64}")

        embeddings, ev, evr = run_pca_asymmetric(
            adata_sorted, W,
            genes_self=genes_self,
            genes_nbr=None,
            lambda_val=lambda_val,
            n_comps=args.n_comps,
            zscore=args.zscore,
            batch_key='sample',
            solver=args.solver,
            dtype=args.dtype,
            zarr_dir=args.banksy_zarr_dir,
        )
        print(f"  PCA done  ({format_elapsed(time.time() - t_pca)})"
              f"  shape={embeddings.shape}")
        print(f"  EVR first 10: {evr[:10]}")
        print(f"  Total variance explained: {evr.sum():.4f}")

        X_pca = np.empty_like(embeddings)
        X_pca[sorted_to_out] = embeddings

        obs_metadata = read_elem(zarr.open_group(stacked_zarr_path, mode='r', zarr_format=2)['obs'])
        var_metadata = read_elem(zarr.open_group(stacked_zarr_path, mode='r', zarr_format=2)['var'])

        print("\n  Writing PCA results to stacked.zarr ...")
        write_to_zarr(stacked_zarr_path, 'obsm/X_pca', X_pca)
        write_to_zarr(stacked_zarr_path, 'uns/pca', {'variance': ev, 'variance_ratio': evr})

        del adata_sorted, W, embeddings

    phase2_elapsed = time.time() - phase2_start
    print(f"[Phase 2 complete] {format_elapsed(phase2_elapsed)}")

    # ============================================================================
    # Phase 3: Harmony
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"[Phase 3] Harmony  (cols={harmony_cols}  theta={args.harmony_theta})")
    print(f"{'='*60}")
    phase3_start = time.time()

    z_stacked      = zarr.open_group(stacked_zarr_path, mode='r', zarr_format=2)
    harmony_exists = args.resume and 'obsm' in z_stacked and 'X_pca_harmony' in z_stacked['obsm']

    if harmony_exists:
        print(f"  [RESUME] Loading X_pca_harmony from stacked.zarr")
        X_pca_harmony = np.asarray(read_elem(z_stacked['obsm/X_pca_harmony']))
    elif args.no_harmony:
        print("  --no_harmony: using raw PCA as X_pca_harmony")
        X_pca_harmony = X_pca
        write_to_zarr(stacked_zarr_path, 'obsm/X_pca_harmony', X_pca_harmony)
    else:
        for col in obs_metadata.columns:
            if obs_metadata[col].dtype == object:
                obs_metadata[col] = pd.Categorical(obs_metadata[col])
        print(f"  Input: {X_pca.shape}  dtype={X_pca.dtype}")
        ho = hm.run_harmony(
            X_pca.astype(np.float32), obs_metadata,
            vars_use=harmony_cols,
            max_iter_harmony=20,
            verbose=True,
            random_state=42,
            theta=args.harmony_theta,
        )
        X_pca_harmony = np.asarray(ho.Z_corr)
        print(f"  Harmony output: {X_pca_harmony.shape}")
        write_to_zarr(stacked_zarr_path, 'obsm/X_pca_harmony', X_pca_harmony)

    phase3_elapsed = time.time() - phase3_start
    print(f"[Phase 3 complete] {format_elapsed(phase3_elapsed)}")

    # ============================================================================
    # Phase 4: Neighbors
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"[Phase 4] Neighbors")
    print(f"{'='*60}")
    phase4_start = time.time()

    adata_umap = ad.AnnData(obs=obs_metadata.copy(), var=var_metadata.copy())
    adata_umap.obsm['X_pca_harmony'] = X_pca_harmony
    if 'sample' not in adata_umap.obs.columns:
        adata_umap.obs['sample'] = adata_umap.obs.index.str.rsplit('_', n=1).str[-1]

    z_stacked  = zarr.open_group(stacked_zarr_path, mode='r', zarr_format=2)
    nbr_exists = (args.resume and
                  'obsp' in z_stacked and
                  'neighbors_umap_connectivities'    in z_stacked['obsp'] and
                  'neighbors_cluster_connectivities' in z_stacked['obsp'])

    if nbr_exists:
        print(f"  [RESUME] Loading neighbor matrices from stacked.zarr")
        adata_umap.obsp['neighbors_umap_connectivities']    = read_elem(z_stacked['obsp/neighbors_umap_connectivities'])
        adata_umap.obsp['neighbors_umap_distances']         = read_elem(z_stacked['obsp/neighbors_umap_distances'])
        adata_umap.obsp['neighbors_cluster_connectivities'] = read_elem(z_stacked['obsp/neighbors_cluster_connectivities'])
        adata_umap.obsp['neighbors_cluster_distances']      = read_elem(z_stacked['obsp/neighbors_cluster_distances'])
        adata_umap.uns['neighbors_umap']    = read_elem(z_stacked['uns/neighbors_umap'])
        adata_umap.uns['neighbors_cluster'] = read_elem(z_stacked['uns/neighbors_cluster'])
    else:
        print(f"  UMAP graph:    n_neighbors={args.n_neighbors}, n_pcs=20, metric={args.metric}")
        sc.pp.neighbors(adata_umap, use_rep='X_pca_harmony', n_neighbors=args.n_neighbors,
                        n_pcs=20, metric=args.metric, random_state=42, key_added='neighbors_umap')
        print(f"  Cluster graph: n_neighbors={args.cluster_n_neighbors}, "
              f"n_pcs={args.cluster_n_pcs}, metric={args.cluster_metric}")
        sc.pp.neighbors(adata_umap, use_rep='X_pca_harmony', n_neighbors=args.cluster_n_neighbors,
                        n_pcs=args.cluster_n_pcs, metric=args.cluster_metric,
                        random_state=42, key_added='neighbors_cluster')

        print("\n  Writing neighbor results to stacked.zarr ...")
        write_to_zarr(stacked_zarr_path, 'obsp/neighbors_umap_connectivities',
                      adata_umap.obsp['neighbors_umap_connectivities'])
        write_to_zarr(stacked_zarr_path, 'obsp/neighbors_umap_distances',
                      adata_umap.obsp['neighbors_umap_distances'])
        write_to_zarr(stacked_zarr_path, 'obsp/neighbors_cluster_connectivities',
                      adata_umap.obsp['neighbors_cluster_connectivities'])
        write_to_zarr(stacked_zarr_path, 'obsp/neighbors_cluster_distances',
                      adata_umap.obsp['neighbors_cluster_distances'])
        write_to_zarr(stacked_zarr_path, 'uns/neighbors_umap',    adata_umap.uns['neighbors_umap'])
        write_to_zarr(stacked_zarr_path, 'uns/neighbors_cluster', adata_umap.uns['neighbors_cluster'])

    phase4_elapsed = time.time() - phase4_start
    print(f"[Phase 4 complete] {format_elapsed(phase4_elapsed)}")

    # ============================================================================
    # Phase 5: UMAP
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"[Phase 5] UMAP  (min_dist={args.min_dist})")
    print(f"{'='*60}")
    phase5_start = time.time()

    z_stacked   = zarr.open_group(stacked_zarr_path, mode='r', zarr_format=2)
    umap_exists = args.resume and 'obsm' in z_stacked and 'X_umap' in z_stacked['obsm']

    if umap_exists:
        print(f"  [RESUME] Loading X_umap from stacked.zarr")
        adata_umap.obsm['X_umap'] = read_elem(z_stacked['obsm/X_umap'])
    else:
        sc.tl.umap(adata_umap, neighbors_key='neighbors_umap',
                   random_state=42, min_dist=args.min_dist, spread=1.0, init_pos='random')
        print(f"  X_umap: {adata_umap.obsm['X_umap'].shape}")
        write_to_zarr(stacked_zarr_path, 'obsm/X_umap', adata_umap.obsm['X_umap'])

    phase5_elapsed = time.time() - phase5_start
    print(f"[Phase 5 complete] {format_elapsed(phase5_elapsed)}")

    # ============================================================================
    # Phase 6: Leiden
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"[Phase 6] Leiden  (resolution={args.leiden_resolution})")
    print(f"{'='*60}")
    phase6_start = time.time()

    z_stacked     = zarr.open_group(stacked_zarr_path, mode='r', zarr_format=2)
    leiden_exists = args.resume and 'leiden' in read_elem(z_stacked['obs']).columns

    if leiden_exists:
        print(f"  [RESUME] leiden column already in stacked.zarr/obs")
        adata_umap.obs['leiden'] = read_elem(z_stacked['obs'])['leiden']
    else:
        leiden_graph = f'neighbors_{args.leiden_graph}'
        print(f"  Using graph: {leiden_graph}")
        sc.tl.leiden(adata_umap, neighbors_key=leiden_graph,
                     flavor='igraph', n_iterations=2, resolution=args.leiden_resolution)
        n_clusters = adata_umap.obs['leiden'].nunique()
        print(f"  {n_clusters} clusters")

        print("\n  Updating obs in stacked.zarr with leiden ...")
        obs_df = read_elem(zarr.open_group(stacked_zarr_path, mode='r', zarr_format=2)['obs'])
        for col in obs_df.columns:
            if hasattr(obs_df[col], 'dtype') and str(obs_df[col].dtype) in ('string', 'large_string'):
                obs_df[col] = obs_df[col].astype(object)
        if obs_df.index.dtype.name in ('string', 'large_string'):
            obs_df.index = obs_df.index.astype(object)
        obs_df['leiden'] = adata_umap.obs['leiden'].values
        write_to_zarr(stacked_zarr_path, 'obs', obs_df)

    phase6_elapsed = time.time() - phase6_start
    print(f"[Phase 6 complete] {format_elapsed(phase6_elapsed)}")

    # ============================================================================
    # Phase 6b: h5ad  (optional, --save_h5ad)
    # ============================================================================
    # Written BEFORE the plots on purpose: the h5ad is the deliverable and the
    # plots are cosmetic, so a failure in Phase 7 must not cost the object.
    phase6b_elapsed = 0.0
    h5ad_path = None
    if args.save_h5ad:
        print(f"\n{'='*60}")
        print(f"[Phase 6b] Saving h5ad")
        print(f"{'='*60}")
        phase6b_start = time.time()
        h5ad_path = f"{save_dir}/{args.h5ad_name}"
        print(f"  obs:  {adata_umap.obs.shape}")
        print(f"  obsm: {list(adata_umap.obsm.keys())}")
        print(f"  obsp: {list(adata_umap.obsp.keys())}")
        write_h5ad_checkpoint(adata_umap, h5ad_path)
        phase6b_elapsed = time.time() - phase6b_start
        print(f"[Phase 6b complete] {format_elapsed(phase6b_elapsed)}")

    # ============================================================================
    # Phase 7: Plots
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"[Phase 7] Plots")
    print(f"{'='*60}")
    phase7_start = time.time()

    plt_dir = f"{save_dir}/plot"
    os.makedirs(plt_dir, exist_ok=True)

    z_stacked_r = zarr.open_group(stacked_zarr_path, mode='r', zarr_format=2)
    if 'pca' in z_stacked_r.get('uns', {}):
        pca_uns = read_elem(z_stacked_r['uns/pca'])
        evr_arr = np.asarray(pca_uns['variance_ratio'])
        cumevr  = np.cumsum(evr_arr)
        n_pcs   = len(evr_arr)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(range(1, n_pcs + 1), evr_arr * 100, color='steelblue', edgecolor='none')
        axes[0].set_xlabel('PC'); axes[0].set_ylabel('Explained variance (%)')
        axes[0].set_title('Per-PC explained variance')
        axes[1].plot(range(1, n_pcs + 1), cumevr * 100, marker='.', color='steelblue')
        axes[1].axhline(90, color='red', linestyle='--', linewidth=0.8, label='90%')
        axes[1].axhline(95, color='orange', linestyle='--', linewidth=0.8, label='95%')
        axes[1].set_xlabel('Number of PCs'); axes[1].set_ylabel('Cumulative explained variance (%)')
        axes[1].set_title('Cumulative explained variance'); axes[1].legend()
        plt.tight_layout()
        plt.savefig(f"{plt_dir}/pca_variance_explained.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved pca_variance_explained.png")

    if 'sample' in adata_umap.obs.columns:
        adata_umap.obs['sample'] = adata_umap.obs['sample'].astype('category')

    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata_umap, ax=ax, show=False, color='sample', size=0.2)
    plt.tight_layout()
    plt.savefig(f"{plt_dir}/umap_sample.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata_umap, ax=ax, show=False, color='leiden', size=0.2)
    plt.tight_layout()
    plt.savefig(f"{plt_dir}/umap_leiden.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved umap_sample.png + umap_leiden.png → {plt_dir}/")

    phase7_elapsed = time.time() - phase7_start
    print(f"[Phase 7 complete] {format_elapsed(phase7_elapsed)}")

    # ============================================================================
    # Summary
    # ============================================================================
    total_elapsed = time.time() - script_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"\nTiming Summary:")
    if subset_indices is not None:
        print(f"  Phase 0 (Create subset zarr):             {format_elapsed(phase0_elapsed)}")
    print(f"  Phase 1 (Load stacked.zarr):              {format_elapsed(phase1_elapsed)}")
    print(f"  Phase 2 (Build W + linop PCA):            {format_elapsed(phase2_elapsed)}")
    print(f"  Phase 3 (Harmony):                        {format_elapsed(phase3_elapsed)}")
    print(f"  Phase 4 (Neighbors):                      {format_elapsed(phase4_elapsed)}")
    print(f"  Phase 5 (UMAP):                           {format_elapsed(phase5_elapsed)}")
    print(f"  Phase 6 (Leiden):                         {format_elapsed(phase6_elapsed)}")
    if args.save_h5ad:
        print(f"  Phase 6b (Save h5ad):                     {format_elapsed(phase6b_elapsed)}")
    print(f"  Phase 7 (Plots):                          {format_elapsed(phase7_elapsed)}")
    print(f"  {'─'*40}")
    print(f"  Total:                                    {format_elapsed(total_elapsed)}")
    print(f"{'='*60}")
    print(f"\nAll results written to: {out_zarr_path}")
