"""
Normalise a raw-counts stacked.zarr in place so it becomes consumable by
sc_dask_pipeline_post_pca_*.

merge_10x_to_zarr_streaming.py writes RAW counts in X and no layers. The
post-PCA pipelines never normalise -- grep normalize_total/log1p in
sc_dask_pipeline_post_pca_multisolver_zarr_h5ad_v2_subset.py finds only
comments -- they assume X is already log-normalised and that raw counts live in
layers/counts for seurat_v3 HVG selection. This script closes that gap:

    layers/counts  <- X            (raw counts preserved)
    X              <- log1p(normalize_total(X, target_sum))

IDEMPOTENCY CHECK
`layers/counts` is the marker. If it exists the store has already been through
this script -- X is log-norm and raw is preserved -- so the script reports and
exits 0 rather than normalising twice. Normalising an already-log1p X would be
silent and unrecoverable without the raw data, which is exactly why the check is
on the layer's existence and not on a flag that could be written before the work
finished.

WHY ONLY X/data IS REWRITTEN
normalize_total scales each row and log1p is elementwise, so the sparsity
pattern is unchanged: indices and indptr are byte-identical before and after.
Only the values array is rewritten, in row chunks.

--target_sum
    <float>              use this value
    median_of_medians    per-sample median library size, then the median across
                         samples. Robust to one aberrantly shallow or deep
                         sample. NOTE with n=2 samples this equals the mean of
                         the two, so the robustness only matters from 3 up.
    pooled_median        median library size over all cells, ignoring sample.
                         Cell-weighted, so a large sample dominates.
A FIXED value is essential for multi-sample stores. scanpy's default
(per-cell-median, i.e. target_sum=None) makes samples non-comparable, which is
the trap 12A8 and 4F17 already hit individually with --target_sum 0.

Library size is computed from X itself, not obs['transcript_counts'], so it
reflects exactly the genes in the store (the merge keeps Gene Expression only).

Usage:
    # report the target_sum and what would happen, change nothing
    python normalize_merged_zarr.py --zarr <dir>/stacked.zarr \
        --target_sum median_of_medians --dry_run

    # do it
    python normalize_merged_zarr.py --zarr <dir>/stacked.zarr \
        --target_sum median_of_medians
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
import zarr
from anndata.io import read_elem

pd.options.mode.string_storage = "python"
pd.options.future.infer_string = False
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

p = argparse.ArgumentParser()
p.add_argument('--zarr', required=True, help='path to stacked.zarr')
p.add_argument('--target_sum', default='median_of_medians',
               help="a number, or 'median_of_medians' / 'pooled_median'")
p.add_argument('--sample_key', default='sample',
               help='obs column grouping cells into samples, for '
                    'median_of_medians')
p.add_argument('--counts_layer', default='counts',
               help='layer name to write the raw counts into. Its existence is '
                    'also the already-normalised marker.')
p.add_argument('--chunk_size', type=int, default=50_000)
p.add_argument('--dry_run', action='store_true',
               help='compute and report the target_sum, write nothing')
p.add_argument('--force', action='store_true',
               help='proceed even if the counts layer already exists. Only '
                    'safe if you know X is still raw; otherwise it double-'
                    'normalises and the original values are gone.')
args = p.parse_args()

t0 = time.time()
z = zarr.open_group(args.zarr, mode='r+', zarr_format=2)

# ── idempotency check ───────────────────────────────────────────────────────
has_layer = ('layers' in z) and (args.counts_layer in z['layers'])
print(f"{args.zarr}")
print(f"  layers/{args.counts_layer} present: {has_layer}")
if has_layer and not args.force:
    print(f"\n  ALREADY NORMALISED -- layers/{args.counts_layer} exists, so X is "
          f"log-norm and raw counts are preserved.")
    print(f"  Nothing to do. (--force overrides, but double-normalising is "
          f"unrecoverable.)")
    sys.exit(0)
if has_layer and args.force:
    print(f"  --force: proceeding despite layers/{args.counts_layer} existing")

n_obs = int(z['X'].attrs['shape'][0])
n_var = int(z['X'].attrs['shape'][1])
indptr = np.asarray(z['X/indptr'][:], dtype=np.int64)
nnz = int(z['X/data'].shape[0])
print(f"  X ({n_obs:,}, {n_var:,})  nnz={nnz:,}  dtype={z['X/data'].dtype}")

# ── pass 1: per-cell library size, streaming ────────────────────────────────
print(f"\n[pass 1] library size per cell (from X, chunk={args.chunk_size:,})")
lib = np.zeros(n_obs, dtype=np.float64)
n_chunks = int(np.ceil(n_obs / args.chunk_size))
ts = time.time()
for ch in range(n_chunks):
    a, b = ch * args.chunk_size, min((ch + 1) * args.chunk_size, n_obs)
    lo, hi = int(indptr[a]), int(indptr[b])
    d = np.asarray(z['X/data'][lo:hi], dtype=np.float64)
    # sum within each row using the indptr boundaries
    bounds = indptr[a:b + 1] - lo
    lib[a:b] = np.add.reduceat(np.append(d, 0.0), bounds[:-1]) \
        if len(d) else 0.0
    # reduceat misbehaves on empty rows; fix those explicitly
    empty = np.flatnonzero(np.diff(bounds) == 0)
    if len(empty):
        lib[a + empty] = 0.0
    if (ch + 1) % 20 == 0 or ch == n_chunks - 1:
        print(f"    chunk {ch+1}/{n_chunks} ({time.time()-ts:.0f}s)")

n_zero = int((lib == 0).sum())
if n_zero:
    print(f"  {n_zero:,} cells have zero counts; left untouched")

# ── resolve target_sum ──────────────────────────────────────────────────────
obs = read_elem(z['obs'])
if args.target_sum in ('median_of_medians', 'pooled_median'):
    if args.target_sum == 'pooled_median':
        target = float(np.median(lib[lib > 0]))
        print(f"\n[target_sum] pooled_median over all cells = {target:,.1f}")
    else:
        if args.sample_key not in obs.columns:
            sys.exit(f"ERROR: obs['{args.sample_key}'] not found; needed for "
                     f"median_of_medians. Columns: {list(obs.columns)}")
        grp = pd.Series(lib, index=obs.index).groupby(
            obs[args.sample_key].astype(str).to_numpy(), observed=True)
        per_sample = grp.apply(lambda s: float(np.median(s[s > 0])))
        target = float(np.median(per_sample.to_numpy()))
        print(f"\n[target_sum] median_of_medians")
        for s, v in per_sample.items():
            print(f"    {s:<26} median {v:>12,.1f}")
        if len(per_sample) == 2:
            print(f"    NOTE n=2 samples, so this equals the mean of the two; "
                  f"the robustness of a median only applies from n=3 up")
        print(f"    -> target_sum = {target:,.1f}")
else:
    try:
        target = float(args.target_sum)
    except ValueError:
        sys.exit(f"ERROR: --target_sum must be a number, "
                 f"'median_of_medians' or 'pooled_median'; got "
                 f"{args.target_sum!r}")
    print(f"\n[target_sum] fixed = {target:,.1f}")

if args.dry_run:
    print(f"\n--dry_run: nothing written. Would have:")
    print(f"    layers/{args.counts_layer} <- X (raw, nnz={nnz:,})")
    print(f"    X <- log1p(X / lib * {target:,.1f})")
    print(f"\nTotal: {time.time()-t0:.0f}s")
    sys.exit(0)

# ── write the raw counts layer FIRST ────────────────────────────────────────
# Order matters: if this step fails, X is still raw and the store is unchanged.
# Rewriting X before preserving the raw values would risk losing them.
print(f"\n[write] layers/{args.counts_layer} <- X (raw)")
lg = z.require_group('layers')
if args.counts_layer in lg:
    del lg[args.counts_layer]
cg = lg.create_group(args.counts_layer)
cg.attrs['encoding-type'] = 'csr_matrix'
cg.attrs['encoding-version'] = '0.1.0'
cg.attrs['shape'] = [n_obs, n_var]
for name in ('data', 'indices', 'indptr'):
    src = z['X'][name]
    dst = cg.create_dataset(name, shape=src.shape, chunks=src.chunks,
                            dtype=src.dtype)
    step = max(1, (1 << 24) // max(1, src.dtype.itemsize))
    for s in range(0, src.shape[0], step):
        e = min(s + step, src.shape[0])
        dst[s:e] = src[s:e]
    print(f"    {name}: {src.shape[0]:,} elems {src.dtype}")

# ── rewrite X/data normalised ───────────────────────────────────────────────
# indices/indptr are untouched: scaling rows and log1p do not change sparsity.
print(f"\n[write] X <- log1p(normalize_total(target_sum={target:,.1f}))")
scale = np.zeros(n_obs, dtype=np.float64)
nzm = lib > 0
scale[nzm] = target / lib[nzm]
ts = time.time()
for ch in range(n_chunks):
    a, b = ch * args.chunk_size, min((ch + 1) * args.chunk_size, n_obs)
    lo, hi = int(indptr[a]), int(indptr[b])
    if hi == lo:
        continue
    d = np.asarray(z['X/data'][lo:hi], dtype=np.float64)
    rows = np.repeat(np.arange(a, b, dtype=np.int64),
                     np.diff(indptr[a:b + 1]))
    d *= scale[rows]
    np.log1p(d, out=d)
    z['X/data'][lo:hi] = d.astype(z['X/data'].dtype)
    if (ch + 1) % 20 == 0 or ch == n_chunks - 1:
        print(f"    chunk {ch+1}/{n_chunks} ({time.time()-ts:.0f}s)")

print(f"\n[done] X is log1p-normalised, raw counts in layers/{args.counts_layer}")
print(f"  target_sum used: {target:,.1f}")
print(f"  this store is now consumable by sc_dask_pipeline_post_pca_*")
print(f"\nTotal: {time.time()-t0:.0f}s")
