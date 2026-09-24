# BANKSY spatial pipeline — 10x/Xenium bundles to clustered object

End-to-end route from raw vendor output to a clustered, annotated object, plus
the single-sample shortcut and the subclustering loop.

All stages stream. Nothing requires the full expression matrix in memory.

---

## Stage 1 — merge 10x directories into one raw-counts zarr

```
merge_10x_to_zarr_streaming.py
    --sample_root <dir containing the per-sample 10x bundles>
    --samples     <name> [<name> ...] | all
    --output_dir  <RAW>
    --chunk_size  50000
    [--all_features]        keep control probes too; default is Gene Expression only
    [--index_unique _]      separator joining barcode and sample in obs_names
```

Writes `<RAW>/stacked.zarr` (anndata zarr v2) plus `samples.txt` and
`merge_summary.csv`:

| element | contents |
|---|---|
| `X` | **raw counts**, CSR, float32 values, **int64 indices** |
| `obs` | every `cells.csv.gz` column, plus `sample` as a Categorical |
| `obsm/spatial` | x/y centroids, float32 |
| `var` | gene names |

**Why this is cheap.** 10x stores the matrix as `shape = [n_features, n_cells]`
with `indptr` indexing **cells** — i.e. CSC with one column per cell. So a
contiguous block of cells is a contiguous read, and that block, with features as
the inner index, is *already* a valid CSR block of the (cells × genes) matrix:

```python
lo, hi = indptr[a], indptr[b]
csr = csr_matrix((data[lo:hi], indices[lo:hi], indptr[a:b+1] - lo),
                 shape=(b - a, n_features))
```

No transpose, no re-sort. Peak memory is one chunk. Samples append straight into
a single store, so there are no intermediate per-sample h5ads.

**int64 indices are mandatory, not defensive.** A single whole-transcriptome
sample can exceed 2^31 nonzeros on its own, so a merged store cannot use int32.
Note that scipy hands back an **int32 `indptr`** for any individual block under
2^31 nnz, so the running offset must be cast before it is added — that overflow
is a real failure mode, not a hypothetical.

**Constraints.** All samples must share an identical gene axis; the script
refuses to merge mismatched panels rather than silently intersecting them. Row
order follows `--samples`, which is what makes `block_diag` W valid downstream.

### Verify it

```
verify_merge_10x_zarr.py --merged_dir <RAW> --sample_root <dir> --n_cells 25
```

Pulls random cells from both the merged store and the source h5 files and
compares exact nonzeros, plus checks `indptr` monotonicity, that
`indptr[-1] == nnz`, that per-sample row blocks are contiguous in `samples.txt`
order, and that `obs_names` carry the right barcode. The whole low-memory
approach rests on the CSC-block claim above, so this is not optional.

---

## Stage 2 — normalise in place

```
normalize_merged_zarr.py
    --zarr        <RAW>/stacked.zarr
    --target_sum  median_of_medians | pooled_median | <float>
    [--sample_key sample]
    [--dry_run]   report the target_sum, write nothing
```

```
layers/counts  <-  X        (raw preserved)
X              <-  log1p( (X[i,:] / lib[i]) * target_sum )
```

Each cell is divided by **its own** library size and multiplied by the one
**shared** `target_sum`, i.e. `sc.pp.normalize_total(target_sum=...)`. The
sample grouping only *chooses* the scalar; it never scales per sample. A
per-sample target would give cells in different samples different totals, which
is the thing to avoid.

`--target_sum`:

- `median_of_medians` — per-sample median library size, then the median across
  samples. Robust to one aberrantly shallow sample. With n=2 it equals the mean
  of the two, so the robustness only begins to matter from n=3.
- `pooled_median` — median over all cells, cell-weighted, so a large sample
  dominates.
- a fixed number — use this to match another dataset's scale.

Since every cell ends at `target_sum` regardless, the choice affects
interpretability and cross-dataset comparability, not correctness.

**Why this stage exists at all:** the post-PCA pipelines never normalise. They
assume X is already log-normalised and that raw counts sit in `layers/counts`
for `seurat_v3` HVG selection. Feeding them a raw-counts store does not error —
it silently z-scores counts and produces a plausible-looking embedding driven by
library size.

**Idempotency:** `layers/counts` existing *is* the already-normalised marker. If
present the script reports and exits 0. `--force` overrides, but
double-normalising is unrecoverable, which is why the marker is a copied layer
rather than a flag that could be written before the work completed. The raw
layer is also written *before* X is touched, so a failure leaves the store
unchanged.

Only `X/data` is rewritten: row scaling and `log1p` do not change sparsity, so
`indices`/`indptr` are byte-identical before and after.

---

## Stage 3 — BANKSY PCA, Harmony, neighbours, UMAP, leiden

```
sc_dask_pipeline_post_pca_multisolver_zarr_h5ad_v2_subset.py
    --input_dir   <RAW>            parent of stacked.zarr
    --output_dir  <OBJ>
    --solver      linop_cpu | linop_gpu | linop_gpu_propack | dask_cpu | dask_gpu
    --lambda_val  <lambda>
    --zscore      per_sample | global
    --harmony_cols sample  --harmony_theta 2.0
    --n_hvg_self  <N>              HVGs for the self half
    --n_comps     50
    --save_h5ad
    [--subset_indices <npy>]       omit to use every cell
    [--fp32]                       opt out of the fp64 default
```

Produces `<OBJ>/stacked.zarr` with `obsm/{X_pca, X_pca_harmony, X_umap}` and
`obs/leiden`, plus `<OBJ>/stacked.h5ad` when `--save_h5ad` is given.

**`--fp64` is the default.** `banksy_pca_asym.py` takes its working dtype from
X, and svds is asked to converge to machine epsilon (`tol=0`). Float32
accumulation error over millions of cells is ~1e-4 — orders of magnitude worse
than the demanded accuracy — so PROPACK cannot satisfy its convergence test and
raises `k=N singular triplets did not converge within kmax`. This is
scale-dependent: it passes at ~1M cells and fails above ~2M. `--fp32` exists
only for cases where Lanczos memory is the binding constraint, since PROPACK
allocates `v` as `(n_cells, 10*k)`.

**`--zscore per_sample` requires cells contiguous per sample**, which stage 1
guarantees.

**Solver note.** `linop_*` is matrix-free and never forms the BANKSY matrix.
`dask_*` materialises a lazy BANKSY matrix and does not scale flat in practice.

### Known gap

This script hardcodes `genes_nbr=None`, so `--n_hvg_self N` yields
**N self + all-genes neighbour** features. On a whole-transcriptome panel that
is ~20k features. The `--nbr_genes {hvg, all, file}` option exists only in
`run_single_sample_banksy_asym.py`; porting it here is outstanding.

---

## Stage 4 — re-cluster on GPU (optional)

```
sc_dask_pipeline_post_pca_linop_zarr_h5ad_v2_rapids.py
    --output_dir <OBJ>              reads obsm/X_pca from here
    --save_dir   <OBJ>_rapids
    --leiden_resolution <r>  --leiden_graph umap
    [--no_harmony]
```

Reuses the existing `X_pca`, so the expensive PCA is not recomputed — only
Harmony, neighbours, UMAP and leiden. Minutes rather than hours.

**Do not add `--resume` when using `--no_harmony`.** If the input already holds
an `X_pca_harmony`, the resume branch wins over the `--no_harmony` branch and
the run silently uses the corrected embedding, making the control meaningless
while appearing to succeed.

---

## Stage 5 — downstream

```
submit_downstream_all.sbatch <OBJECT_STEM> [CLUSTER_KEY] [RAPIDS_SFX]
```

Six steps: cluster QC, metagenes on two marker sets, cluster → cell-type
assignment, density grid, pseudobulk DEG. Expects `<STEM>/stacked.zarr` and
`<STEM>_rapids/stacked.h5ad`; `RAPIDS_SFX` accommodates variants such as
`_noharmony`.

**Metagene scoring and every other post-hoc readout see the full gene panel.**
HVG selection only annotates `var.highly_variable` and picks the PCA feature
set; it never subsets the object. So a marker gene outside the HVG set is scored
normally — but had no influence on the clustering.

---

## Subclustering loop

Re-enters at stage 3 with an index array.

```
build_celltype_indices_h5ad.py          select cells BY cell type
    --h5ad <OBJ>/stacked.h5ad  --assignment <...>/cluster_celltype_assignment.csv
    --celltypes epithelial  --out_npy <SUB>/indices.npy

build_epi_after_qc_indices.py           select by EXCLUDING clusters
```

Then stage 3 with `--subset_indices`.

**The invariant: W is built on ALL cells and only then sliced.** Building W on
the subset would make `H0 = W @ X` an average over a tissue with every other
lineage deleted — the neighbourhood half would describe an artefact rather than
the microenvironment. `adata` and `W` are sliced with the same ascending index
array. The subset pipelines also report how many retained cells end up with no
surviving spatial neighbour (empty W row, zero H0).

Indices address the **post-QC** object. `build_celltype_indices_h5ad.py` records
`n_cells_total` in a sidecar json; pass it to `--subset_expect_cells` so a QC
mismatch aborts instead of silently selecting the wrong cells.

---

## Single-sample shortcut

For one 10x directory, stages 1–4 collapse into one script that loads, QCs,
normalises, builds W, runs the PCA, and does neighbours/UMAP/leiden/plots:

```
run_single_sample_banksy_asym.py
    --data_path <10x dir | h5ad>   --out_dir <OBJ>
    --n_hvg_self <N>
    --nbr_genes  hvg | all | file    [--nbr_genes_file <genes.txt>]
    --lambda_val <lambda>  --leiden_resolution <r>  --leiden_graph umap
    [--subset_indices <npy>] [--subset_expect_cells <n>]
    [--target_sum <fixed>]
```

`--nbr_genes` controls the neighbour half:

- `hvg` (default) — the same HVGs as the self half, so `2 × N` features
- `all` — every gene, so `N + n_vars`
- `file` — an explicit list. This is how you give the neighbour half a
  **parent object's** HVGs while the self half uses HVGs recomputed on a
  subset: the self half then resolves substructure within the subset, while the
  neighbour half retains the genes describing what each cell sits *next to*.
  Restricting the neighbour half to subset-only HVGs discards exactly the
  microenvironment signal BANKSY exists to capture.

HVG selection runs **after** any `--subset_indices`, so `--n_hvg_self` naturally
gives subset-specific HVGs.

**Pass a fixed `--target_sum`** if the output will ever be merged or compared
with another sample. The default normalises to that sample's own median, which
makes samples non-comparable, and the script warns when it does so.

---

## Where the scripts live

Present in this directory:

```
banksy_pca_asym.py                                      run_pca_asymmetric, the solver
banksy_pca_multi.py                                     build_W_block, prepare_multi*
merge_10x_to_zarr_streaming.py                          stage 1
verify_merge_10x_zarr.py                                stage 1 verification
normalize_merged_zarr.py                                stage 2
sc_dask_pipeline_post_pca_multisolver_zarr_h5ad_v2_subset.py   stage 3
```

Not yet copied here — currently under `spatial_omics_crc/scripts/`:

```
sc_dask_pipeline_post_pca_linop_zarr_h5ad_v2_rapids.py  stage 4
submit_downstream_all.sbatch                            stage 5
run_single_sample_banksy_asym.py                        single-sample shortcut
build_celltype_indices_h5ad.py                          subclustering
pipeline_h5ad/build_epi_after_qc_indices.py             subclustering
```

`banksy_pca_asym.py` is treated as a package: pre- and post-processing belongs
in the calling script, not inside `run_pca_asymmetric`. `kmax` is fixed at
`10*k` there, which is why precision rather than iteration count is the lever
for convergence failures.

---

## Memory reference

| stage | peak driver | note |
|---|---|---|
| 1 merge | one chunk of the matrix | obs is O(total cells) and becomes the limit past ~50M cells |
| 2 normalise | one chunk | two streaming passes |
| 3 BANKSY PCA | X sparse + Lanczos vectors `(n_cells, 10*k)` | fp64 doubles the values array |
| 4 GPU | VRAM for the PCs | PCA not recomputed |
| 5 downstream | pseudobulk aggregation | one pass with an indicator matmul |

Chunk sizing is panel-dependent: nnz-per-cell differs by an order of magnitude
between a targeted panel and a whole-transcriptome one, so a chunk size tuned
for one will blow up on the other.
