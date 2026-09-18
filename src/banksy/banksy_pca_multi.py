# Multi-sample BANKSY PCA preparation — importable module.
#
# Handles per-sample load/normalize, per-sample spatial graph, block-diagonal W,
# and joint HVG selection. Returns (adata_concat, W_block) ready for run_pca().
#
# Three entry points:
#   prepare_multi(h5ad_paths, ...)          — load from per-sample h5ad files
#   prepare_multi_zarr(zarr_path, ...)      — load from pre-concatenated zarr (raw counts in .X)
#   prepare_multi_adata(adata, ...)         — data already in memory (normalize=True/False)
#
# Low-memory entry points (for very large datasets):
#   sorted_idx = get_sorted_idx(adata, batch_key)
#   W_block    = build_W_block(adata_sorted, batch_key, ...)
#
#   Use these instead of prepare_multi_adata when the dataset is too large to hold
#   two copies of X simultaneously.  The caller controls when to sort X and when to
#   free the original, so the W build runs against only the sorted copy:
#
#     sorted_idx = get_sorted_idx(adata, batch_key='sample')
#     adata      = adata[sorted_idx].copy()   # peak: 2× X (no fragmentation yet)
#     gc.collect()                             # original freed by rebind above
#     W_block    = build_W_block(adata, ...)  # only 1× X + W in memory
#
# Usage:
#   import sys
#   sys.path.insert(0, "/home/users/astar/gis/stuyk1/scratch/spatial_omics_crc/scripts")
#   from banksy_pca_multi import prepare_multi, prepare_multi_zarr, prepare_multi_adata
#   from banksy_pca_multi import get_sorted_idx, build_W_block
#   from banksy_pca import run_pca
#
#   # from per-sample h5ads
#   adata, W = prepare_multi(h5ad_paths, target_sum=272.0, n_hvg=2000)
#
#   # from pre-concatenated zarr
#   adata, W = prepare_multi_zarr(zarr_path, target_sum=272.0, batch_key='sample', n_hvg=2000)
#
#   # from in-memory AnnData already normalized
#   adata, W = prepare_multi_adata(adata, batch_key='sample', normalize=False, n_hvg=None)
#
#   embeddings, ev, evr = run_pca(adata, W, solver='linop_cpu', n_comps=50)
#
# Design
# ------
#   - W is built per sample, then combined as block_diag → zero cross-sample edges
#   - HVG selection uses batch_key so genes variable across (not just within) samples
#   - banksy_pca.py is never modified — this module only produces its inputs
#   - _build_W_from_sorted uses only obsm['spatial'] + obs[batch_key]; never touches X

import gc
import os
import time
import numpy as np
import scipy.sparse as sparse
import anndata
import scanpy as sc
import pandas as pd

pd.options.mode.string_storage = "python"
pd.options.future.infer_string = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_normalize(path, target_sum, obs_x_col, obs_y_col):
    """Load one h5ad, restore counts, normalize, log1p, ensure spatial obsm."""
    adata = sc.read_h5ad(path)
    adata.X = adata.layers['counts'].copy()
    adata.layers = {}
    adata.uns = {}
    adata.obsm['spatial'] = adata.obs[[obs_x_col, obs_y_col]].values
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def _build_W_one(adata, num_neighbours, nbr_weight_decay):
    """Build BANKSY spatial weight matrix for a single sample."""
    from banksy.initialize_banksy import initialize_banksy
    banksy_dict = initialize_banksy(
        adata, coord_keys=('x', 'y', 'spatial'),
        num_neighbours=num_neighbours,
        nbr_weight_decay=nbr_weight_decay,
        max_m=0,
        plt_edge_hist=False, plt_nbr_weights=False,
        plt_agf_angles=False, plt_theta=False,
    )
    return banksy_dict[nbr_weight_decay]['weights'][0]


def _build_W_from_sorted(adata, batch_key, num_neighbours, nbr_weight_decay):
    """Build block-diagonal W from an adata already sorted by (batch_key, obs_name).

    Only uses obsm['spatial'] and obs[batch_key] — never accesses X.

    Parameters
    ----------
    adata : AnnData
        Must be sorted by (batch_key, obs_name) before calling.
    batch_key, num_neighbours, nbr_weight_decay : as in prepare_multi_adata.

    Returns
    -------
    W_block : csr_matrix  (n_total, n_total)
    """
    samples   = adata.obs[batch_key].unique().tolist()
    n_samples = len(samples)
    print(f"  {n_samples} samples: {samples}", flush=True)

    W_list = []
    for i, sample in enumerate(samples):
        mask    = adata.obs[batch_key] == sample
        adata_i = adata[mask]
        print(f"\n[{i+1}/{n_samples}] {sample}  ({mask.sum()} cells)", flush=True)
        t1 = time.time()
        W_i = _build_W_one(adata_i, num_neighbours, nbr_weight_decay)
        W_list.append(W_i)
        print(f"  W shape={W_i.shape}  nnz={W_i.nnz}  ({time.time()-t1:.1f}s)",
              flush=True)
        gc.collect()

    print(f"\nBuilding block-diagonal W ({n_samples} blocks) ...", flush=True)
    W_block = sparse.block_diag(W_list, format='csr')
    assert W_block.shape == (adata.shape[0], adata.shape[0]), \
        f"W_block shape {W_block.shape} != expected ({adata.shape[0]}, {adata.shape[0]})"
    print(f"  W_block shape={W_block.shape}  nnz={W_block.nnz}", flush=True)
    del W_list
    gc.collect()

    return W_block


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def get_sorted_idx(adata, batch_key='sample'):
    """Return the index permutation that sorts cells by (batch_key, obs_name).

    Only reads obs columns — does not access X.  Use together with
    build_W_block() to avoid holding two copies of X in memory:

        sorted_idx = get_sorted_idx(adata, batch_key='sample')
        adata      = adata[sorted_idx].copy()   # sort X; rebind frees original
        gc.collect()
        W_block    = build_W_block(adata, ...)

    Parameters
    ----------
    adata : AnnData
    batch_key : str

    Returns
    -------
    sorted_idx : np.ndarray of int, shape (n_cells,)
        adata[sorted_idx] is sorted by (batch_key, obs_name).
        sorted_idx[i] = j means sorted position i came from original position j,
        so ``X_pca[sorted_idx] = embeddings`` maps results back to original order.
    """
    return np.lexsort([adata.obs_names.values,
                       adata.obs[batch_key].values])


def build_W_block(adata, batch_key='sample',
                  num_neighbours=15,
                  nbr_weight_decay='scaled_gaussian'):
    """Build block-diagonal spatial W from an adata sorted by (batch_key, obs_name).

    Only uses obsm['spatial'] and obs[batch_key] — does not access X.
    Use after get_sorted_idx() + sorting + freeing the original adata to
    keep peak memory at one copy of X + W rather than two copies + W.

    Parameters
    ----------
    adata : AnnData
        Must be sorted by (batch_key, obs_name).
    batch_key : str
    num_neighbours : int
    nbr_weight_decay : str

    Returns
    -------
    W_block : csr_matrix  (n_cells, n_cells)
    """
    return _build_W_from_sorted(adata, batch_key, num_neighbours, nbr_weight_decay)


def prepare_multi(h5ad_paths, target_sum,
                  n_hvg=2000,
                  batch_key='sample',
                  num_neighbours=15,
                  nbr_weight_decay='scaled_gaussian',
                  obs_x_col='x_centroid',
                  obs_y_col='y_centroid'):
    """Prepare multiple samples for joint BANKSY PCA.

    Parameters
    ----------
    h5ad_paths : list of str
        Per-sample h5ad files. Each must have a 'counts' layer.
    target_sum : float
        Normalization target (e.g. from target_sum.txt).
    n_hvg : int or None
        Jointly-selected HVGs using batch_key. None = keep all genes.
    batch_key : str
        obs column name to store the per-sample label.
    num_neighbours : int
        Spatial neighbours per cell for BANKSY graph.
    nbr_weight_decay : str
        Kernel for spatial weights ('scaled_gaussian' or 'reciprocal').
    obs_x_col, obs_y_col : str
        obs column names for x/y coordinates.

    Returns
    -------
    adata_concat : AnnData  (n_total_cells, n_genes)
    W_block : scipy.sparse.csr_matrix  (n_total_cells, n_total_cells)
    """
    adatas = []
    W_list = []
    labels = []

    for i, path in enumerate(h5ad_paths):
        sample_name = os.path.basename(os.path.dirname(path))
        labels.append(sample_name)
        t0 = time.time()
        print(f"\n[{i+1}/{len(h5ad_paths)}] {sample_name}", flush=True)

        print(f"  Loading and normalizing ...", flush=True)
        adata_i = _load_normalize(path, target_sum, obs_x_col, obs_y_col)
        sort_idx = np.argsort(adata_i.obs_names)
        adata_i = adata_i[sort_idx].copy()
        print(f"  shape={adata_i.shape}  ({time.time()-t0:.1f}s)", flush=True)

        print(f"  Building spatial graph (k={num_neighbours}) ...", flush=True)
        t1 = time.time()
        W_i = _build_W_one(adata_i, num_neighbours, nbr_weight_decay)
        W_list.append(W_i)
        print(f"  W shape={W_i.shape}  nnz={W_i.nnz}  ({time.time()-t1:.1f}s)", flush=True)

        adatas.append(adata_i)
        gc.collect()

    print(f"\nConcatenating {len(adatas)} samples ...", flush=True)
    adata_concat = anndata.concat(
        adatas,
        label=batch_key,
        keys=labels,
        index_unique="-",
        merge='same',
    )
    n_total = adata_concat.shape[0]
    print(f"  Concatenated shape: {adata_concat.shape}  "
          f"(batch col='{batch_key}')", flush=True)
    del adatas
    gc.collect()

    print(f"\nBuilding block-diagonal W ({len(W_list)} blocks) ...", flush=True)
    W_block = sparse.block_diag(W_list, format='csr')
    assert W_block.shape == (n_total, n_total), \
        f"W_block shape {W_block.shape} != expected ({n_total}, {n_total})"
    print(f"  W_block shape={W_block.shape}  nnz={W_block.nnz}", flush=True)
    del W_list
    gc.collect()

    if n_hvg is not None:
        print(f"\nSelecting {n_hvg} HVGs (batch_key='{batch_key}') ...", flush=True)
        sc.pp.highly_variable_genes(adata_concat, n_top_genes=n_hvg, batch_key=batch_key)
        n_found = int(adata_concat.var['highly_variable'].sum())
        print(f"  HVG: keeping {n_found}/{adata_concat.shape[1]} genes", flush=True)
        adata_concat = adata_concat[:, adata_concat.var['highly_variable']].copy()
        print(f"  Shape after HVG: {adata_concat.shape}", flush=True)

    print(f"\nprepare_multi done — "
          f"{adata_concat.shape[0]} cells × {adata_concat.shape[1]} genes  "
          f"across {len(labels)} samples", flush=True)
    return adata_concat, W_block


def prepare_multi_adata(adata, target_sum=None,
                        batch_key='sample',
                        n_hvg=2000,
                        num_neighbours=15,
                        nbr_weight_decay='scaled_gaussian',
                        obs_x_col='x_centroid',
                        obs_y_col='y_centroid',
                        normalize=True):
    """Prepare an in-memory AnnData for joint BANKSY PCA.

    Parameters
    ----------
    adata : AnnData
        Combined data. obs[batch_key] must exist. Raw counts in .X unless
        normalize=False.
    target_sum : float or None
        Normalization target. Required when normalize=True.
    batch_key : str
        obs column identifying the sample for each cell.
    n_hvg : int or None
        Jointly-selected HVGs. None = keep all genes.
    num_neighbours : int
        Spatial neighbours per cell for BANKSY graph.
    nbr_weight_decay : str
        Kernel for spatial weights ('scaled_gaussian' or 'reciprocal').
    obs_x_col, obs_y_col : str
        obs column names for x/y coordinates (used when 'spatial' not in obsm).
    normalize : bool
        If True (default), run normalize_total(target_sum) + log1p on .X.
        If False, assume .X already contains normalized, log1p'd values.

    Returns
    -------
    adata : AnnData  (n_total_cells, n_genes)  — sorted by (batch_key, obs_name)
    W_block : scipy.sparse.csr_matrix  (n_total_cells, n_total_cells)

    Note
    ----
    Peak memory is 2× adata.X because the sorted copy is created while the
    caller's reference (normalize=False) or the normalized copy (normalize=True)
    is still live.  For very large datasets (>100 M cells), prefer the explicit
    two-step pattern using get_sorted_idx() + build_W_block() in the caller so
    the original can be freed before W is built.
    """
    if batch_key not in adata.obs.columns:
        raise ValueError(
            f"batch_key='{batch_key}' not found in obs columns: "
            f"{list(adata.obs.columns)}"
        )

    if normalize:
        # Copy before mutating so caller's AnnData is unchanged.
        adata = adata.copy()
        if target_sum is None:
            raise ValueError("target_sum is required when normalize=True")
        print("Normalizing (normalize_total + log1p) ...", flush=True)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    else:
        print("Skipping normalization (normalize=False).", flush=True)

    if 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = adata.obs[[obs_x_col, obs_y_col]].values

    # Sort X first (while memory is as clean as possible), then build W.
    # W build (pynndescent × n_samples) can fragment significant memory; doing
    # the X sort before W build keeps the 2× X peak free of that fragmentation.
    print(f"Sorting cells by ('{batch_key}', obs_name) ...", flush=True)
    sorted_idx = get_sorted_idx(adata, batch_key)
    adata = adata[sorted_idx].copy()   # peak: existing adata + sorted copy
    gc.collect()

    print(f"Building spatial graph (k={num_neighbours}, "
          f"decay={nbr_weight_decay}) ...", flush=True)
    W_block = _build_W_from_sorted(adata, batch_key, num_neighbours, nbr_weight_decay)

    if n_hvg is not None:
        print(f"\nSelecting {n_hvg} HVGs (batch_key='{batch_key}') ...", flush=True)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, batch_key=batch_key)
        n_found = int(adata.var['highly_variable'].sum())
        print(f"  HVG: keeping {n_found}/{adata.shape[1]} genes", flush=True)
        adata = adata[:, adata.var['highly_variable']].copy()
        print(f"  Shape after HVG: {adata.shape}", flush=True)

    n_samples = adata.obs[batch_key].nunique()
    print(f"\nprepare_multi_adata done — "
          f"{adata.shape[0]} cells × {adata.shape[1]} genes  "
          f"across {n_samples} samples", flush=True)
    return adata, W_block


def prepare_multi_zarr(zarr_path, target_sum,
                       batch_key='sample',
                       n_hvg=2000,
                       num_neighbours=15,
                       nbr_weight_decay='scaled_gaussian',
                       obs_x_col='x_centroid',
                       obs_y_col='y_centroid'):
    """Prepare a pre-concatenated zarr for joint BANKSY PCA.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store. Raw counts must be in .X; obs[batch_key] must exist.
    target_sum : float
        Normalization target.
    batch_key : str
        obs column identifying the sample for each cell.
    n_hvg : int or None
        Jointly-selected HVGs. None = keep all genes.
    num_neighbours : int
        Spatial neighbours per cell for BANKSY graph.
    nbr_weight_decay : str
        Kernel for spatial weights ('scaled_gaussian' or 'reciprocal').
    obs_x_col, obs_y_col : str
        obs column names for x/y coordinates (used when 'spatial' not in obsm).

    Returns
    -------
    adata : AnnData  (n_total_cells, n_genes)
    W_block : scipy.sparse.csr_matrix  (n_total_cells, n_total_cells)
    """
    print(f"Loading zarr: {zarr_path} ...", flush=True)
    t0 = time.time()
    adata = anndata.read_zarr(zarr_path)
    print(f"  Loaded shape={adata.shape}  ({time.time()-t0:.1f}s)", flush=True)

    return prepare_multi_adata(
        adata,
        target_sum=target_sum,
        batch_key=batch_key,
        n_hvg=n_hvg,
        num_neighbours=num_neighbours,
        nbr_weight_decay=nbr_weight_decay,
        obs_x_col=obs_x_col,
        obs_y_col=obs_y_col,
        normalize=True,
    )
