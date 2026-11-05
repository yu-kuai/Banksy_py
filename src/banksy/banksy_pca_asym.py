# Asymmetric BANKSY PCA: independent gene sets for X_self and H0_nbr,
# with optional per-sample Z-scoring.
#
# BANKSY matrix:
#   M = [ sqrt(1-λ) * Z(X_self).T ]   n_self rows
#       [ sqrt(λ)   * Z(H0_nbr).T ]   n_nbr  rows
#
# Solvers:
#   linop_cpu  : matrix-free LinearOperator + scipy svds PROPACK (CPU)
#   linop_gpu  : matrix-free CuPy LinearOperator + cupyx svds ARPACK (GPU)
#   dask_cpu   : lazy dask BANKSY + scanpy covariance_eigh (CPU)
#   dask_gpu   : lazy dask BANKSY + rapids-singlecell covariance_eigh (GPU)
#
# Z-scoring modes:
#   'global'     — mean/std across all cells (default, matches banksy_pca.py)
#   'per_batch' — mean/std within each sample independently
#
# Gene filter modes (all combinations supported):
#   genes_self=None, genes_nbr=None  →  standard symmetric BANKSY
#   genes_self=HVGs, genes_nbr=None  →  HVG self, full neighborhood
#   genes_self=A,    genes_nbr=B     →  fully independent gene sets
#
# Usage:
#   from banksy_pca_asym import run_banksy_pca
#
#   adata_full, W = prepare_multi_adata(merged_adata, normalize=False, n_hvg=None)
#
#   # symmetric, per-sample Z-score
#   embeddings, ev, evr = run_banksy_pca(adata_full, W,
#                                             zscore='per_batch',
#                                             batch_key='sample')
#
#   # asymmetric HVG self + full H0, global Z-score
#   embeddings, ev, evr = run_banksy_pca(adata_full, W,
#                                             genes_self=hvg_list,
#                                             genes_nbr=None)

import time
import threading
import os
import numpy as np
import scipy.sparse as sparse
import anndata
import pandas as pd
import psutil

_proc = psutil.Process(os.getpid())
_t0   = time.time()

SOLVERS = ('linop_cpu', 'linop_gpu', 'dask_cpu', 'dask_gpu', 'sparse_cov_cpu', 'sparse_cov_gpu')


class MemMonitor:
    """Background thread polling VRAM (if CuPy available) and CPU RSS."""

    def __init__(self, interval: float = 0.5):
        self.interval  = interval
        self.peak_vram = 0.0
        self.peak_rss  = 0.0
        self._stop     = threading.Event()
        self._thread   = None

    def _poll(self):
        import sys
        # Only use CuPy if it is already imported — never trigger a fresh import
        # from a background thread. A fresh import on a CPU node holds Python's
        # module-level import lock for cupy while CUDA initializes, causing any
        # dask code in the main thread that probes for cupy to deadlock.
        _cp      = sys.modules.get('cupy', None)
        _has_gpu = _cp is not None

        while not self._stop.is_set():
            if _has_gpu:
                try:
                    free, total = _cp.cuda.runtime.memGetInfo()
                    vram_used = (total - free) / (1024 ** 3)
                    if vram_used > self.peak_vram:
                        self.peak_vram = vram_used
                except Exception:
                    pass
            rss_used = _proc.memory_info().rss / (1024 ** 3)
            if rss_used > self.peak_rss:
                self.peak_rss = rss_used
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self.peak_vram = 0.0
        self.peak_rss  = 0.0
        self._thread   = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        elapsed = time.time() - _t0
        print(f"[PEAK  {elapsed:7.1f}s] {'peak during monitored region':<55} "
              f"VRAM={self.peak_vram:.2f} GB  RSS={self.peak_rss:.2f} GB",
              flush=True)
        return self.peak_vram, self.peak_rss


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------

def _sparse_mean_std(X_csr):
    mean    = np.asarray(X_csr.mean(axis=0)).ravel()
    X2      = X_csr.copy()
    X2.data **= 2
    sq_mean = np.asarray(X2.mean(axis=0)).ravel()
    std     = np.sqrt(np.maximum(sq_mean - mean ** 2, 0.0))
    return mean, std


def _nbr_mean_std(W, X_csr, chunk_size, verbose=True):
    n_cells = W.shape[0]
    n_nbr   = X_csr.shape[1]
    nbr_sum    = np.zeros(n_nbr, dtype=np.float64)
    nbr_sq_sum = np.zeros(n_nbr, dtype=np.float64)
    for i in range(0, n_cells, chunk_size):
        j     = min(i + chunk_size, n_cells)
        chunk = W[i:j, :] @ X_csr
        if sparse.issparse(chunk):
            chunk = chunk.toarray()
        chunk = chunk.astype(np.float64)
        nbr_sum    += chunk.sum(axis=0)
        nbr_sq_sum += (chunk ** 2).sum(axis=0)
        if verbose:
            print(f"  nbr mean/std pass: {j}/{n_cells} cells", flush=True)
    mean = nbr_sum / n_cells
    std  = np.sqrt(np.maximum(nbr_sq_sum / n_cells - mean ** 2, 0.0))
    return mean, std


def _get_sample_slices(adata, batch_key):
    """Return list of (start, end) cell index ranges per sample.
    Assumes adata is sorted by batch_key (contiguous per sample).
    """
    labels = adata.obs[batch_key].values
    slices = []
    start  = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            slices.append((start, i))
            start = i
    slices.append((start, len(labels)))
    return slices


def _per_batch_stats(X_csr_self, X_csr_nbr, W, sample_slices, chunk_size):
    """Compute per-sample mean/std for X_self and H0_nbr = W_s @ X_nbr_s."""
    n_samples = len(sample_slices)
    n_self    = X_csr_self.shape[1]
    n_nbr     = X_csr_nbr.shape[1]

    mu_self  = np.zeros((n_samples, n_self),  dtype=np.float64)
    std_self = np.ones( (n_samples, n_self),  dtype=np.float64)
    mu_nbr   = np.zeros((n_samples, n_nbr),   dtype=np.float64)
    std_nbr  = np.ones( (n_samples, n_nbr),   dtype=np.float64)

    _nbr_is_self = (X_csr_nbr is X_csr_self)
    for s, (start, end) in enumerate(sample_slices):
        print(f"  [sample {s+1}/{n_samples}] cells {start}:{end}", flush=True)
        X_s = _csr_row_view(X_csr_self, start, end)
        mu_self[s], std_self[s] = _sparse_mean_std(X_s)
        W_s     = W[start:end, start:end]
        X_nbr_s = X_s if _nbr_is_self else _csr_row_view(X_csr_nbr, start, end)
        mu_nbr[s], std_nbr[s] = _nbr_mean_std(W_s, X_nbr_s,
                                               chunk_size=chunk_size,
                                               verbose=False)

    std_self[std_self == 0] = 1.0
    std_nbr[ std_nbr  == 0] = 1.0
    return mu_self, std_self, mu_nbr, std_nbr


# ---------------------------------------------------------------------------
# Dask helpers (used by dask_cpu and dask_gpu)
# ---------------------------------------------------------------------------

# Process-global cache for X_sample in zarr-stream mode.
# Holds at most one sample at a time — evicted when a new (s_start, s_end) key
# is seen. Safe only with the synchronous dask scheduler (same process, serial
# task execution), which is always set before calling sc.pp.pca here.
_zarr_sample_cache: dict = {}


def _load_csr_from_zarr(zarr_path, row_start, row_end, n_genes, gene_indices=None):
    """Load CSR rows [row_start:row_end] from a zarr X store (data/indices/indptr).
    If gene_indices is provided, subset columns to those indices after loading.
    """
    import zarr as _zarr
    z       = _zarr.open_group(zarr_path, mode='r', zarr_format=2)
    iptr    = z['X/indptr'][row_start:row_end + 1]
    nnz0    = int(iptr[0])
    nnz1    = int(iptr[-1])
    data    = z['X/data'][nnz0:nnz1].astype(np.float32)
    indices = z['X/indices'][nnz0:nnz1]
    indptr  = (iptr - nnz0).astype(np.int32)
    X = sparse.csr_matrix((data, indices, indptr),
                           shape=(row_end - row_start, n_genes))
    if gene_indices is not None:
        X = X[:, gene_indices]
    return X


def _banksy_chunk_asym(cell_start, cell_end, W, X_csr_self, X_csr_nbr,
                       X_mean, X_std, nbr_mean, nbr_std,
                       scale_own, scale_nbr):
    X_chunk   = X_csr_self[cell_start:cell_end].toarray().astype(np.float64)
    nbr_chunk = W[cell_start:cell_end, :] @ X_csr_nbr
    if sparse.issparse(nbr_chunk):
        nbr_chunk = nbr_chunk.toarray()
    nbr_chunk = nbr_chunk.astype(np.float64)
    X_z   = (X_chunk   - X_mean)   / (X_std   + 1e-10) * scale_own
    nbr_z = (nbr_chunk - nbr_mean) / (nbr_std + 1e-10) * scale_nbr
    return np.concatenate([X_z, nbr_z], axis=1)


def _build_banksy_dask_asym(X_csr_self, X_csr_nbr, W, lambda_val,
                             X_mean, X_std, nbr_mean, nbr_std, chunk_size):
    import dask
    import dask.array as da
    n_cells   = X_csr_self.shape[0]
    n_self    = X_csr_self.shape[1]
    n_nbr     = X_csr_nbr.shape[1]
    scale_own = float(np.sqrt(1.0 - lambda_val))
    scale_nbr = float(np.sqrt(lambda_val))
    delayed_chunks = []
    for i in range(0, n_cells, chunk_size):
        j = min(i + chunk_size, n_cells)
        chunk_d = dask.delayed(_banksy_chunk_asym)(
            i, j, W, X_csr_self, X_csr_nbr,
            X_mean, X_std, nbr_mean, nbr_std, scale_own, scale_nbr,
        )
        arr = da.from_delayed(chunk_d, shape=(j - i, n_self + n_nbr), dtype=np.float64)
        delayed_chunks.append(arr)
    banksy_dask = da.concatenate(delayed_chunks, axis=0)
    assert banksy_dask.chunksize[1] == banksy_dask.shape[1]
    return banksy_dask


def _build_banksy_dask_asym_persample(X_csr_self, X_csr_nbr, W, lambda_val,
                                       mu_self, std_self, mu_nbr, std_nbr,
                                       sample_slices, chunk_size=50_000):
    """Per-sample dask BANKSY: sub-chunked within each sample so no single
    delayed task materialises more than chunk_size rows at once."""
    import dask
    import dask.array as da
    n_self    = X_csr_self.shape[1]
    n_nbr     = X_csr_nbr.shape[1]
    scale_own = float(np.sqrt(1.0 - lambda_val))
    scale_nbr = float(np.sqrt(lambda_val))
    delayed_chunks = []
    for s, (start, end) in enumerate(sample_slices):
        for i in range(start, end, chunk_size):
            j = min(i + chunk_size, end)
            chunk_d = dask.delayed(_banksy_chunk_asym)(
                i, j, W, X_csr_self, X_csr_nbr,
                mu_self[s], std_self[s], mu_nbr[s], std_nbr[s],
                scale_own, scale_nbr,
            )
            arr = da.from_delayed(chunk_d, shape=(j - i, n_self + n_nbr), dtype=np.float64)
            delayed_chunks.append(arr)
    banksy_dask = da.concatenate(delayed_chunks, axis=0)
    assert banksy_dask.chunksize[1] == banksy_dask.shape[1]
    return banksy_dask


def _banksy_chunk_asym_zarr(cell_start, cell_end, s_start, s_end,
                             input_zarr, n_genes, W,
                             X_mean, X_std, nbr_mean, nbr_std,
                             scale_own, scale_nbr):
    """Like _banksy_chunk_asym but reads X from zarr instead of in-memory CSR.
    Loads only one sample's rows (s_start:s_end) per call — X never fully in RAM.
    W[cell_start:cell_end, s_start:s_end] is used because W is block-diagonal,
    so only the sample block is non-zero.
    """
    X_self   = _load_csr_from_zarr(input_zarr, cell_start, cell_end, n_genes)
    X_sample = _load_csr_from_zarr(input_zarr, s_start,    s_end,    n_genes)
    X_chunk   = X_self.toarray().astype(np.float64)
    W_block   = W[cell_start:cell_end, s_start:s_end]
    nbr_chunk = W_block @ X_sample
    if sparse.issparse(nbr_chunk):
        nbr_chunk = nbr_chunk.toarray()
    nbr_chunk = nbr_chunk.astype(np.float64)
    X_z   = (X_chunk   - X_mean)   / (X_std   + 1e-10) * scale_own
    nbr_z = (nbr_chunk - nbr_mean) / (nbr_std + 1e-10) * scale_nbr
    return np.concatenate([X_z, nbr_z], axis=1)


def _build_banksy_dask_asym_persample_zarr(input_zarr, n_genes, W, lambda_val,
                                            mu_self, std_self, mu_nbr, std_nbr,
                                            sample_slices, chunk_size=50_000):
    """Per-sample zarr-backed dask BANKSY: X is never loaded into RAM.
    Each delayed task reads only its own sub-chunk rows + its sample rows from zarr.
    """
    import dask
    import dask.array as da
    n_cols    = n_genes * 2
    scale_own = float(np.sqrt(1.0 - lambda_val))
    scale_nbr = float(np.sqrt(lambda_val))
    delayed_chunks = []
    for s, (s_start, s_end) in enumerate(sample_slices):
        for i in range(s_start, s_end, chunk_size):
            j = min(i + chunk_size, s_end)
            chunk_d = dask.delayed(_banksy_chunk_asym_zarr)(
                i, j, s_start, s_end,
                input_zarr, n_genes, W,
                mu_self[s], std_self[s], mu_nbr[s], std_nbr[s],
                scale_own, scale_nbr,
            )
            arr = da.from_delayed(chunk_d, shape=(j - i, n_cols), dtype=np.float64)
            delayed_chunks.append(arr)
    banksy_dask = da.concatenate(delayed_chunks, axis=0)
    assert banksy_dask.chunksize[1] == banksy_dask.shape[1]
    return banksy_dask


def _banksy_chunk_asym_zarr_v2(cell_start, cell_end, s_start, s_end,
                                input_zarr, n_genes, W,
                                X_mean, X_std, nbr_mean, nbr_std,
                                scale_own, scale_nbr,
                                gene_indices=None):
    """Like _banksy_chunk_asym_zarr but caches X_sample across sub-chunk calls.

    Within one sample, X_sample (full sample rows) is identical for every
    sub-chunk task.  With the synchronous dask scheduler tasks run serially
    in the same process, so the module-level _zarr_sample_cache is shared.
    The cache holds exactly one entry; when a new (s_start, s_end) key is
    seen the old entry is evicted immediately, so peak memory = one X_sample.
    """
    global _zarr_sample_cache
    cache_key = (input_zarr, s_start, s_end)

    if cache_key not in _zarr_sample_cache:
        _zarr_sample_cache.clear()   # evict previous sample
        _zarr_sample_cache[cache_key] = _load_csr_from_zarr(
            input_zarr, s_start, s_end, n_genes, gene_indices)

    X_self   = _load_csr_from_zarr(input_zarr, cell_start, cell_end, n_genes, gene_indices)
    X_sample = _zarr_sample_cache[cache_key]

    X_chunk   = X_self.toarray().astype(np.float64)
    W_block   = W[cell_start:cell_end, s_start:s_end]
    nbr_chunk = W_block @ X_sample
    if sparse.issparse(nbr_chunk):
        nbr_chunk = nbr_chunk.toarray()
    nbr_chunk = nbr_chunk.astype(np.float64)
    X_z   = (X_chunk   - X_mean)   / (X_std   + 1e-10) * scale_own
    nbr_z = (nbr_chunk - nbr_mean) / (nbr_std + 1e-10) * scale_nbr
    return np.concatenate([X_z, nbr_z], axis=1)


def _build_banksy_dask_asym_persample_zarr_v2(input_zarr, n_genes, W, lambda_val,
                                               mu_self, std_self, mu_nbr, std_nbr,
                                               sample_slices, chunk_size=50_000,
                                               gene_indices=None):
    """Like _build_banksy_dask_asym_persample_zarr but uses _banksy_chunk_asym_zarr_v2
    so X_sample is loaded from zarr once per sample instead of once per sub-chunk.
    Requires synchronous dask scheduler.
    """
    import dask
    import dask.array as da
    n_selected = len(gene_indices) if gene_indices is not None else n_genes
    n_cols    = n_selected * 2
    scale_own = float(np.sqrt(1.0 - lambda_val))
    scale_nbr = float(np.sqrt(lambda_val))
    delayed_chunks = []
    for s, (s_start, s_end) in enumerate(sample_slices):
        for i in range(s_start, s_end, chunk_size):
            j = min(i + chunk_size, s_end)
            chunk_d = dask.delayed(_banksy_chunk_asym_zarr_v2)(
                i, j, s_start, s_end,
                input_zarr, n_genes, W,
                mu_self[s], std_self[s], mu_nbr[s], std_nbr[s],
                scale_own, scale_nbr,
                gene_indices,
            )
            arr = da.from_delayed(chunk_d, shape=(j - i, n_cols), dtype=np.float64)
            delayed_chunks.append(arr)
    banksy_dask = da.concatenate(delayed_chunks, axis=0)
    assert banksy_dask.chunksize[1] == banksy_dask.shape[1]
    return banksy_dask


def _per_batch_stats_from_zarr(input_zarr, n_genes, W, sample_slices, chunk_size,
                                 gene_indices=None):
    """Compute per-sample mean/std by loading one sample at a time from zarr.
    Peak memory = one sample's X (~3 GB for the largest sample) rather than all of X.
    """
    n_samples  = len(sample_slices)
    n_selected = len(gene_indices) if gene_indices is not None else n_genes
    mu_self  = np.zeros((n_samples, n_selected), dtype=np.float64)
    std_self = np.ones( (n_samples, n_selected), dtype=np.float64)
    mu_nbr   = np.zeros((n_samples, n_selected), dtype=np.float64)
    std_nbr  = np.ones( (n_samples, n_selected), dtype=np.float64)
    for s, (start, end) in enumerate(sample_slices):
        print(f"  [sample {s+1}/{n_samples}] cells {start}:{end}", flush=True)
        X_s = _load_csr_from_zarr(input_zarr, start, end, n_genes, gene_indices)
        mu_self[s], std_self[s] = _sparse_mean_std(X_s)
        W_s = W[start:end, start:end]
        mu_nbr[s], std_nbr[s]  = _nbr_mean_std(W_s, X_s,
                                                chunk_size=chunk_size,
                                                verbose=False)
        del X_s
    std_self[std_self == 0] = 1.0
    std_nbr[ std_nbr  == 0] = 1.0
    return mu_self, std_self, mu_nbr, std_nbr


def _global_stats_from_zarr(input_zarr, n_genes, W, sample_slices, chunk_size,
                             gene_indices=None):
    """Compute global (across all cells) mean/std for X_self and W@X_nbr from zarr.
    W is block-diagonal so we process one sample at a time for memory efficiency.
    """
    n_total     = sum(end - start for start, end in sample_slices)
    n_selected  = len(gene_indices) if gene_indices is not None else n_genes
    sum_self    = np.zeros(n_selected, dtype=np.float64)
    sum_sq_self = np.zeros(n_selected, dtype=np.float64)
    sum_nbr     = np.zeros(n_selected, dtype=np.float64)
    sum_sq_nbr  = np.zeros(n_selected, dtype=np.float64)
    n_samples   = len(sample_slices)
    for s, (start, end) in enumerate(sample_slices):
        print(f"  [global stats {s+1}/{n_samples}] cells {start}:{end}", flush=True)
        X_s = _load_csr_from_zarr(input_zarr, start, end, n_genes, gene_indices)
        X2 = X_s.copy(); X2.data **= 2
        sum_self    += np.asarray(X_s.sum(axis=0)).ravel()
        sum_sq_self += np.asarray(X2.sum(axis=0)).ravel()
        del X2
        W_s = W[start:end, start:end]
        for i in range(0, end - start, chunk_size):
            j     = min(i + chunk_size, end - start)
            chunk = W_s[i:j, :] @ X_s
            if sparse.issparse(chunk):
                chunk = chunk.toarray()
            chunk = np.asarray(chunk, dtype=np.float64)
            sum_nbr    += chunk.sum(axis=0)
            sum_sq_nbr += (chunk ** 2).sum(axis=0)
        del X_s
    mean_self = sum_self    / n_total
    mean_nbr  = sum_nbr     / n_total
    std_self  = np.sqrt(np.maximum(sum_sq_self / n_total - mean_self ** 2, 0.0))
    std_nbr   = np.sqrt(np.maximum(sum_sq_nbr  / n_total - mean_nbr  ** 2, 0.0))
    std_self[std_self == 0] = 1.0
    std_nbr[ std_nbr  == 0] = 1.0
    return mean_self, std_self, mean_nbr, std_nbr


def _banksy_chunk_asym_global_zarr(cell_start, cell_end, s_start, s_end,
                                    input_zarr, n_genes, W,
                                    global_mean_self, global_std_self,
                                    global_mean_nbr, global_std_nbr,
                                    scale_own, scale_nbr,
                                    gene_indices=None):
    """Chunk builder for global-zscore zarr-stream: same cache logic as v2
    but applies a single global mean/std instead of per-sample stats.
    """
    global _zarr_sample_cache
    cache_key = (input_zarr, s_start, s_end)
    if cache_key not in _zarr_sample_cache:
        _zarr_sample_cache.clear()
        _zarr_sample_cache[cache_key] = _load_csr_from_zarr(
            input_zarr, s_start, s_end, n_genes, gene_indices)
    X_self   = _load_csr_from_zarr(input_zarr, cell_start, cell_end, n_genes, gene_indices)
    X_sample = _zarr_sample_cache[cache_key]
    X_chunk   = X_self.toarray().astype(np.float64)
    W_block   = W[cell_start:cell_end, s_start:s_end]
    nbr_chunk = W_block @ X_sample
    if sparse.issparse(nbr_chunk):
        nbr_chunk = nbr_chunk.toarray()
    nbr_chunk = np.asarray(nbr_chunk, dtype=np.float64)
    X_z   = (X_chunk   - global_mean_self) / (global_std_self + 1e-10) * scale_own
    nbr_z = (nbr_chunk - global_mean_nbr)  / (global_std_nbr  + 1e-10) * scale_nbr
    return np.concatenate([X_z, nbr_z], axis=1)


def _build_banksy_dask_asym_global_zarr(input_zarr, n_genes, W, lambda_val,
                                         global_mean_self, global_std_self,
                                         global_mean_nbr, global_std_nbr,
                                         sample_slices, chunk_size=50_000,
                                         gene_indices=None):
    """Lazy dask BANKSY array for global-zscore zarr-stream.
    Mirrors _build_banksy_dask_asym_persample_zarr_v2 but uses global stats.
    Requires synchronous dask scheduler.
    """
    import dask
    import dask.array as da
    n_selected = len(gene_indices) if gene_indices is not None else n_genes
    n_cols    = n_selected * 2
    scale_own = float(np.sqrt(1.0 - lambda_val))
    scale_nbr = float(np.sqrt(lambda_val))
    delayed_chunks = []
    for s_start, s_end in sample_slices:
        for i in range(s_start, s_end, chunk_size):
            j       = min(i + chunk_size, s_end)
            chunk_d = dask.delayed(_banksy_chunk_asym_global_zarr)(
                i, j, s_start, s_end,
                input_zarr, n_genes, W,
                global_mean_self, global_std_self,
                global_mean_nbr, global_std_nbr,
                scale_own, scale_nbr,
                gene_indices,
            )
            arr = da.from_delayed(chunk_d, shape=(j - i, n_cols), dtype=np.float64)
            delayed_chunks.append(arr)
    banksy_dask = da.concatenate(delayed_chunks, axis=0)
    assert banksy_dask.chunksize[1] == banksy_dask.shape[1]
    return banksy_dask


def _make_banksy_adata_asym(adata, genes_self, genes_nbr, banksy_dask):
    var_own = (adata[:, genes_self].var if genes_self is not None else adata.var).copy()
    var_nbr = (adata[:, genes_nbr].var if genes_nbr is not None else adata.var).copy()
    var_own['is_nbr'] = False
    var_nbr = var_nbr.copy()
    var_nbr.index = var_nbr.index.astype(str) + '_nbr'
    var_nbr['is_nbr'] = True
    return anndata.AnnData(
        X=banksy_dask,
        obs=adata.obs.copy(),
        var=pd.concat([var_own, var_nbr]),
    )


def _csr_row_view(X, start, end):
    """Return a CSR matrix for rows [start:end] of X sharing X's data/indices arrays.

    scipy's csr_matrix[start:end] always copies data+indices (even in 1.17+).
    This function constructs the slice using numpy views so no bulk copy occurs —
    only a tiny adjusted indptr array (~4 B × (end-start+1)) is allocated.
    The caller must not modify the returned matrix's data in place.
    """
    rs = int(X.indptr[start])
    re = int(X.indptr[end])
    return sparse.csr_matrix(
        (X.data[rs:re], X.indices[rs:re], X.indptr[start:end + 1] - rs),
        shape=(end - start, X.shape[1]),
        copy=False,
    )


# ---------------------------------------------------------------------------
# Solver: linop_cpu
# ---------------------------------------------------------------------------

def _pca_linop_cpu_asym(X_csr_self, X_csr_nbr, W, lambda_val,
                         X_mean, X_std, nbr_mean, nbr_std, n_comps,
                         zscore, sample_slices,
                         mu_self, std_self, mu_nbr, std_nbr,
                         fp_dtype=np.float64,
                         linop_svd_solver='propack'):
    from scipy.sparse.linalg import LinearOperator, svds, eigsh
    n_cells   = X_csr_self.shape[0]
    n_self    = X_csr_self.shape[1]
    n_nbr     = X_csr_nbr.shape[1]
    scale_own = float(np.sqrt(1.0 - lambda_val))
    scale_nbr = float(np.sqrt(lambda_val))
    Wt        = sparse.csr_matrix(W.T).astype(fp_dtype, copy=False)

    if zscore == 'global':
        sx         = (X_std    + 1e-10).astype(fp_dtype)
        sn         = (nbr_std  + 1e-10).astype(fp_dtype)
        mu_over_sx = (X_mean   / sx).astype(fp_dtype)
        mu_over_sn = (nbr_mean / sn).astype(fp_dtype)
        X_mean_f   = X_mean.astype(fp_dtype)
        nbr_mean_f = nbr_mean.astype(fp_dtype)

        def matvec(v):
            v    = np.asarray(v, dtype=fp_dtype).ravel()
            vsum = v.sum()
            top    = (X_csr_self.T @ v    - X_mean_f   * vsum) / sx * scale_own
            Wt_v   = Wt @ v
            bottom = (X_csr_nbr.T  @ Wt_v - nbr_mean_f * vsum) / sn * scale_nbr
            return np.concatenate([top, bottom])

        def rmatvec(u):
            u  = np.asarray(u, dtype=fp_dtype).ravel()
            u1 = u[:n_self] * scale_own
            u2 = u[n_self:] * scale_nbr
            top    = X_csr_self @ (u1 / sx) - fp_dtype(mu_over_sx @ u1)
            u2s    = u2 / sn
            bottom = W @ (X_csr_nbr @ u2s) - fp_dtype(mu_over_sn @ u2)
            return top + bottom

    else:  # per_batch
        # Use view-based row slices so X's bulk data/indices are NOT copied.
        # scipy's csr[i:j] always copies in 1.17+; _csr_row_view shares the arrays.
        # Total: ~3 MB × n_samples instead of n_GB × n_samples.
        X_self_blocks = [_csr_row_view(X_csr_self, s, e)            for s, e in sample_slices]
        X_nbr_blocks  = (X_self_blocks if (X_csr_nbr is X_csr_self)
                         else [_csr_row_view(X_csr_nbr, s, e)       for s, e in sample_slices])
        W_blocks      = [W[s:e, s:e].astype(fp_dtype, copy=False)   for s, e in sample_slices]
        # Pre-cast per-sample stats to fp_dtype so matvec output stays in fp_dtype
        sx_blocks = [(std_self[s] + 1e-10).astype(fp_dtype) for s in range(len(sample_slices))]
        sn_blocks = [(std_nbr[s]  + 1e-10).astype(fp_dtype) for s in range(len(sample_slices))]
        mu_self_f = [mu_self[s].astype(fp_dtype)             for s in range(len(sample_slices))]
        mu_nbr_f  = [mu_nbr[s].astype(fp_dtype)              for s in range(len(sample_slices))]

        def matvec(v):
            v        = np.asarray(v, dtype=fp_dtype).ravel()
            Wt_v     = Wt @ v
            res_self = np.zeros(n_self, dtype=fp_dtype)
            res_nbr  = np.zeros(n_nbr,  dtype=fp_dtype)
            for s, (start, end) in enumerate(sample_slices):
                v_s    = v[start:end]
                vsum_s = v_s.sum()
                res_self += (X_self_blocks[s].T @ v_s
                             - mu_self_f[s] * vsum_s) / sx_blocks[s]
                res_nbr  += (X_nbr_blocks[s].T @ Wt_v[start:end]
                             - mu_nbr_f[s]  * vsum_s) / sn_blocks[s]
            return np.concatenate([res_self * scale_own, res_nbr * scale_nbr])

        def rmatvec(u):
            u   = np.asarray(u, dtype=fp_dtype).ravel()
            u1  = u[:n_self] * scale_own
            u2  = u[n_self:] * scale_nbr
            res = np.zeros(n_cells, dtype=fp_dtype)
            for s, (start, end) in enumerate(sample_slices):
                u1s  = u1 / sx_blocks[s]
                u2s  = u2 / sn_blocks[s]
                res[start:end] += (X_self_blocks[s] @ u1s
                                   - np.dot(mu_self_f[s], u1s))
                res[start:end] += (W_blocks[s] @ (X_nbr_blocks[s] @ u2s)
                                   - np.dot(mu_nbr_f[s], u2s))
            return res

    if linop_svd_solver == 'bbt_arpack':
        # Work in n_feat-dimensional space via eigsh on B @ B.T (n_feat × n_feat).
        # Working vectors are n_feat-dimensional → O(ncv × n_feat) memory, constant in n_cells.
        # Each operator application: rmatvec(v) → 1 n_cells-sized intermediate (400 MB at 100M),
        # then matvec(w) → n_feat-sized result. The intermediate is never accumulated.
        n_feat = n_self + n_nbr

        def bbt_matvec(v):
            w = rmatvec(np.asarray(v, dtype=fp_dtype))  # (n_cells,) — one vector only
            return matvec(w)                             # (n_feat,)

        A_op = LinearOperator(shape=(n_feat, n_feat), matvec=bbt_matvec, dtype=fp_dtype)
        print(f"  B@B.T operator shape: {A_op.shape}  k={n_comps}  solver=bbt_arpack",
              flush=True)
        _monitor = MemMonitor(interval=0.5).start()
        eigenvalues, U_feat = eigsh(A_op, k=n_comps, which='LM')
        _monitor.stop()

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, U_feat = eigenvalues[idx], U_feat[:, idx]
        S = np.sqrt(np.maximum(eigenvalues, 0.0))

        # Recovery: embeddings = B.T @ U_feat  (= right singular vectors × singular values)
        # One rmatvec call per PC — each is a single (n_cells,) vector.
        print(f"  Recovery: embeddings = B.T @ U ({n_comps} rmatvec calls) ...", flush=True)
        embeddings = np.column_stack([rmatvec(U_feat[:, i]) for i in range(n_comps)])

        ev        = eigenvalues / n_cells
        total_var = n_self * (1.0 - lambda_val) + n_nbr * lambda_val
        evr       = ev / total_var
        return embeddings, ev, evr

    n_feat = n_self + n_nbr

    def matmat_op(X):
        X   = np.asarray(X, dtype=fp_dtype)
        out = np.empty((n_feat, X.shape[1]), dtype=fp_dtype)
        for i in range(X.shape[1]):
            out[:, i] = matvec(X[:, i])
        return out

    def rmatmat_op(X):
        X   = np.asarray(X, dtype=fp_dtype)
        out = np.empty((n_cells, X.shape[1]), dtype=fp_dtype)
        for i in range(X.shape[1]):
            out[:, i] = rmatvec(X[:, i])
        return out

    M_op = LinearOperator(shape=(n_feat, n_cells),
                          matvec=matvec, rmatvec=rmatvec,
                          matmat=matmat_op, rmatmat=rmatmat_op,
                          dtype=fp_dtype)
    print(f"  LinearOperator shape: {M_op.shape}  k={n_comps}  solver={linop_svd_solver}",
          flush=True)
    _monitor = MemMonitor(interval=0.5).start()
    U, S, Vt = svds(M_op, k=n_comps, solver=linop_svd_solver)
    _monitor.stop()

    idx = np.argsort(S)[::-1]
    S, Vt = S[idx], Vt[idx, :]
    embeddings = Vt.T * S
    ev        = S ** 2 / n_cells
    total_var = n_self * (1.0 - lambda_val) + n_nbr * lambda_val
    evr       = ev / total_var
    return embeddings, ev, evr


# ---------------------------------------------------------------------------
# Solver: sparse_cov_cpu
# ---------------------------------------------------------------------------

def _build_banksy_cov_sparse(X_csr_self, X_csr_nbr, W,
                              zscore, sample_slices,
                              X_mean, X_std, nbr_mean, nbr_std,
                              mu_self, std_self, mu_nbr, std_nbr,
                              lambda_val, chunk_size, fp_dtype):
    """Build the (2g × 2g) BANKSY covariance C = BᵀB using the sparse XᵀX identity:

        ZᵀZ = D⁻¹(XᵀX − n·μμᵀ)D⁻¹

    X is never densified. H = W@X is materialised one chunk at a time.
    Works for both zscore='global' and zscore='per_batch'.
    """
    n_cells      = X_csr_self.shape[0]
    n_self       = X_csr_self.shape[1]
    n_nbr        = X_csr_nbr.shape[1]
    scale_own    = float(np.sqrt(1.0 - lambda_val))
    scale_nbr    = float(np.sqrt(lambda_val))
    _nbr_is_self = (X_csr_nbr is X_csr_self)
    # For global zscore, sample_slices is None — treat all cells as one block.
    # W is block-diagonal so W[i:j, :] @ X gives correct per-row neighbour averages.
    if sample_slices is None:
        sample_slices = [(0, n_cells)]
    n_samp = len(sample_slices)

    C_ss = np.zeros((n_self, n_self), dtype=fp_dtype)
    C_sh = np.zeros((n_self, n_nbr),  dtype=fp_dtype)
    C_hh = np.zeros((n_nbr,  n_nbr),  dtype=fp_dtype)

    for s, (start, end) in enumerate(sample_slices):
        n_s  = end - start
        X_s  = _csr_row_view(X_csr_self, start, end).astype(fp_dtype, copy=False)
        Xn_s = (X_s if _nbr_is_self
                else _csr_row_view(X_csr_nbr, start, end).astype(fp_dtype, copy=False))
        W_s  = W[start:end, start:end].astype(fp_dtype, copy=False)
        print(f"  [sparse_cov {s+1}/{n_samp}] cells {start}:{end}", flush=True)

        if zscore == 'global':
            mu_x_s = X_mean.astype(fp_dtype)
            mu_n_s = nbr_mean.astype(fp_dtype)
            sx_s   = (X_std   + 1e-10).astype(fp_dtype)
            sn_s   = (nbr_std + 1e-10).astype(fp_dtype)
        else:
            mu_x_s = mu_self[s].astype(fp_dtype)
            mu_n_s = mu_nbr[s].astype(fp_dtype)
            sx_s   = (std_self[s] + 1e-10).astype(fp_dtype)
            sn_s   = (std_nbr[s]  + 1e-10).astype(fp_dtype)

        # Self block: XᵀX - n_s·μμᵀ  (sparse × sparse, no densification of X)
        C_ss_s  = (X_s.T @ X_s).toarray()
        C_ss_s -= n_s * np.outer(mu_x_s, mu_x_s)
        C_ss   += (C_ss_s / sx_s[:, None]) / sx_s[None, :]
        del C_ss_s

        # H = W_s @ Xn_s: accumulated in chunks — avoids materialising full H_s.
        # h_chunk is ~51% nonzero (k=15 neighbours × 4.6% gene density), so
        # scipy sparse × sparse is slow; convert to dense and use BLAS instead.
        C_hh_s = np.zeros((n_nbr, n_nbr), dtype=fp_dtype)
        C_sh_s = np.zeros((n_self, n_nbr), dtype=fp_dtype)
        for i in range(0, n_s, chunk_size):
            j       = min(i + chunk_size, n_s)
            h_chunk = (W_s[i:j, :] @ Xn_s).toarray().astype(fp_dtype)  # dense (chunk × n_nbr)
            x_chunk = X_s[i:j].toarray().astype(fp_dtype)               # dense (chunk × n_self)
            C_hh_s += h_chunk.T @ h_chunk   # BLAS dgemm
            C_sh_s += x_chunk.T @ h_chunk   # BLAS dgemm

        C_hh_s -= n_s * np.outer(mu_n_s, mu_n_s)
        C_sh_s -= n_s * np.outer(mu_x_s, mu_n_s)
        C_hh   += (C_hh_s / sn_s[:, None]) / sn_s[None, :]
        C_sh   += (C_sh_s / sx_s[:, None]) / sn_s[None, :]
        del C_hh_s, C_sh_s

    # Lambda scaling
    C_ss *= scale_own ** 2
    C_hh *= scale_nbr ** 2
    C_sh *= scale_own * scale_nbr

    n_feat = n_self + n_nbr
    C = np.empty((n_feat, n_feat), dtype=fp_dtype)
    C[:n_self, :n_self] = C_ss
    C[:n_self, n_self:] = C_sh
    C[n_self:, :n_self] = C_sh.T
    C[n_self:, n_self:] = C_hh
    return C


def _build_banksy_cov_sparse_zarr(input_zarr, n_genes, W,
                                   zscore, sample_slices,
                                   X_mean, X_std, nbr_mean, nbr_std,
                                   mu_self, std_self, mu_nbr, std_nbr,
                                   lambda_val, chunk_size, fp_dtype,
                                   gene_indices=None):
    """Like _build_banksy_cov_sparse but reads X one sample at a time from zarr.
    X is never fully in RAM — peak RSS bounded by largest single sample.
    """
    n_samp    = len(sample_slices)
    n_self    = len(gene_indices) if gene_indices is not None else n_genes
    n_nbr     = n_self
    scale_own = float(np.sqrt(1.0 - lambda_val))
    scale_nbr = float(np.sqrt(lambda_val))

    C_ss = np.zeros((n_self, n_self), dtype=fp_dtype)
    C_sh = np.zeros((n_self, n_nbr),  dtype=fp_dtype)
    C_hh = np.zeros((n_nbr,  n_nbr),  dtype=fp_dtype)

    for s, (start, end) in enumerate(sample_slices):
        n_s = end - start
        X_s = _load_csr_from_zarr(input_zarr, start, end, n_genes,
                                   gene_indices).astype(fp_dtype, copy=False)
        W_s = W[start:end, start:end].astype(fp_dtype, copy=False)
        print(f"  [sparse_cov_zarr {s+1}/{n_samp}] cells {start}:{end}", flush=True)

        if zscore == 'global':
            mu_x_s = X_mean.astype(fp_dtype)
            mu_n_s = nbr_mean.astype(fp_dtype)
            sx_s   = (X_std   + 1e-10).astype(fp_dtype)
            sn_s   = (nbr_std + 1e-10).astype(fp_dtype)
        else:
            mu_x_s = mu_self[s].astype(fp_dtype)
            mu_n_s = mu_nbr[s].astype(fp_dtype)
            sx_s   = (std_self[s] + 1e-10).astype(fp_dtype)
            sn_s   = (std_nbr[s]  + 1e-10).astype(fp_dtype)

        # Self block: XᵀX - n_s·μμᵀ (sparse, no densification)
        C_ss_s  = (X_s.T @ X_s).toarray()
        C_ss_s -= n_s * np.outer(mu_x_s, mu_x_s)
        C_ss   += (C_ss_s / sx_s[:, None]) / sx_s[None, :]
        del C_ss_s

        # H = W_s @ X_s: accumulated in row-chunks to bound peak RSS
        C_hh_s = np.zeros((n_nbr, n_nbr), dtype=fp_dtype)
        C_sh_s = np.zeros((n_self, n_nbr), dtype=fp_dtype)
        for i in range(0, n_s, chunk_size):
            j       = min(i + chunk_size, n_s)
            h_chunk = (W_s[i:j, :] @ X_s).toarray().astype(fp_dtype)
            x_chunk = X_s[i:j].toarray().astype(fp_dtype)
            C_hh_s += h_chunk.T @ h_chunk
            C_sh_s += x_chunk.T @ h_chunk

        C_hh_s -= n_s * np.outer(mu_n_s, mu_n_s)
        C_sh_s -= n_s * np.outer(mu_x_s, mu_n_s)
        C_hh   += (C_hh_s / sn_s[:, None]) / sn_s[None, :]
        C_sh   += (C_sh_s / sx_s[:, None]) / sn_s[None, :]
        del C_hh_s, C_sh_s, X_s, W_s

    C_ss *= scale_own ** 2
    C_hh *= scale_nbr ** 2
    C_sh *= scale_own * scale_nbr

    n_feat = n_self + n_nbr
    C = np.empty((n_feat, n_feat), dtype=fp_dtype)
    C[:n_self, :n_self] = C_ss
    C[:n_self, n_self:] = C_sh
    C[n_self:, :n_self] = C_sh.T
    C[n_self:, n_self:] = C_hh
    return C


def _pca_sparse_cov_cpu_asym(X_csr_self, X_csr_nbr, W, lambda_val,
                              X_mean, X_std, nbr_mean, nbr_std, n_comps,
                              zscore, sample_slices,
                              mu_self, std_self, mu_nbr, std_nbr,
                              chunk_size=50_000, fp_dtype=np.float64,
                              input_zarr=None, n_genes=None, gene_indices=None):
    """BANKSY PCA via explicit sparse covariance + eigh.

    Builds B.T @ B ∈ R^{2g × 2g} using the sparse identity ZᵀZ = D⁻¹(XᵀX−nμμᵀ)D⁻¹.
    X is never densified; H = W@X is streamed in cell-chunks.
    When input_zarr is set, X is never loaded into RAM — read per sample from zarr.
    Embeddings recovered via rmatvec (in-memory) or vectorised per-sample zarr reads.
    """
    from scipy.linalg import eigh
    _zarr_mode = input_zarr is not None
    if _zarr_mode:
        n_self = len(gene_indices) if gene_indices is not None else n_genes
        n_nbr  = n_self
        n_cells = sample_slices[-1][1]
    else:
        n_cells   = X_csr_self.shape[0]
        n_self    = X_csr_self.shape[1]
        n_nbr     = X_csr_nbr.shape[1]
    n_feat    = n_self + n_nbr
    scale_own = float(np.sqrt(1.0 - lambda_val))
    scale_nbr = float(np.sqrt(lambda_val))

    print(f"\n  Building BANKSY covariance ({n_feat}×{n_feat}) via sparse XᵀX ...",
          flush=True)
    _mon_cov = MemMonitor(interval=0.5).start()
    if _zarr_mode:
        C = _build_banksy_cov_sparse_zarr(
            input_zarr, n_genes, W, zscore, sample_slices,
            X_mean, X_std, nbr_mean, nbr_std,
            mu_self, std_self, mu_nbr, std_nbr,
            lambda_val, chunk_size, fp_dtype, gene_indices=gene_indices,
        )
    else:
        C = _build_banksy_cov_sparse(
            X_csr_self, X_csr_nbr, W, zscore, sample_slices,
            X_mean, X_std, nbr_mean, nbr_std,
            mu_self, std_self, mu_nbr, std_nbr,
            lambda_val, chunk_size, fp_dtype,
        )
    _mon_cov.stop()
    print(f"  Cov built.  peak_rss={_mon_cov.peak_rss:.2f} GB", flush=True)

    print(f"  eigh (LAPACK) k={n_comps} on ({n_feat}×{n_feat}) ...", flush=True)
    _mon_eig = MemMonitor(interval=0.5).start()
    eigenvalues, U_feat = eigh(C, subset_by_index=[n_feat - n_comps, n_feat - 1],
                               driver='evr', overwrite_a=True)
    _mon_eig.stop()
    del C
    print(f"  eigh done.  peak_rss={_mon_eig.peak_rss:.2f} GB", flush=True)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, U_feat = eigenvalues[idx], U_feat[:, idx]

    # Recovery: embeddings = B @ U_feat  (B is n_cells × n_feat)
    if _zarr_mode:
        # Vectorised per-sample zarr read: load X_s once, compute all n_comps together.
        # n_samples zarr reads total (vs n_comps × n_samples for separate rmatvec calls).
        U1 = (U_feat[:n_self, :] * scale_own).astype(fp_dtype)  # n_self × n_comps
        U2 = (U_feat[n_self:, :] * scale_nbr).astype(fp_dtype)  # n_nbr  × n_comps
        if zscore == 'global':
            sx_g      = (X_std   + 1e-10).astype(fp_dtype)
            sn_g      = (nbr_std + 1e-10).astype(fp_dtype)
            U1s_g     = U1 / sx_g[:, None]
            U2s_g     = U2 / sn_g[:, None]
            mu_x_U1s  = X_mean.astype(fp_dtype) @ U1s_g    # (n_comps,)
            mu_n_U2s  = nbr_mean.astype(fp_dtype) @ U2s_g  # (n_comps,)
        print(f"  Recovery (zarr, vectorised): {len(sample_slices)} samples ...", flush=True)
        embeddings = np.zeros((n_cells, n_comps), dtype=fp_dtype)
        for s, (start, end) in enumerate(sample_slices):
            X_s = _load_csr_from_zarr(input_zarr, start, end, n_genes,
                                      gene_indices).astype(fp_dtype, copy=False)
            W_s = W[start:end, start:end].astype(fp_dtype, copy=False)
            if zscore == 'per_batch':
                sx      = (std_self[s] + 1e-10).astype(fp_dtype)
                sn      = (std_nbr[s]  + 1e-10).astype(fp_dtype)
                U1s     = U1 / sx[:, None]
                U2s     = U2 / sn[:, None]
                mu_x_U1s = mu_self[s].astype(fp_dtype) @ U1s
                mu_n_U2s = mu_nbr[s].astype(fp_dtype)  @ U2s
            else:
                U1s = U1s_g
                U2s = U2s_g
            embeddings[start:end, :] = (
                X_s @ U1s - mu_x_U1s
                + W_s @ (X_s @ U2s) - mu_n_U2s
            )
            del X_s, W_s
    else:
        # In-memory path: one rmatvec call per PC (same as bbt_arpack).
        _nbr_is_self = (X_csr_nbr is X_csr_self)
        if zscore == 'global':
            sx         = (X_std    + 1e-10).astype(fp_dtype)
            sn         = (nbr_std  + 1e-10).astype(fp_dtype)
            mu_over_sx = (X_mean   / sx).astype(fp_dtype)
            mu_over_sn = (nbr_mean / sn).astype(fp_dtype)
            X_fp       = X_csr_self.astype(fp_dtype, copy=False)
            Xn_fp      = X_csr_nbr.astype(fp_dtype, copy=False)
            W_fp       = W.astype(fp_dtype, copy=False)

            def rmatvec(u):
                u  = np.asarray(u, dtype=fp_dtype).ravel()
                u1 = u[:n_self] * scale_own
                u2 = u[n_self:] * scale_nbr
                return (X_fp @ (u1 / sx) - float(mu_over_sx @ u1)
                        + W_fp @ (Xn_fp @ (u2 / sn)) - float(mu_over_sn @ u2))
        else:
            X_self_blocks = [_csr_row_view(X_csr_self, s, e).astype(fp_dtype, copy=False)
                             for s, e in sample_slices]
            X_nbr_blocks  = (X_self_blocks if _nbr_is_self
                             else [_csr_row_view(X_csr_nbr, s, e).astype(fp_dtype, copy=False)
                                   for s, e in sample_slices])
            W_blocks  = [W[s:e, s:e].astype(fp_dtype, copy=False) for s, e in sample_slices]
            sx_blocks = [(std_self[s] + 1e-10).astype(fp_dtype) for s in range(len(sample_slices))]
            sn_blocks = [(std_nbr[s]  + 1e-10).astype(fp_dtype) for s in range(len(sample_slices))]
            mu_sf     = [mu_self[s].astype(fp_dtype) for s in range(len(sample_slices))]
            mu_nf     = [mu_nbr[s].astype(fp_dtype)  for s in range(len(sample_slices))]

            def rmatvec(u):
                u   = np.asarray(u, dtype=fp_dtype).ravel()
                u1  = u[:n_self] * scale_own
                u2  = u[n_self:] * scale_nbr
                res = np.zeros(n_cells, dtype=fp_dtype)
                for s, (start, end) in enumerate(sample_slices):
                    u1s = u1 / sx_blocks[s]
                    u2s = u2 / sn_blocks[s]
                    res[start:end] += X_self_blocks[s] @ u1s - np.dot(mu_sf[s], u1s)
                    res[start:end] += (W_blocks[s] @ (X_nbr_blocks[s] @ u2s)
                                       - np.dot(mu_nf[s], u2s))
                return res

        print(f"  Recovery: {n_comps} rmatvec calls ...", flush=True)
        embeddings = np.column_stack([rmatvec(U_feat[:, i]) for i in range(n_comps)])

    ev        = eigenvalues / n_cells
    total_var = n_self * (1.0 - lambda_val) + n_nbr * lambda_val
    evr       = ev / total_var
    return embeddings, ev, evr


# ---------------------------------------------------------------------------
# Solver: sparse_cov_gpu
# ---------------------------------------------------------------------------

def _build_banksy_cov_sparse_gpu(X_csr_self, X_csr_nbr, W_gpu,
                                   zscore, sample_slices,
                                   X_mean, X_std, nbr_mean, nbr_std,
                                   mu_self, std_self, mu_nbr, std_nbr,
                                   lambda_val, chunk_size):
    """GPU version of _build_banksy_cov_sparse.
    W_gpu already on GPU; X_s transferred per sample. VRAM flat.
    Uses gram_csr_upper CUDA kernel for XᵀX; cuBLAS DGEMM for h_chunk products.
    """
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from rapids_singlecell._cuda import _spca_cuda as _spca

    n_cells      = X_csr_self.shape[0]
    n_self       = X_csr_self.shape[1]
    n_nbr        = X_csr_nbr.shape[1]
    scale_own    = float(np.sqrt(1.0 - lambda_val))
    scale_nbr    = float(np.sqrt(lambda_val))
    _nbr_is_self = (X_csr_nbr is X_csr_self)
    if sample_slices is None:
        sample_slices = [(0, n_cells)]
    n_samp = len(sample_slices)

    C_ss = cp.zeros((n_self, n_self), dtype=cp.float32)
    C_sh = cp.zeros((n_self, n_nbr),  dtype=cp.float32)
    C_hh = cp.zeros((n_nbr,  n_nbr),  dtype=cp.float32)

    for s, (start, end) in enumerate(sample_slices):
        n_s     = end - start
        X_s_cpu = _csr_row_view(X_csr_self, start, end).astype(np.float32)
        X_s_gpu = csp.csr_matrix(X_s_cpu); del X_s_cpu
        X_s_gpu.sort_indices()
        Xn_s_gpu = (X_s_gpu if _nbr_is_self
                    else csp.csr_matrix(
                        _csr_row_view(X_csr_nbr, start, end).astype(np.float32)))
        if not _nbr_is_self:
            Xn_s_gpu.sort_indices()
        W_s_gpu = W_gpu[start:end, start:end]
        print(f"  [sparse_cov_gpu {s+1}/{n_samp}] cells {start}:{end}", flush=True)

        if zscore == 'global':
            mu_x_s = cp.asarray(X_mean.astype(np.float32))
            mu_n_s = cp.asarray(nbr_mean.astype(np.float32))
            sx_s   = cp.asarray((X_std   + 1e-10).astype(np.float32))
            sn_s   = cp.asarray((nbr_std + 1e-10).astype(np.float32))
        else:
            mu_x_s = cp.asarray(mu_self[s].astype(np.float32))
            mu_n_s = cp.asarray(mu_nbr[s].astype(np.float32))
            sx_s   = cp.asarray((std_self[s] + 1e-10).astype(np.float32))
            sn_s   = cp.asarray((std_nbr[s]  + 1e-10).astype(np.float32))

        # XᵀX upper triangle via custom CUDA kernel → mirror → subtract nμμᵀ
        C_ss_s = cp.zeros((n_self, n_self), dtype=cp.float32)
        _spca.gram_csr_upper(
            X_s_gpu.indptr, X_s_gpu.indices, X_s_gpu.data,
            nrows=n_s, ncols=n_self, out=C_ss_s,
            stream=cp.cuda.get_current_stream().ptr,
        )
        _spca.copy_upper_to_lower(out=C_ss_s, ncols=n_self,
                                   stream=cp.cuda.get_current_stream().ptr)
        C_ss_s -= n_s * cp.outer(mu_x_s, mu_x_s)
        C_ss   += (C_ss_s / sx_s[:, None]) / sx_s[None, :]
        del C_ss_s

        C_hh_s = cp.zeros((n_nbr, n_nbr), dtype=cp.float32)
        C_sh_s = cp.zeros((n_self, n_nbr), dtype=cp.float32)
        for i in range(0, n_s, chunk_size):
            j       = min(i + chunk_size, n_s)
            h_chunk = (W_s_gpu[i:j, :] @ Xn_s_gpu).toarray()  # dense (chunk × n_nbr) GPU
            x_chunk = X_s_gpu[i:j].toarray()                   # dense (chunk × n_self) GPU
            C_hh_s += h_chunk.T @ h_chunk   # cuBLAS DGEMM
            C_sh_s += x_chunk.T @ h_chunk   # cuBLAS DGEMM

        C_hh_s -= n_s * cp.outer(mu_n_s, mu_n_s)
        C_sh_s -= n_s * cp.outer(mu_x_s, mu_n_s)
        C_hh   += (C_hh_s / sn_s[:, None]) / sn_s[None, :]
        C_sh   += (C_sh_s / sx_s[:, None]) / sn_s[None, :]
        del C_hh_s, C_sh_s, X_s_gpu, Xn_s_gpu, W_s_gpu
        cp.cuda.Stream.null.synchronize()

    C_ss *= scale_own ** 2
    C_hh *= scale_nbr ** 2
    C_sh *= scale_own * scale_nbr

    n_feat = n_self + n_nbr
    C = cp.empty((n_feat, n_feat), dtype=cp.float32)
    C[:n_self, :n_self] = C_ss
    C[:n_self, n_self:] = C_sh
    C[n_self:, :n_self] = C_sh.T
    C[n_self:, n_self:] = C_hh
    return C


def _build_banksy_cov_sparse_zarr_gpu(input_zarr, n_genes, W_gpu,
                                       zscore, sample_slices,
                                       X_mean, X_std, nbr_mean, nbr_std,
                                       mu_self, std_self, mu_nbr, std_nbr,
                                       lambda_val, chunk_size,
                                       gene_indices=None):
    """GPU zarr-stream version: X read from zarr per sample, transferred to GPU.
    W_gpu already on GPU. VRAM flat (bounded by largest sample).
    """
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from rapids_singlecell._cuda import _spca_cuda as _spca

    n_samp    = len(sample_slices)
    n_self    = len(gene_indices) if gene_indices is not None else n_genes
    n_nbr     = n_self
    scale_own = float(np.sqrt(1.0 - lambda_val))
    scale_nbr = float(np.sqrt(lambda_val))

    C_ss = cp.zeros((n_self, n_self), dtype=cp.float32)
    C_sh = cp.zeros((n_self, n_nbr),  dtype=cp.float32)
    C_hh = cp.zeros((n_nbr,  n_nbr),  dtype=cp.float32)

    for s, (start, end) in enumerate(sample_slices):
        n_s     = end - start
        X_s_cpu = _load_csr_from_zarr(input_zarr, start, end, n_genes,
                                       gene_indices).astype(np.float32)
        X_s_gpu = csp.csr_matrix(X_s_cpu); del X_s_cpu
        X_s_gpu.sort_indices()
        W_s_gpu = W_gpu[start:end, start:end]
        print(f"  [sparse_cov_gpu_zarr {s+1}/{n_samp}] cells {start}:{end}", flush=True)

        if zscore == 'global':
            mu_x_s = cp.asarray(X_mean.astype(np.float32))
            mu_n_s = cp.asarray(nbr_mean.astype(np.float32))
            sx_s   = cp.asarray((X_std   + 1e-10).astype(np.float32))
            sn_s   = cp.asarray((nbr_std + 1e-10).astype(np.float32))
        else:
            mu_x_s = cp.asarray(mu_self[s].astype(np.float32))
            mu_n_s = cp.asarray(mu_nbr[s].astype(np.float32))
            sx_s   = cp.asarray((std_self[s] + 1e-10).astype(np.float32))
            sn_s   = cp.asarray((std_nbr[s]  + 1e-10).astype(np.float32))

        C_ss_s = cp.zeros((n_self, n_self), dtype=cp.float32)
        _spca.gram_csr_upper(
            X_s_gpu.indptr, X_s_gpu.indices, X_s_gpu.data,
            nrows=n_s, ncols=n_self, out=C_ss_s,
            stream=cp.cuda.get_current_stream().ptr,
        )
        _spca.copy_upper_to_lower(out=C_ss_s, ncols=n_self,
                                   stream=cp.cuda.get_current_stream().ptr)
        C_ss_s -= n_s * cp.outer(mu_x_s, mu_x_s)
        C_ss   += (C_ss_s / sx_s[:, None]) / sx_s[None, :]
        del C_ss_s

        C_hh_s = cp.zeros((n_nbr, n_nbr), dtype=cp.float32)
        C_sh_s = cp.zeros((n_self, n_nbr), dtype=cp.float32)
        for i in range(0, n_s, chunk_size):
            j       = min(i + chunk_size, n_s)
            h_chunk = (W_s_gpu[i:j, :] @ X_s_gpu).toarray()
            x_chunk = X_s_gpu[i:j].toarray()
            C_hh_s += h_chunk.T @ h_chunk
            C_sh_s += x_chunk.T @ h_chunk

        C_hh_s -= n_s * cp.outer(mu_n_s, mu_n_s)
        C_sh_s -= n_s * cp.outer(mu_x_s, mu_n_s)
        C_hh   += (C_hh_s / sn_s[:, None]) / sn_s[None, :]
        C_sh   += (C_sh_s / sx_s[:, None]) / sn_s[None, :]
        del C_hh_s, C_sh_s, X_s_gpu, W_s_gpu
        cp.cuda.Stream.null.synchronize()

    C_ss *= scale_own ** 2
    C_hh *= scale_nbr ** 2
    C_sh *= scale_own * scale_nbr

    n_feat = n_self + n_nbr
    C = cp.empty((n_feat, n_feat), dtype=cp.float32)
    C[:n_self, :n_self] = C_ss
    C[:n_self, n_self:] = C_sh
    C[n_self:, :n_self] = C_sh.T
    C[n_self:, n_self:] = C_hh
    return C


def _pca_sparse_cov_gpu_asym(X_csr_self, X_csr_nbr, W, lambda_val,
                               X_mean, X_std, nbr_mean, nbr_std, n_comps,
                               zscore, sample_slices,
                               mu_self, std_self, mu_nbr, std_nbr,
                               chunk_size=20_000,
                               input_zarr=None, n_genes=None, gene_indices=None):
    """BANKSY PCA via explicit sparse covariance + GPU eigh.
    Same algorithm as sparse_cov_cpu with GPU acceleration:
    - gram_csr_upper CUDA kernel for XᵀX (no dense X, upper-triangle atomicAdd)
    - cuBLAS DGEMM for h_chunk products
    - cp.linalg.eigh (cuSOLVER) for eigendecomposition
    - Vectorised per-sample projection on GPU for recovery
    VRAM flat: W_gpu constant; X_s and h_chunk freed per sample.
    """
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    rmm.reinitialize(managed_memory=False, pool_allocator=False, devices=0)
    cp.cuda.set_allocator(rmm_cupy_allocator)

    _zarr_mode = input_zarr is not None
    if _zarr_mode:
        n_self  = len(gene_indices) if gene_indices is not None else n_genes
        n_nbr   = n_self
        n_cells = sample_slices[-1][1]
    else:
        n_cells = X_csr_self.shape[0]
        n_self  = X_csr_self.shape[1]
        n_nbr   = X_csr_nbr.shape[1]
    if sample_slices is None:
        sample_slices = [(0, n_cells)]
    n_feat    = n_self + n_nbr
    scale_own = float(np.sqrt(1.0 - lambda_val))
    scale_nbr = float(np.sqrt(lambda_val))

    print(f"  Transferring W ({W.shape}) to GPU ...", flush=True)
    W_gpu = csp.csr_matrix(W.astype(np.float32))

    print(f"\n  Building BANKSY covariance ({n_feat}×{n_feat}) on GPU ...", flush=True)
    _mon_cov = MemMonitor(interval=0.5).start()
    if _zarr_mode:
        C_gpu = _build_banksy_cov_sparse_zarr_gpu(
            input_zarr, n_genes, W_gpu, zscore, sample_slices,
            X_mean, X_std, nbr_mean, nbr_std,
            mu_self, std_self, mu_nbr, std_nbr,
            lambda_val, chunk_size, gene_indices=gene_indices,
        )
    else:
        C_gpu = _build_banksy_cov_sparse_gpu(
            X_csr_self, X_csr_nbr, W_gpu, zscore, sample_slices,
            X_mean, X_std, nbr_mean, nbr_std,
            mu_self, std_self, mu_nbr, std_nbr,
            lambda_val, chunk_size,
        )
    _mon_cov.stop()
    print(f"  Cov built.  peak_rss={_mon_cov.peak_rss:.2f} GB", flush=True)

    print(f"  eigh (cuSOLVER) on ({n_feat}×{n_feat}) ...", flush=True)
    _mon_eig = MemMonitor(interval=0.5).start()
    eigenvalues_gpu, U_feat_gpu = cp.linalg.eigh(C_gpu)
    _mon_eig.stop()
    del C_gpu
    print(f"  eigh done.  peak_rss={_mon_eig.peak_rss:.2f} GB", flush=True)

    # eigh returns ascending order → reverse to descending
    eigenvalues = cp.asnumpy(eigenvalues_gpu[::-1][:n_comps])
    U_feat_gpu  = cp.flip(U_feat_gpu, axis=1)[:, :n_comps].astype(cp.float32)  # n_feat × n_comps

    # Recovery: vectorised per-sample on GPU
    U1_gpu = U_feat_gpu[:n_self, :] * scale_own  # n_self × n_comps
    U2_gpu = U_feat_gpu[n_self:, :] * scale_nbr  # n_nbr  × n_comps

    if zscore == 'global':
        sx_g     = cp.asarray((X_std   + 1e-10).astype(np.float32))
        sn_g     = cp.asarray((nbr_std + 1e-10).astype(np.float32))
        U1s_g    = U1_gpu / sx_g[:, None]
        U2s_g    = U2_gpu / sn_g[:, None]
        mu_x_U1s = cp.asarray(X_mean.astype(np.float32))    @ U1s_g  # (n_comps,)
        mu_n_U2s = cp.asarray(nbr_mean.astype(np.float32))  @ U2s_g

    print(f"  Recovery (GPU, vectorised): {len(sample_slices)} samples ...", flush=True)
    embeddings_gpu = cp.zeros((n_cells, n_comps), dtype=cp.float32)

    for s, (start, end) in enumerate(sample_slices):
        if _zarr_mode:
            X_s_cpu = _load_csr_from_zarr(input_zarr, start, end, n_genes,
                                           gene_indices).astype(np.float32)
        else:
            X_s_cpu = _csr_row_view(X_csr_self, start, end).astype(np.float32)
        X_s_gpu = csp.csr_matrix(X_s_cpu); del X_s_cpu
        W_s_gpu = W_gpu[start:end, start:end]

        if zscore == 'per_batch':
            sx       = cp.asarray((std_self[s] + 1e-10).astype(np.float32))
            sn       = cp.asarray((std_nbr[s]  + 1e-10).astype(np.float32))
            U1s      = U1_gpu / sx[:, None]
            U2s      = U2_gpu / sn[:, None]
            mu_x_U1s = cp.asarray(mu_self[s].astype(np.float32)) @ U1s
            mu_n_U2s = cp.asarray(mu_nbr[s].astype(np.float32))  @ U2s
        else:
            U1s = U1s_g
            U2s = U2s_g

        embeddings_gpu[start:end, :] = (
            X_s_gpu @ U1s - mu_x_U1s
            + W_s_gpu @ (X_s_gpu @ U2s) - mu_n_U2s
        )
        del X_s_gpu, W_s_gpu

    del W_gpu
    embeddings = cp.asnumpy(embeddings_gpu)
    del embeddings_gpu

    ev        = eigenvalues / n_cells
    total_var = n_self * (1.0 - lambda_val) + n_nbr * lambda_val
    evr       = ev / total_var
    return embeddings, ev, evr


# ---------------------------------------------------------------------------
# Solver: linop_gpu
# ---------------------------------------------------------------------------

def _pca_linop_gpu_asym(X_csr_self, X_csr_nbr, W, lambda_val,
                         X_mean, X_std, nbr_mean, nbr_std, n_comps,
                         zscore, sample_slices,
                         mu_self, std_self, mu_nbr, std_nbr,
                         dtype_str='float32'):
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import LinearOperator as CuLinOp, svds as cu_svds

    dtype    = cp.float32 if dtype_str == 'float32' else cp.float64
    np_dtype = np.float32 if dtype_str == 'float32' else np.float64
    n_cells  = X_csr_self.shape[0]
    n_self   = X_csr_self.shape[1]
    n_nbr    = X_csr_nbr.shape[1]
    scale_own = dtype(np.sqrt(1.0 - lambda_val))
    scale_nbr = dtype(np.sqrt(lambda_val))

    print("  Transferring W to GPU ...", flush=True)
    W_gpu      = csp.csr_matrix(W.astype(np_dtype))
    Wt_gpu     = csp.csr_matrix(W_gpu.T)

    if zscore == 'global':
        print("  Transferring X to GPU (global mode) ...", flush=True)
        X_self_gpu = csp.csr_matrix(X_csr_self.astype(np_dtype))
        X_nbr_gpu  = csp.csr_matrix(X_csr_nbr.astype(np_dtype))
        sx         = cp.asarray((X_std    + 1e-10).astype(np_dtype))
        sn         = cp.asarray((nbr_std  + 1e-10).astype(np_dtype))
        mu_over_sx = cp.asarray((X_mean   / (X_std   + 1e-10)).astype(np_dtype))
        mu_over_sn = cp.asarray((nbr_mean / (nbr_std + 1e-10)).astype(np_dtype))
        X_mean_g   = cp.asarray(X_mean.astype(np_dtype))
        nbr_mean_g = cp.asarray(nbr_mean.astype(np_dtype))

        def matvec(v):
            v    = cp.asarray(v, dtype=dtype).ravel()
            vsum = v.sum()
            top    = (X_self_gpu.T @ v    - X_mean_g   * vsum) / sx * scale_own
            Wt_v   = Wt_gpu @ v
            bottom = (X_nbr_gpu.T  @ Wt_v - nbr_mean_g * vsum) / sn * scale_nbr
            return cp.concatenate([top, bottom])

        def rmatvec(u):
            u  = cp.asarray(u, dtype=dtype).ravel()
            u1 = u[:n_self] * scale_own
            u2 = u[n_self:] * scale_nbr
            top    = X_self_gpu @ (u1 / sx) - float(mu_over_sx @ u1)
            u2s    = u2 / sn
            bottom = W_gpu @ (X_nbr_gpu @ u2s) - float(mu_over_sn @ u2)
            return top + bottom

    else:  # per_batch — precompute block slices on GPU
        print(f"  Building {len(sample_slices)} GPU block slices ...", flush=True)
        X_self_gpu_blocks = [csp.csr_matrix(X_csr_self[s:e].astype(np_dtype))
                             for s, e in sample_slices]
        X_nbr_gpu_blocks  = [csp.csr_matrix(X_csr_nbr[s:e].astype(np_dtype))
                             for s, e in sample_slices]
        W_gpu_blocks      = [csp.csr_matrix(W[s:e, s:e].astype(np_dtype))
                             for s, e in sample_slices]
        mu_self_gpu  = [cp.asarray(mu_self[s].astype(np_dtype))
                        for s in range(len(sample_slices))]
        std_self_gpu = [cp.asarray(std_self[s].astype(np_dtype))
                        for s in range(len(sample_slices))]
        mu_nbr_gpu   = [cp.asarray(mu_nbr[s].astype(np_dtype))
                        for s in range(len(sample_slices))]
        std_nbr_gpu  = [cp.asarray(std_nbr[s].astype(np_dtype))
                        for s in range(len(sample_slices))]
        def matvec(v):
            v        = cp.asarray(v, dtype=dtype).ravel()
            Wt_v     = Wt_gpu @ v
            res_self = cp.zeros(n_self, dtype=dtype)
            res_nbr  = cp.zeros(n_nbr,  dtype=dtype)
            for s, (start, end) in enumerate(sample_slices):
                v_s    = v[start:end]
                vsum_s = v_s.sum()
                sx_s   = std_self_gpu[s] + dtype(1e-10)
                sn_s   = std_nbr_gpu[s]  + dtype(1e-10)
                res_self += (X_self_gpu_blocks[s].T @ v_s
                             - mu_self_gpu[s] * vsum_s) / sx_s
                res_nbr  += (X_nbr_gpu_blocks[s].T @ Wt_v[start:end]
                             - mu_nbr_gpu[s]  * vsum_s) / sn_s
            return cp.concatenate([res_self * scale_own, res_nbr * scale_nbr])

        def rmatvec(u):
            u   = cp.asarray(u, dtype=dtype).ravel()
            u1  = u[:n_self] * scale_own
            u2  = u[n_self:] * scale_nbr
            res = cp.zeros(n_cells, dtype=dtype)
            for s, (start, end) in enumerate(sample_slices):
                sx_s = std_self_gpu[s] + dtype(1e-10)
                sn_s = std_nbr_gpu[s]  + dtype(1e-10)
                u1s  = u1 / sx_s
                u2s  = u2 / sn_s
                res[start:end] += (X_self_gpu_blocks[s] @ u1s
                                   - float(mu_self_gpu[s] @ u1s))
                res[start:end] += (W_gpu_blocks[s] @ (X_nbr_gpu_blocks[s] @ u2s)
                                   - float(mu_nbr_gpu[s] @ u2s))
            return res

    M_op = CuLinOp(shape=(n_self + n_nbr, n_cells),
                   matvec=matvec, rmatvec=rmatvec, dtype=dtype)
    print(f"  CuPy LinearOperator shape: {M_op.shape}  k={n_comps}  dtype={dtype}",
          flush=True)
    _monitor = MemMonitor(interval=0.5).start()
    _, S, Vt = cu_svds(M_op, k=n_comps)
    _monitor.stop()

    del M_op, W_gpu, Wt_gpu
    if zscore == 'global':
        del X_self_gpu, X_nbr_gpu
        del sx, sn, mu_over_sx, mu_over_sn, X_mean_g, nbr_mean_g
    else:
        del X_self_gpu_blocks, X_nbr_gpu_blocks, W_gpu_blocks
        del mu_self_gpu, std_self_gpu, mu_nbr_gpu, std_nbr_gpu
    cp.get_default_memory_pool().free_all_blocks()

    S  = cp.asnumpy(S)
    Vt = cp.asnumpy(Vt)
    idx = np.argsort(S)[::-1]
    S, Vt = S[idx], Vt[idx, :]
    embeddings = Vt.T * S
    ev        = S ** 2 / n_cells
    total_var = n_self * (1.0 - lambda_val) + n_nbr * lambda_val
    evr       = ev / total_var
    return embeddings, ev, evr


# ---------------------------------------------------------------------------
# Solver: dask_cpu
# ---------------------------------------------------------------------------

def _pca_dask_cpu_asym(adata, X_csr_self, X_csr_nbr, W, lambda_val,
                        X_mean, X_std, nbr_mean, nbr_std,
                        n_comps, chunk_size, zarr_dir, genes_self, genes_nbr,
                        zscore, sample_slices,
                        mu_self, std_self, mu_nbr, std_nbr,
                        input_zarr=None, gene_indices=None):
    import dask
    import dask.array as da
    import scanpy as sc

    if input_zarr is not None and zscore == 'per_batch':
        # zarr-streaming v2: X_sample cached per sample (1 zarr read per sample
        # instead of 1 per sub-chunk). Requires synchronous scheduler.
        n_genes = mu_self.shape[1]
        _zarr_sample_cache.clear()
        banksy_dask = _build_banksy_dask_asym_persample_zarr_v2(
            input_zarr, n_genes, W, lambda_val,
            mu_self, std_self, mu_nbr, std_nbr, sample_slices,
            chunk_size=chunk_size, gene_indices=gene_indices,
        )
    elif input_zarr is not None and zscore == 'global':
        n_genes = len(X_mean)
        _zarr_sample_cache.clear()
        banksy_dask = _build_banksy_dask_asym_global_zarr(
            input_zarr, n_genes, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std, sample_slices,
            chunk_size=chunk_size, gene_indices=gene_indices,
        )
    elif zscore == 'global':
        banksy_dask = _build_banksy_dask_asym(
            X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std, chunk_size,
        )
    else:
        banksy_dask = _build_banksy_dask_asym_persample(
            X_csr_self, X_csr_nbr, W, lambda_val,
            mu_self, std_self, mu_nbr, std_nbr, sample_slices,
            chunk_size=chunk_size,
        )

    if zarr_dir is not None:
        print(f"  Writing BANKSY zarr to {zarr_dir} ...", flush=True)
        banksy_dask.to_zarr(zarr_dir, overwrite=True)
        banksy_dask = da.from_zarr(zarr_dir)

    banksy_adata = _make_banksy_adata_asym(adata, genes_self, genes_nbr, banksy_dask)
    dask.config.set(scheduler='synchronous')
    _monitor = MemMonitor(interval=0.5).start()
    sc.pp.pca(banksy_adata, n_comps=n_comps, svd_solver='covariance_eigh')
    _monitor.stop()

    pcs = np.array(banksy_adata.obsm['X_pca'])
    ev  = np.array(banksy_adata.uns['pca']['variance'])
    evr = np.array(banksy_adata.uns['pca']['variance_ratio'])
    return pcs, ev, evr


# ---------------------------------------------------------------------------
# Solver: dask_gpu
# ---------------------------------------------------------------------------

def _pca_dask_gpu_asym(adata, X_csr_self, X_csr_nbr, W, lambda_val,
                        X_mean, X_std, nbr_mean, nbr_std,
                        n_comps, chunk_size, genes_self, genes_nbr,
                        zscore, sample_slices,
                        mu_self, std_self, mu_nbr, std_nbr,
                        input_zarr=None, gene_indices=None):
    import gc
    import cupy as cp
    import rapids_singlecell as rsc
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    import dask

    rmm.reinitialize(managed_memory=False, pool_allocator=False, devices=0)
    cp.cuda.set_allocator(rmm_cupy_allocator)

    if input_zarr is not None and zscore == 'per_batch':
        n_genes = mu_self.shape[1]
        _zarr_sample_cache.clear()
        banksy_dask = _build_banksy_dask_asym_persample_zarr_v2(
            input_zarr, n_genes, W, lambda_val,
            mu_self, std_self, mu_nbr, std_nbr, sample_slices,
            chunk_size=chunk_size, gene_indices=gene_indices,
        )
    elif input_zarr is not None and zscore == 'global':
        n_genes = len(X_mean)
        _zarr_sample_cache.clear()
        banksy_dask = _build_banksy_dask_asym_global_zarr(
            input_zarr, n_genes, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std, sample_slices,
            chunk_size=chunk_size, gene_indices=gene_indices,
        )
    elif zscore == 'global':
        banksy_dask = _build_banksy_dask_asym(
            X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std, chunk_size,
        )
    else:
        banksy_dask = _build_banksy_dask_asym_persample(
            X_csr_self, X_csr_nbr, W, lambda_val,
            mu_self, std_self, mu_nbr, std_nbr, sample_slices,
            chunk_size=chunk_size,
        )

    banksy_adata = _make_banksy_adata_asym(adata, genes_self, genes_nbr, banksy_dask)
    dask.config.set(scheduler='synchronous')
    rsc.get.anndata_to_GPU(banksy_adata)

    _monitor = MemMonitor(interval=0.5).start()
    rsc.pp.pca(banksy_adata, n_comps=n_comps, svd_solver='covariance_eigh',
               zero_center=False)
    _monitor.stop()

    pca_result = banksy_adata.obsm['X_pca']
    if hasattr(pca_result, 'compute'):
        pca_result = pca_result.compute()
    if isinstance(pca_result, cp.ndarray):
        pca_result = cp.asnumpy(pca_result)
    pcs = np.array(pca_result)

    evr_raw = banksy_adata.uns['pca']['variance_ratio']
    if isinstance(evr_raw, cp.ndarray):
        evr_raw = cp.asnumpy(evr_raw)
    evr = np.array(evr_raw)

    ev_raw = banksy_adata.uns['pca'].get('variance', np.zeros_like(evr))
    if hasattr(ev_raw, 'compute'):
        ev_raw = ev_raw.compute()
    if isinstance(ev_raw, cp.ndarray):
        ev_raw = cp.asnumpy(ev_raw)
    ev = np.array(ev_raw)

    del banksy_adata, banksy_dask
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return pcs, ev, evr


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_banksy_pca(adata, W,
                       genes_self=None,
                       genes_nbr=None,
                       lambda_val=0.2,
                       n_comps=50,
                       chunk_size=50_000,
                       zscore='global',
                       batch_key='sample',
                       solver='linop_cpu',
                       dtype='float32',
                       zarr_dir=None,
                       input_zarr=None,
                       linop_svd_solver='propack',
                       gene_indices=None):
    """BANKSY PCA with independent gene filters, optional per-sample Z-scoring,
    and multiple solver backends.

    Parameters
    ----------
    adata : AnnData (n_cells, n_all_genes)
        Normalized, log1p'd. Cell order must match W rows/cols.
        When zscore='per_batch', must be sorted by batch_key (contiguous
        per sample) — as returned by prepare_multi_adata / _build_W_block.
    W : scipy sparse (n_cells, n_cells)
        Block-diagonal spatial weight matrix.
    genes_self : array-like of str, or None
        Gene names for X_self. None = all genes.
    genes_nbr : array-like of str, or None
        Gene names for X_nbr (H0 = W @ X_nbr). None = all genes.
    lambda_val : float
    n_comps : int
    chunk_size : int
        Row-chunk size for streaming H0 mean/std (linop solvers) or dask chunks.
    zscore : 'global' or 'per_batch'
        'global'     — mean/std across all cells (default).
        'per_batch' — mean/std within each sample independently.
    batch_key : str
        obs column identifying samples. Used only when zscore='per_batch'.
    solver : str
        One of: 'linop_cpu', 'linop_gpu', 'dask_cpu', 'dask_gpu'.
    dtype : str
        'float32' or 'float64'. GPU solvers only.
    zarr_dir : str or None
        If set, dask_cpu solver caches the BANKSY matrix to this zarr path
        before PCA (avoids recomputing the dask graph twice).
    input_zarr : str or None
        If set, dask_cpu+per_batch: X is read directly from this zarr inside
        each dask task — adata.X is never needed. Keeps peak RAM ~constant
        regardless of n_cells. Only valid with zscore='per_batch'.

    Returns
    -------
    embeddings : ndarray (n_cells, n_comps)
    explained_variance : ndarray (n_comps,)
    explained_variance_ratio : ndarray (n_comps,)
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got '{solver}'")
    if zscore not in ('global', 'per_batch'):
        raise ValueError(f"zscore must be 'global' or 'per_batch', got '{zscore}'")

    # When input_zarr is set for dask_{cpu,gpu}, X is never loaded into RAM.
    # Covers both per_batch (persample_zarr_v2) and global (global_zarr) paths.
    _use_zarr_x = (input_zarr is not None
                   and solver in ('dask_cpu', 'dask_gpu', 'sparse_cov_cpu', 'sparse_cov_gpu'))

    if _use_zarr_x:
        import zarr as _zarr_mod
        _z_meta = _zarr_mod.open_group(input_zarr, mode='r', zarr_format=2)
        n_genes_zarr = int(_z_meta['X/indptr'].shape[0] - 1
                           if 'X/indptr' in _z_meta
                           else _z_meta['var/_index'].shape[0])
        # n_genes from var/_index is more reliable
        n_genes_zarr = int(_z_meta['var/_index'].shape[0])
        del _z_meta
        n_cells = len(adata)
        n_self  = len(gene_indices) if gene_indices is not None else n_genes_zarr
        n_nbr   = n_self
        X_csr_self = X_csr_nbr = None
        _fp = np.float32
    else:
        _X_self = adata[:, genes_self].X if genes_self is not None else adata.X
        _X_nbr  = adata[:, genes_nbr].X if genes_nbr is not None else adata.X
        # tocsr(copy=False) returns self when already CSR — no copy of bulk data.
        # sparse.csr_matrix(A) ALWAYS copies in scipy 1.17+ (confirmed); avoid it here.
        X_csr_self = (_X_self.tocsr(copy=False) if sparse.issparse(_X_self)
                      else sparse.csr_matrix(_X_self))
        if not np.issubdtype(X_csr_self.dtype, np.floating):
            X_csr_self = X_csr_self.astype(np.float64)
        # Preserve float32 input (saves ~100 GB at 100M+ cells vs float64 upcast).
        _fp = np.float32 if X_csr_self.dtype == np.float32 else np.float64
        # Share when genes_self == genes_nbr — avoids a second no-copy wrapper or conversion.
        if _X_nbr is _X_self:
            X_csr_nbr = X_csr_self
        else:
            X_csr_nbr = (_X_nbr.tocsr(copy=False) if sparse.issparse(_X_nbr)
                         else sparse.csr_matrix(_X_nbr))
            if not np.issubdtype(X_csr_nbr.dtype, np.floating):
                X_csr_nbr = X_csr_nbr.astype(np.float64)
        del _X_self, _X_nbr
        n_cells = X_csr_self.shape[0]
        n_self  = X_csr_self.shape[1]
        n_nbr   = X_csr_nbr.shape[1]

    n_comps = min(n_comps, min(n_cells, n_self + n_nbr) - 1)

    print(f"[run_banksy_pca] solver={solver}  lambda={lambda_val}  "
          f"n_comps={n_comps}  n_self={n_self}  n_nbr={n_nbr}  zscore={zscore}"
          f"  input_zarr={'yes' if _use_zarr_x else 'no'}",
          flush=True)

    # ------------------------------------------------------------------
    # Compute stats
    # ------------------------------------------------------------------

    X_mean = X_std = nbr_mean = nbr_std = None
    sample_slices = mu_self = std_self = mu_nbr = std_nbr = None

    if zscore == 'global':
        if _use_zarr_x:
            sample_slices = _get_sample_slices(adata, batch_key)
            print(f"Computing global stats from zarr ({len(sample_slices)} samples) ...",
                  flush=True)
            X_mean, X_std, nbr_mean, nbr_std = _global_stats_from_zarr(
                input_zarr, n_genes_zarr, W, sample_slices, chunk_size,
                gene_indices=gene_indices,
            )
        else:
            print("Computing X_self mean/std (global) ...", flush=True)
            X_mean, X_std = _sparse_mean_std(X_csr_self)
            print(f"Streaming H0_nbr mean/std (global, chunk_size={chunk_size}) ...",
                  flush=True)
            nbr_mean, nbr_std = _nbr_mean_std(W, X_csr_nbr, chunk_size)
    else:
        if batch_key not in adata.obs.columns:
            raise ValueError(
                f"batch_key='{batch_key}' not found in obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        sample_slices = _get_sample_slices(adata, batch_key)
        n_samples     = len(sample_slices)
        print(f"Computing per-sample stats ({n_samples} samples) ...", flush=True)
        if _use_zarr_x:
            mu_self, std_self, mu_nbr, std_nbr = _per_batch_stats_from_zarr(
                input_zarr, n_genes_zarr, W, sample_slices, chunk_size,
                gene_indices=gene_indices,
            )
        else:
            mu_self, std_self, mu_nbr, std_nbr = _per_batch_stats(
                X_csr_self, X_csr_nbr, W, sample_slices, chunk_size
            )

    # ------------------------------------------------------------------
    # Dispatch to solver
    # ------------------------------------------------------------------

    if solver == 'linop_cpu':
        return _pca_linop_cpu_asym(
            X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std, n_comps,
            zscore, sample_slices, mu_self, std_self, mu_nbr, std_nbr,
            fp_dtype=_fp,
            linop_svd_solver=linop_svd_solver,
        )

    if solver == 'linop_gpu':
        return _pca_linop_gpu_asym(
            X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std, n_comps,
            zscore, sample_slices, mu_self, std_self, mu_nbr, std_nbr,
            dtype_str=dtype,
        )

    if solver == 'dask_cpu':
        return _pca_dask_cpu_asym(
            adata, X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std,
            n_comps, chunk_size, zarr_dir, genes_self, genes_nbr,
            zscore, sample_slices, mu_self, std_self, mu_nbr, std_nbr,
            input_zarr=input_zarr, gene_indices=gene_indices,
        )

    if solver == 'dask_gpu':
        return _pca_dask_gpu_asym(
            adata, X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std,
            n_comps, chunk_size, genes_self, genes_nbr,
            zscore, sample_slices, mu_self, std_self, mu_nbr, std_nbr,
            input_zarr=input_zarr, gene_indices=gene_indices,
        )

    if solver == 'sparse_cov_cpu':
        return _pca_sparse_cov_cpu_asym(
            X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std, n_comps,
            zscore, sample_slices, mu_self, std_self, mu_nbr, std_nbr,
            chunk_size=chunk_size, fp_dtype=_fp,
            input_zarr=input_zarr if _use_zarr_x else None,
            n_genes=n_genes_zarr if _use_zarr_x else None,
            gene_indices=gene_indices,
        )

    if solver == 'sparse_cov_gpu':
        return _pca_sparse_cov_gpu_asym(
            X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std, n_comps,
            zscore, sample_slices, mu_self, std_self, mu_nbr, std_nbr,
            chunk_size=chunk_size,
            input_zarr=input_zarr if _use_zarr_x else None,
            n_genes=n_genes_zarr if _use_zarr_x else None,
            gene_indices=gene_indices,
        )

