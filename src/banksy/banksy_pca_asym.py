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
#   'per_sample' — mean/std within each sample independently
#
# Gene filter modes (all combinations supported):
#   genes_self=None, genes_nbr=None  →  standard symmetric BANKSY
#   genes_self=HVGs, genes_nbr=None  →  HVG self, full neighborhood
#   genes_self=A,    genes_nbr=B     →  fully independent gene sets
#
# Usage:
#   from banksy_pca_asym import run_pca_asymmetric
#
#   adata_full, W = prepare_multi_adata(merged_adata, normalize=False, n_hvg=None)
#
#   # symmetric, per-sample Z-score
#   embeddings, ev, evr = run_pca_asymmetric(adata_full, W,
#                                             zscore='per_sample',
#                                             batch_key='sample')
#
#   # asymmetric HVG self + full H0, global Z-score
#   embeddings, ev, evr = run_pca_asymmetric(adata_full, W,
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

SOLVERS = ('linop_cpu', 'linop_gpu', 'linop_gpu_propack', 'dask_cpu', 'dask_gpu')


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


def _per_sample_stats(X_csr_self, X_csr_nbr, W, sample_slices, chunk_size):
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

def _load_csr_from_zarr(zarr_path, row_start, row_end, n_genes):
    """Load CSR rows [row_start:row_end] from a zarr X store (data/indices/indptr)."""
    import zarr as _zarr
    z       = _zarr.open_group(zarr_path, mode='r', zarr_format=2)
    iptr    = z['X/indptr'][row_start:row_end + 1]
    nnz0    = int(iptr[0])
    nnz1    = int(iptr[-1])
    data    = z['X/data'][nnz0:nnz1].astype(np.float32)
    indices = z['X/indices'][nnz0:nnz1]
    indptr  = (iptr - nnz0).astype(np.int32)
    return sparse.csr_matrix((data, indices, indptr),
                              shape=(row_end - row_start, n_genes))


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


def _per_sample_stats_from_zarr(input_zarr, n_genes, W, sample_slices, chunk_size):
    """Compute per-sample mean/std by loading one sample at a time from zarr.
    Peak memory = one sample's X (~3 GB for the largest sample) rather than all of X.
    """
    n_samples = len(sample_slices)
    mu_self  = np.zeros((n_samples, n_genes), dtype=np.float64)
    std_self = np.ones( (n_samples, n_genes), dtype=np.float64)
    mu_nbr   = np.zeros((n_samples, n_genes), dtype=np.float64)
    std_nbr  = np.ones( (n_samples, n_genes), dtype=np.float64)
    for s, (start, end) in enumerate(sample_slices):
        print(f"  [sample {s+1}/{n_samples}] cells {start}:{end}", flush=True)
        X_s = _load_csr_from_zarr(input_zarr, start, end, n_genes)
        mu_self[s], std_self[s] = _sparse_mean_std(X_s)
        W_s = W[start:end, start:end]
        mu_nbr[s], std_nbr[s]  = _nbr_mean_std(W_s, X_s,
                                                chunk_size=chunk_size,
                                                verbose=False)
        del X_s
    std_self[std_self == 0] = 1.0
    std_nbr[ std_nbr  == 0] = 1.0
    return mu_self, std_self, mu_nbr, std_nbr


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
                         fp_dtype=np.float64):
    from scipy.sparse.linalg import LinearOperator, svds
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
            v    = np.asarray(v, dtype=fp_dtype)
            vsum = v.sum()
            top    = (X_csr_self.T @ v    - X_mean_f   * vsum) / sx * scale_own
            Wt_v   = Wt @ v
            bottom = (X_csr_nbr.T  @ Wt_v - nbr_mean_f * vsum) / sn * scale_nbr
            return np.concatenate([top, bottom])

        def rmatvec(u):
            u  = np.asarray(u, dtype=fp_dtype)
            u1 = u[:n_self] * scale_own
            u2 = u[n_self:] * scale_nbr
            top    = X_csr_self @ (u1 / sx) - fp_dtype(mu_over_sx @ u1)
            u2s    = u2 / sn
            bottom = W @ (X_csr_nbr @ u2s) - fp_dtype(mu_over_sn @ u2)
            return top + bottom

    else:  # per_sample
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
            v        = np.asarray(v, dtype=fp_dtype)
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
            u   = np.asarray(u, dtype=fp_dtype)
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

    M_op = LinearOperator(shape=(n_self + n_nbr, n_cells),
                          matvec=matvec, rmatvec=rmatvec, dtype=fp_dtype)
    print(f"  LinearOperator shape: {M_op.shape}  k={n_comps}  solver=propack",
          flush=True)
    _monitor = MemMonitor(interval=0.5).start()
    U, S, Vt = svds(M_op, k=n_comps, solver='propack')
    _monitor.stop()

    idx = np.argsort(S)[::-1]
    S, Vt = S[idx], Vt[idx, :]
    embeddings = Vt.T * S
    ev        = S ** 2 / n_cells
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
                         dtype_str='float32', use_propack=False):
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import LinearOperator as CuLinOp
    if use_propack:
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from cupy_propack import svds as cu_svds
    else:
        from cupyx.scipy.sparse.linalg import svds as cu_svds

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

    else:  # per_sample — precompute block slices on GPU
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
    if use_propack:
        _, S, Vt = cu_svds(M_op, k=n_comps, verbose=True)
    else:
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
                        input_zarr=None):
    import dask
    import dask.array as da
    import scanpy as sc

    if input_zarr is not None and zscore == 'per_sample':
        # zarr-streaming path: X never fully in RAM; each task reads its chunk.
        # mu_self has shape (n_samples, n_genes) — reliable source of n_genes.
        n_genes = mu_self.shape[1]
        banksy_dask = _build_banksy_dask_asym_persample_zarr(
            input_zarr, n_genes, W, lambda_val,
            mu_self, std_self, mu_nbr, std_nbr, sample_slices,
            chunk_size=chunk_size,
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
                        mu_self, std_self, mu_nbr, std_nbr):
    import gc
    import cupy as cp
    import rapids_singlecell as rsc
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    import dask

    rmm.reinitialize(managed_memory=False, pool_allocator=False, devices=0)
    cp.cuda.set_allocator(rmm_cupy_allocator)

    if zscore == 'global':
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

def run_pca_asymmetric(adata, W,
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
                       input_zarr=None):
    """BANKSY PCA with independent gene filters, optional per-sample Z-scoring,
    and multiple solver backends.

    Parameters
    ----------
    adata : AnnData (n_cells, n_all_genes)
        Normalized, log1p'd. Cell order must match W rows/cols.
        When zscore='per_sample', must be sorted by batch_key (contiguous
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
    zscore : 'global' or 'per_sample'
        'global'     — mean/std across all cells (default).
        'per_sample' — mean/std within each sample independently.
    batch_key : str
        obs column identifying samples. Used only when zscore='per_sample'.
    solver : str
        One of: 'linop_cpu', 'linop_gpu', 'dask_cpu', 'dask_gpu'.
    dtype : str
        'float32' or 'float64'. GPU solvers only.
    zarr_dir : str or None
        If set, dask_cpu solver caches the BANKSY matrix to this zarr path
        before PCA (avoids recomputing the dask graph twice).
    input_zarr : str or None
        If set, dask_cpu+per_sample: X is read directly from this zarr inside
        each dask task — adata.X is never needed. Keeps peak RAM ~constant
        regardless of n_cells. Only valid with zscore='per_sample'.

    Returns
    -------
    embeddings : ndarray (n_cells, n_comps)
    explained_variance : ndarray (n_comps,)
    explained_variance_ratio : ndarray (n_comps,)
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got '{solver}'")
    if zscore not in ('global', 'per_sample'):
        raise ValueError(f"zscore must be 'global' or 'per_sample', got '{zscore}'")

    # When input_zarr is set for dask_cpu+per_sample, X is never loaded into RAM.
    # We get n_genes from zarr and skip X extraction entirely.
    _use_zarr_x = (input_zarr is not None
                   and solver == 'dask_cpu'
                   and zscore == 'per_sample')

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
        n_self  = n_genes_zarr
        n_nbr   = n_genes_zarr
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

    print(f"[run_pca_asymmetric] solver={solver}  lambda={lambda_val}  "
          f"n_comps={n_comps}  n_self={n_self}  n_nbr={n_nbr}  zscore={zscore}"
          f"  input_zarr={'yes' if _use_zarr_x else 'no'}",
          flush=True)

    # ------------------------------------------------------------------
    # Compute stats
    # ------------------------------------------------------------------

    X_mean = X_std = nbr_mean = nbr_std = None
    sample_slices = mu_self = std_self = mu_nbr = std_nbr = None

    if zscore == 'global':
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
        if input_zarr is not None:
            mu_self, std_self, mu_nbr, std_nbr = _per_sample_stats_from_zarr(
                input_zarr, n_self, W, sample_slices, chunk_size
            )
        else:
            mu_self, std_self, mu_nbr, std_nbr = _per_sample_stats(
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
        )

    if solver == 'linop_gpu':
        return _pca_linop_gpu_asym(
            X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std, n_comps,
            zscore, sample_slices, mu_self, std_self, mu_nbr, std_nbr,
            dtype_str=dtype,
        )

    if solver == 'linop_gpu_propack':
        return _pca_linop_gpu_asym(
            X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std, n_comps,
            zscore, sample_slices, mu_self, std_self, mu_nbr, std_nbr,
            dtype_str=dtype, use_propack=True,
        )

    if solver == 'dask_cpu':
        return _pca_dask_cpu_asym(
            adata, X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std,
            n_comps, chunk_size, zarr_dir, genes_self, genes_nbr,
            zscore, sample_slices, mu_self, std_self, mu_nbr, std_nbr,
            input_zarr=input_zarr,
        )

    if solver == 'dask_gpu':
        return _pca_dask_gpu_asym(
            adata, X_csr_self, X_csr_nbr, W, lambda_val,
            X_mean, X_std, nbr_mean, nbr_std,
            n_comps, chunk_size, genes_self, genes_nbr,
            zscore, sample_slices, mu_self, std_self, mu_nbr, std_nbr,
        )
