from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _sanitize_matrix(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mean = df.mean()
    std = df.std(ddof=0).replace(0, 1.0)
    scaled = (df - mean) / std
    return scaled, mean, std


def pca_decompose(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    x = _sanitize_matrix(df.to_numpy(dtype=float))
    n_samples = len(x)
    try:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        explained_variance = (s ** 2) / (n_samples - 1) if n_samples > 1 else s ** 2
    except np.linalg.LinAlgError:
        if n_samples <= 1:
            u = np.zeros((n_samples, x.shape[1]))
            s = np.zeros((min(x.shape),))
            vt = np.zeros((x.shape[1], x.shape[1]))
            explained_variance = s ** 2
        else:
            cov = (x.T @ x) / (n_samples - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            s = np.sqrt(np.clip(eigvals * (n_samples - 1), 0.0, None))
            vt = eigvecs.T
            denom = np.where(s > 0, s, 1.0)
            u = (x @ eigvecs) / denom
            explained_variance = eigvals

    total = explained_variance.sum()
    explained_ratio = explained_variance / total if total != 0 else explained_variance
    return {
        "u": u,
        "s": s,
        "vt": vt,
        "explained_variance": explained_variance,
        "explained_ratio": explained_ratio,
    }


def choose_components(explained_ratio: np.ndarray, threshold: float) -> int:
    if explained_ratio.size == 0:
        return 0
    cumulative = np.cumsum(explained_ratio)
    return int(np.searchsorted(cumulative, threshold) + 1)


def reconstruct(u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros((u.shape[0], vt.shape[1]))
    u_k = u[:, :k]
    s_k = s[:k]
    vt_k = vt[:k, :]
    return (u_k * s_k) @ vt_k


def pca_denoise(
    df: pd.DataFrame,
    variance_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    scaled, mean, std = standardize(df)
    decomp = pca_decompose(scaled)
    k = choose_components(decomp["explained_ratio"], variance_threshold)
    recon = reconstruct(decomp["u"], decomp["s"], decomp["vt"], k)
    recon_df = pd.DataFrame(recon, columns=df.columns, index=df.index)
    denoised = recon_df * std + mean

    error = (scaled - recon_df).pow(2).mean().to_dict()
    decomp["components_used"] = k
    decomp["reconstruction_error"] = error
    return denoised, decomp
