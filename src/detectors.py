"""Anomaly scoring rules over frozen features.

All scorers compare test features to a normal reference set. Higher score means
more anomalous. Image-level scorers operate on one descriptor per slice; the
patch scorer operates on the dense token grid and reduces per slice.
"""
import numpy as np
import torch
from sklearn.neighbors import LocalOutlierFactor


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _zfit(ref):
    return ref.mean(0, keepdims=True), ref.std(0, keepdims=True) + 1e-8


def knn_distance(bank, X, k=5):
    """Mean distance to the k nearest reference vectors."""
    dev = _device()
    b = torch.tensor(bank, device=dev).float()
    x = torch.tensor(X, device=dev).float()
    d = torch.cdist(x, b).topk(min(k, len(b)), largest=False).values.mean(1)
    return d.cpu().numpy()


def lof_score(bank, X, k=5):
    """Local Outlier Factor novelty score (higher = more anomalous)."""
    m = LocalOutlierFactor(n_neighbors=k, novelty=True)
    m.fit(bank)
    return -m.score_samples(X)


def centroid_distance(bank, X, n_clusters=16, seed=0):
    """Distance to the nearest k-means centroid of the reference set."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(bank)
    c = km.cluster_centers_
    dev = _device()
    d = torch.cdist(torch.tensor(X, device=dev).float(), torch.tensor(c, device=dev).float())
    return d.min(1).values.cpu().numpy()


def image_scores(ref, tn, ta, scorer=knn_distance, k=5):
    """Standardise by the reference, then score normal-test and abnormal-test."""
    mu, sd = _zfit(ref)
    r, n, a = (ref - mu) / sd, (tn - mu) / sd, (ta - mu) / sd
    return scorer(r, n, k=k), scorer(r, a, k=k)


def patch_scores(ref_p, tn_p, ta_p, k=5, bank_subsample=20000, seed=0):
    """Patch memory-bank score: a slice gets the max over its patches of the
    k-NN distance to a (subsampled) bank of normal patch tokens."""
    rng = np.random.default_rng(seed)
    bank = ref_p.reshape(-1, ref_p.shape[-1])
    if bank.shape[0] > bank_subsample:
        bank = bank[rng.choice(bank.shape[0], bank_subsample, replace=False)]
    mu = bank.mean(0, keepdims=True)
    sd = bank.std(0, keepdims=True) + 1e-8
    dev = _device()
    bt = torch.tensor((bank - mu) / sd, device=dev).float()

    def score(P):
        out = []
        for img in P:
            q = torch.tensor((img - mu) / sd, device=dev).float()
            d = torch.cdist(q, bt).topk(min(k, len(bt)), largest=False).values.mean(1)
            out.append(float(d.max()))
        return np.array(out)

    return score(tn_p), score(ta_p)


def zfuse(*streams):
    """Fuse several score streams by standardising each on its own normal-test
    part then averaging. Each stream is a tuple (normal_scores, abnormal_scores)."""
    zn, za = [], []
    for sn, sa in streams:
        m, s = sn.mean(), sn.std() + 1e-8
        zn.append((sn - m) / s)
        za.append((sa - m) / s)
    return np.mean(zn, 0), np.mean(za, 0)
