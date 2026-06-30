"""Evaluation statistics with patient/volume-level clustering.

Provides AUROC-based metrics, naive (slice-level) and cluster (volume-level)
bootstrap confidence intervals, a paired cluster bootstrap for the difference
of two scorers, a two one-sided test (TOST) for equivalence, and a DeLong test
for two correlated ROC curves.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------- point metrics
def point_metrics(s_normal, s_abnormal, threshold_pct=95):
    y = np.r_[np.zeros(len(s_normal)), np.ones(len(s_abnormal))]
    s = np.r_[s_normal, s_abnormal]
    auc = roc_auc_score(y, s)
    thr = np.percentile(s_normal, threshold_pct)        # fixed operating point
    pred = (s >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    sens = tp / (tp + fn + 1e-9); spec = tn / (tn + fp + 1e-9)
    order = np.argsort(-s); p_at_5 = y[order][:5].mean()
    base_rate = y.mean()
    return dict(auroc=float(auc), sensitivity=float(sens), specificity=float(spec),
                p_at_5=float(p_at_5), abnormal_base_rate=float(base_rate),
                n_normal=int(len(s_normal)), n_abnormal=int(len(s_abnormal)))


# ----------------------------------------------------------------- bootstraps
def naive_ci(scores, labels, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(B):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) > 1:
            a.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_ci(scores, labels, groups, B=2000, seed=0):
    """Bootstrap by resampling whole groups (volumes/patients)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    a = []
    for _ in range(B):
        samp = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        y = labels[idx]
        if len(np.unique(y)) > 1:
            a.append(roc_auc_score(y, scores[idx]))
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(roc_auc_score(labels, scores)), float(lo), float(hi), int(len(uniq))


def paired_cluster(score_a, score_b, labels, groups, B=2000, seed=0):
    """Paired cluster bootstrap of AUROC(b) - AUROC(a) with a two-sided p-value
    and 95% / 90% intervals (the 90% interval is used for TOST)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    base = roc_auc_score(labels, score_b) - roc_auc_score(labels, score_a)
    diffs = []
    for _ in range(B):
        samp = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        diffs.append(roc_auc_score(y, score_b[idx]) - roc_auc_score(y, score_a[idx]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    ci95 = np.percentile(diffs, [2.5, 97.5])
    ci90 = np.percentile(diffs, [5, 95])
    return dict(delta=float(base), p=float(max(p, 1.0 / B)),
                ci95=[float(ci95[0]), float(ci95[1])],
                ci90=[float(ci90[0]), float(ci90[1])], n_groups=int(len(uniq)))


def tost(ci90, margin=0.03):
    """Equivalence holds when the 90% interval of the difference lies inside
    (-margin, +margin)."""
    lo, hi = ci90
    return bool(lo > -margin and hi < margin)


# -------------------------------------------------------------------- DeLong
def delong(labels, scores_a, scores_b):
    """DeLong test for two correlated ROC curves on pooled scores.
    Returns (auc_a, auc_b, p_value)."""
    y = np.asarray(labels)

    def midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N); T2[J] = T
        return T2

    sc = np.stack([scores_a, scores_b])
    m = int(y.sum()); n = len(y) - m; k = sc.shape[0]
    tx = np.array([midrank(p) for p in sc[:, y == 1]])
    ty = np.array([midrank(q) for q in sc[:, y == 0]])
    tz = np.array([midrank(s) for s in sc])
    auc = (tz[:, y == 1].sum(1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, y == 1] - tx) / n
    v10 = 1 - (tz[:, y == 0] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    S = np.array([[S]]) if k == 1 else S
    l = np.array([1, -1])
    z = (auc[0] - auc[1]) / np.sqrt(l @ S @ l + 1e-12)
    return float(auc[0]), float(auc[1]), float(2 * stats.norm.sf(abs(z)))
