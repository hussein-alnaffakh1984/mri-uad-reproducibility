"""Stage 4: ablations on a single cohort.

  * k sensitivity: image-level AUROC across several neighbour counts.
  * scoring rule: kNN vs LOF vs their z-fusion, with a patient-level paired
    bootstrap p-value for kNN vs LOF.
  * backbone ablation: each single backbone vs the z-fused ensemble, with a
    patient-level paired bootstrap p-value for backbone vs ensemble.

Usage:
    python scripts/04_ablations.py --cache out/cache_example.npz --out out/
"""
import argparse
import json
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detectors import image_scores, knn_distance, lof_score, zfuse
from src.stats import cluster_ci, paired_cluster


def std_stream(ref, tn, ta, scorer, k):
    """Standardise a score stream on the reference for fair z-fusion."""
    mu, sd = ref.mean(0, keepdims=True), ref.std(0, keepdims=True) + 1e-8
    r, n, a = (ref - mu) / sd, (tn - mu) / sd, (ta - mu) / sd
    sr = scorer(r, r, k=k)
    s = np.r_[scorer(r, n, k=k), scorer(r, a, k=k)]
    return (s - sr.mean()) / (sr.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 3, 5, 10, 20, 50])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.cache, allow_pickle=True)
    k = int(d["k"])
    refc, tnc, tac = d["ref_vit"], d["tn_vit"], d["ta_vit"]
    refw, tnw, taw = d["ref_cnn"], d["tn_cnn"], d["ta_cnn"]
    labels = np.r_[np.zeros(len(tnc)), np.ones(len(tac))]
    groups = np.r_[d["tn_vol"].astype(str), d["ta_vol"].astype(str)]

    # k sensitivity (image-level ViT, kNN)
    ksens = {}
    for kk in args.ks:
        sn, sa = image_scores(refc, tnc, tac, scorer=knn_distance, k=kk)
        ksens[kk] = float(roc_auc_score(labels, np.r_[sn, sa]))

    # scoring rule kNN vs LOF vs fusion
    knn_n, knn_a = image_scores(refc, tnc, tac, scorer=knn_distance, k=k)
    lof_n, lof_a = image_scores(refc, tnc, tac, scorer=lof_score, k=k)
    fus_n, fus_a = zfuse((knn_n, knn_a), (lof_n, lof_a))
    zk = std_stream(refc, tnc, tac, knn_distance, k)
    zl = std_stream(refc, tnc, tac, lof_score, k)
    rule = {
        "knn": {"auroc": float(roc_auc_score(labels, np.r_[knn_n, knn_a]))},
        "lof": {"auroc": float(roc_auc_score(labels, np.r_[lof_n, lof_a]))},
        "knn_lof": {"auroc": float(roc_auc_score(labels, np.r_[fus_n, fus_a]))},
        "knn_vs_lof_paired": paired_cluster(zk, zl, labels, groups),
    }

    # backbone ablation
    zv = std_stream(refc, tnc, tac, knn_distance, k)
    zc = std_stream(refw, tnw, taw, knn_distance, k)
    ens = (zv + zc) / 2
    abl = {
        "vit_only": float(roc_auc_score(labels, zv)),
        "cnn_only": float(roc_auc_score(labels, zc)),
        "ensemble": float(roc_auc_score(labels, ens)),
        "cnn_vs_ensemble_paired": paired_cluster(zc, ens, labels, groups),
    }

    res = {"k_sensitivity": ksens, "scoring_rule": rule, "backbone_ablation": abl}
    name = os.path.splitext(os.path.basename(args.cache))[0].replace("cache_", "")
    out = os.path.join(args.out, f"ablations_{name}.json")
    json.dump(res, open(out, "w"), indent=2)
    print("k sensitivity:", {kk: round(v, 3) for kk, v in ksens.items()})
    print("kNN vs LOF paired p:", round(rule["knn_vs_lof_paired"]["p"], 3))
    print("CNN vs ensemble paired p:", round(abl["cnn_vs_ensemble_paired"]["p"], 3))
    print("saved ->", out)


if __name__ == "__main__":
    main()
