"""Stage 2: detection metrics with cluster-bootstrap confidence intervals.

Scores the cohort with each single backbone (image level) and their z-fused
ensemble, then reports AUROC / sensitivity / specificity / precision-at-k and
both naive (slice) and cluster (volume) confidence intervals.

Usage:
    python scripts/02_run_detection.py --cache out/cache_example.npz --out out/
"""
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detectors import image_scores, zfuse, knn_distance
from src.stats import point_metrics, naive_ci, cluster_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.cache, allow_pickle=True)
    k = args.k or int(d["k"])

    groups = np.r_[d["tn_vol"].astype(str), d["ta_vol"].astype(str)]
    labels = np.r_[np.zeros(len(d["tn_vit"])), np.ones(len(d["ta_vit"]))]

    res = {}
    streams = {}
    for name, refk, tnk, tak in [("vit", "ref_vit", "tn_vit", "ta_vit"),
                                 ("cnn", "ref_cnn", "tn_cnn", "ta_cnn")]:
        sn, sa = image_scores(d[refk], d[tnk], d[tak], scorer=knn_distance, k=k)
        streams[name] = (sn, sa)
        s = np.r_[sn, sa]
        m = point_metrics(sn, sa)
        m["naive_ci"] = naive_ci(s, labels)
        auc, lo, hi, npat = cluster_ci(s, labels, groups)
        m["cluster_ci"] = [lo, hi]
        m["n_groups"] = npat
        res[name] = m

    en_n, en_a = zfuse(streams["vit"], streams["cnn"])
    s = np.r_[en_n, en_a]
    m = point_metrics(en_n, en_a)
    m["naive_ci"] = naive_ci(s, labels)
    auc, lo, hi, npat = cluster_ci(s, labels, groups)
    m["cluster_ci"] = [lo, hi]
    m["n_groups"] = npat
    res["ensemble"] = m

    name = os.path.splitext(os.path.basename(args.cache))[0].replace("cache_", "")
    out = os.path.join(args.out, f"detection_{name}.json")
    json.dump(res, open(out, "w"), indent=2)
    for kk, v in res.items():
        print(f"{kk:9s} AUROC={v['auroc']:.3f} cluster-CI[{v['cluster_ci'][0]:.3f},"
              f"{v['cluster_ci'][1]:.3f}] (n={v['n_groups']} vols) "
              f"P@5={v['p_at_5']:.3f} base={v['abnormal_base_rate']:.3f}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
