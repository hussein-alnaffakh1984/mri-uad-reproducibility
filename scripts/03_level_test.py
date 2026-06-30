"""Stage 3: image-level versus patch-level representation test.

Reports, for one cohort:
  * 5-fold cross-validated AUROC at each level,
  * a DeLong p-value computed once on the pooled out-of-fold scores,
  * a paired cluster bootstrap of the (patch - image) AUROC difference
    (volume-level) with a two-sided p-value, and
  * a TOST equivalence verdict at a configurable margin.

Usage:
    python scripts/03_level_test.py --cache out/cache_example.npz --out out/ [--margin 0.03]
"""
import argparse
import json
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detectors import knn_distance, image_scores, patch_scores
from src.stats import delong, paired_cluster, tost


def cv_auc_image(norm_c, ta_c, k, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(norm_c)); rng.shuffle(idx)
    aucs = []
    for f in np.array_split(idx, 5):
        bank = np.setdiff1d(idx, f)
        s = np.r_[knn_distance(norm_c[bank], norm_c[f], k=k),
                  knn_distance(norm_c[bank], ta_c, k=k)]
        y = np.r_[np.zeros(len(f)), np.ones(len(ta_c))]
        aucs.append(roc_auc_score(y, s))
    return float(np.mean(aucs))


def cv_auc_patch(norm_p, ta_p, k, sub, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(norm_p)); rng.shuffle(idx)
    aucs = []
    for f in np.array_split(idx, 5):
        bank = np.setdiff1d(idx, f)
        sN, sA = patch_scores(norm_p[bank], norm_p[f], ta_p, k=k, bank_subsample=sub, seed=seed)
        y = np.r_[np.zeros(len(f)), np.ones(len(ta_p))]
        aucs.append(roc_auc_score(y, np.r_[sN, sA]))
    return float(np.mean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--margin", type=float, default=0.03)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.cache, allow_pickle=True)
    k = int(d["k"]); sub = int(d["bank_subsample"]); seed = int(d["seed"])

    refc, tnc, tac = d["ref_vit"], d["tn_vit"], d["ta_vit"]
    refp, tnp, tap = d["ref_vitp"], d["tn_vitp"], d["ta_vitp"]
    norm_c = np.concatenate([refc, tnc]); norm_p = np.concatenate([refp, tnp])

    img_cv = cv_auc_image(norm_c, tac, k, seed)
    pat_cv = cv_auc_patch(norm_p, tap, k, sub, seed)

    sN_i, sA_i = image_scores(refc, tnc, tac, scorer=knn_distance, k=k)
    sN_p, sA_p = patch_scores(refp, tnp, tap, k=k, bank_subsample=sub, seed=seed)
    labels = np.r_[np.zeros(len(tnc)), np.ones(len(tac))]
    a_img, a_pat, p_delong = delong(labels, np.r_[sN_i, sA_i], np.r_[sN_p, sA_p])

    groups = np.r_[d["tn_vol"].astype(str), d["ta_vol"].astype(str)]
    pc = paired_cluster(np.r_[sN_i, sA_i], np.r_[sN_p, sA_p], labels, groups, seed=seed)
    equivalent = tost(pc["ci90"], margin=args.margin)

    res = dict(image_cv=img_cv, patch_cv=pat_cv,
               delong_p_slicelevel=p_delong,
               paired_delta_patch_minus_image=pc["delta"], paired_p=pc["p"],
               paired_ci95=pc["ci95"], paired_ci90=pc["ci90"],
               n_groups=pc["n_groups"], tost_margin=args.margin, equivalent=equivalent)
    name = os.path.splitext(os.path.basename(args.cache))[0].replace("cache_", "")
    out = os.path.join(args.out, f"leveltest_{name}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"image CV={img_cv:.3f}  patch CV={pat_cv:.3f}")
    print(f"DeLong p (slice-level, optimistic) = {p_delong:.2e}")
    print(f"paired patch-image delta={pc['delta']:+.3f}  patient-level p={pc['p']:.3f}  "
          f"95%CI[{pc['ci95'][0]:+.3f},{pc['ci95'][1]:+.3f}]  equivalent@{args.margin}={equivalent}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
