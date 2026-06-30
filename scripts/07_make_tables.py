"""Stage 7: regenerate tables (CSV) from the result JSON files.

Produces tidy CSVs that mirror the manuscript tables. All values come from
out/*.json; nothing is hard-coded.

Usage:
    python scripts/07_make_tables.py --results out/ --out tables/
"""
import argparse
import glob
import json
import os
import csv


def _load(results, prefix):
    files = sorted(glob.glob(os.path.join(results, f"{prefix}_*.json")))
    return {os.path.basename(f)[len(prefix) + 1:-5]: json.load(open(f)) for f in files}


def table_detection(results, out):
    data = _load(results, "detection")
    if not data:
        return
    with open(os.path.join(out, "table_detection.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "method", "AUROC", "cluster_lo", "cluster_hi",
                    "sensitivity", "specificity", "P@5", "abnormal_base_rate", "n_groups"])
        for cohort, methods in data.items():
            for m, v in methods.items():
                w.writerow([cohort, m, f"{v['auroc']:.3f}",
                            f"{v['cluster_ci'][0]:.3f}", f"{v['cluster_ci'][1]:.3f}",
                            f"{v['sensitivity']:.3f}", f"{v['specificity']:.3f}",
                            f"{v['p_at_5']:.3f}", f"{v['abnormal_base_rate']:.3f}", v["n_groups"]])


def table_level_test(results, out):
    data = _load(results, "leveltest")
    if not data:
        return
    with open(os.path.join(out, "table_level_test.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "image_cv", "patch_cv", "delong_p_slicelevel",
                    "paired_delta", "paired_p", "ci95_lo", "ci95_hi",
                    "equivalent", "n_groups"])
        for c, v in data.items():
            w.writerow([c, f"{v['image_cv']:.3f}", f"{v['patch_cv']:.3f}",
                        f"{v['delong_p_slicelevel']:.2e}",
                        f"{v['paired_delta_patch_minus_image']:+.3f}", f"{v['paired_p']:.3f}",
                        f"{v['paired_ci95'][0]:+.3f}", f"{v['paired_ci95'][1]:+.3f}",
                        v["equivalent"], v["n_groups"]])


def table_ablations(results, out):
    data = _load(results, "ablations")
    if not data:
        return
    with open(os.path.join(out, "table_k_sensitivity.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        ks = sorted({k for v in data.values() for k in v["k_sensitivity"]}, key=lambda z: int(z))
        w.writerow(["cohort"] + [f"k={k}" for k in ks])
        for c, v in data.items():
            w.writerow([c] + [f"{v['k_sensitivity'].get(k, ''):.3f}" if k in v["k_sensitivity"] else ""
                              for k in ks])
    with open(os.path.join(out, "table_scoring_rule.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "kNN", "LOF", "kNN+LOF", "kNN_vs_LOF_p"])
        for c, v in data.items():
            r = v["scoring_rule"]
            w.writerow([c, f"{r['knn']['auroc']:.3f}", f"{r['lof']['auroc']:.3f}",
                        f"{r['knn_lof']['auroc']:.3f}", f"{r['knn_vs_lof_paired']['p']:.3f}"])
    with open(os.path.join(out, "table_backbone_ablation.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "ViT_only", "CNN_only", "ensemble", "CNN_vs_ensemble_p"])
        for c, v in data.items():
            a = v["backbone_ablation"]
            w.writerow([c, f"{a['vit_only']:.3f}", f"{a['cnn_only']:.3f}",
                        f"{a['ensemble']:.3f}", f"{a['cnn_vs_ensemble_paired']['p']:.3f}"])


def table_focality(results, out):
    path = os.path.join(results, "focality.json")
    if not os.path.exists(path):
        return
    f = json.load(open(path))
    with open(os.path.join(out, "table_focality.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence", "mean_focality", "mean_area", "mean_nbox", "mean_spread", "n"])
        for s, v in f.items():
            w.writerow([s, v["mean_focality"], v["mean_area"], v["mean_nbox"], v["mean_spread"], v["n"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="out")
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    table_detection(args.results, args.out)
    table_level_test(args.results, args.out)
    table_ablations(args.results, args.out)
    table_focality(args.results, args.out)
    print("tables written to", args.out)


if __name__ == "__main__":
    main()
