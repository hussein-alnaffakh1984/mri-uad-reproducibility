"""Stage 6: regenerate figures from the result JSON files written by stages 2-5.

Each figure is produced only if the matching result file is present, so this
runs after whichever stages you have executed. No numbers are hard-coded; all
values are read from out/*.json.

Usage:
    python scripts/06_make_figures.py --results out/ --out figures/
"""
import argparse
import glob
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(results, prefix):
    files = sorted(glob.glob(os.path.join(results, f"{prefix}_*.json")))
    return {os.path.basename(f)[len(prefix) + 1:-5]: json.load(open(f)) for f in files}


def fig_level_test(results, out):
    data = _load(results, "leveltest")
    if not data:
        return
    names = list(data)
    img = [data[n]["image_cv"] for n in names]
    pat = [data[n]["patch_cv"] for n in names]
    x = range(len(names))
    plt.figure(figsize=(1.6 * len(names) + 2, 4))
    plt.plot(x, img, "o-", label="image level")
    plt.plot(x, pat, "s-", label="patch level")
    plt.xticks(list(x), names, rotation=20, ha="right")
    plt.ylabel("AUROC (5-fold CV)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_level_test.png"), dpi=300)
    plt.close()


def fig_k_sensitivity(results, out):
    data = _load(results, "ablations")
    if not data:
        return
    plt.figure(figsize=(6, 4))
    for n, v in data.items():
        ks = sorted(v["k_sensitivity"], key=lambda z: int(z))
        plt.plot([int(z) for z in ks], [v["k_sensitivity"][z] for z in ks], "o-", label=n)
    plt.xlabel("k (neighbours)"); plt.ylabel("AUROC")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_k_sensitivity.png"), dpi=300)
    plt.close()


def fig_focality(results, out):
    path = os.path.join(results, "focality.json")
    if not os.path.exists(path):
        return
    f = json.load(open(path))
    seqs = list(f)
    plt.figure(figsize=(6, 4))
    plt.scatter([f[s]["mean_area"] for s in seqs], [f[s]["mean_focality"] for s in seqs])
    for s in seqs:
        plt.annotate(s, (f[s]["mean_area"], f[s]["mean_focality"]))
    plt.xlabel("mean lesion area fraction"); plt.ylabel("mean focality")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_focality.png"), dpi=300)
    plt.close()


def fig_detection(results, out):
    data = _load(results, "detection")
    if not data:
        return
    for name, v in data.items():
        methods = list(v)
        aur = [v[m]["auroc"] for m in methods]
        lo = [v[m]["auroc"] - v[m]["cluster_ci"][0] for m in methods]
        hi = [v[m]["cluster_ci"][1] - v[m]["auroc"] for m in methods]
        plt.figure(figsize=(1.4 * len(methods) + 2, 4))
        plt.errorbar(range(len(methods)), aur, yerr=[lo, hi], fmt="o", capsize=4)
        plt.xticks(range(len(methods)), methods, rotation=20, ha="right")
        plt.ylabel("AUROC (cluster 95% CI)")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"fig_detection_{name}.png"), dpi=300)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="out")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_level_test(args.results, args.out)
    fig_k_sensitivity(args.results, args.out)
    fig_focality(args.results, args.out)
    fig_detection(args.results, args.out)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
