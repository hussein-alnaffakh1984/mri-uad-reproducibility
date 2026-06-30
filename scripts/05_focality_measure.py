"""Stage 5: bounding-box focality measure.

For each abnormal slice with annotations, compute a scalar focality score from
the bounding boxes that is high when the abnormality is small, single, and
spatially compact, and low when it is large or scattered:

    focality = (1 - area_fraction) * (1 / n_boxes) * (1 - spread)

where area_fraction is the union box area over the image area, n_boxes is the
number of boxes on the slice, and spread is the normalised dispersion of box
centres. The per-sequence mean is reported alongside the mean area, so the two
can be compared.

Usage:
    python scripts/05_focality_measure.py --config configs/example_cohort.yaml --out out/
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import seq_of


def slice_focality(boxes, W, H):
    """boxes: list of (x, y, w, h). Returns (focality, area_fraction, n_boxes, spread)."""
    if not boxes:
        return None
    areas = [(w * h) / (W * H) for (_, _, w, h) in boxes]
    area_fraction = min(1.0, float(np.sum(areas)))
    n = len(boxes)
    centres = np.array([[x + w / 2.0, y + h / 2.0] for (x, y, w, h) in boxes], dtype=float)
    if n > 1:
        c = centres.mean(0)
        spread = float(np.mean(np.linalg.norm(centres - c, axis=1)) / np.hypot(W, H))
    else:
        spread = 0.0
    focality = (1 - area_fraction) * (1.0 / n) * (1 - spread)
    return float(focality), float(area_fraction), int(n), float(spread)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))

    ann = pd.read_csv(cfg["annotation_csv"])
    ann.columns = [c.lower().strip() for c in ann.columns]
    ann["base"] = ann["file"].astype(str).str.replace(".h5", "", regex=False)
    ann["slice_i"] = pd.to_numeric(ann["slice"], errors="coerce")
    for c in ["x", "y", "width", "height"]:
        ann[c] = pd.to_numeric(ann.get(c), errors="coerce")
    ann = ann.dropna(subset=["slice_i", "width", "height"])
    ann["seq"] = ann["base"].map(seq_of)

    W = H = args.img_size
    rows = []
    for (base, z), g in ann.groupby(["base", "slice_i"]):
        boxes = list(zip(g["x"].fillna(0), g["y"].fillna(0), g["width"], g["height"]))
        f = slice_focality(boxes, W, H)
        if f:
            rows.append({"seq": seq_of(base), "focality": f[0], "area": f[1],
                         "n_boxes": f[2], "spread": f[3]})
    df = pd.DataFrame(rows)
    summary = {}
    for seq, gg in df.groupby("seq"):
        summary[seq] = {"mean_focality": round(float(gg["focality"].mean()), 3),
                        "mean_area": round(float(gg["area"].mean()), 3),
                        "mean_nbox": round(float(gg["n_boxes"].mean()), 3),
                        "mean_spread": round(float(gg["spread"].mean()), 3),
                        "n": int(len(gg))}
    out = os.path.join(args.out, "focality.json")
    json.dump(summary, open(out, "w"), indent=2)
    df.to_csv(os.path.join(args.out, "focality_per_slice.csv"), index=False)
    print(json.dumps(summary, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
