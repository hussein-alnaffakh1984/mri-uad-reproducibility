"""Stage 1: extract frozen features and write the reusable feature cache.

The cache (.npz) is the reproducible artifact ("memory bank" / weights of the
training-free detector). It stores, for the reference, normal-test, and
abnormal-test sets:
    *_vit   image-level ViT descriptors
    *_cnn   image-level CNN descriptors
    *_vitp  dense ViT patch tokens
    *_vol   source volume id per slice
    *_slice source slice index per slice

Usage:
    python scripts/01_extract_features.py --config configs/example_cohort.yaml [--out out/]
"""
import argparse
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_cohort
from src.backbones import Backbones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--search-dirs", nargs="*", default=[],
                    help="extra directories to look for volume files by basename")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))

    print("building cohort:", cfg["name"], "sequence", cfg["sequence"])
    coh = build_cohort(cfg, search_dirs=args.search_dirs)
    print(f"  ref={len(coh['ref_imgs'])}  test_normal={len(coh['tn_imgs'])}  "
          f"test_abnormal={len(coh['ta_imgs'])}")

    bk = Backbones()
    E_ref = bk.embed(coh["ref_imgs"])
    E_tn = bk.embed(coh["tn_imgs"])
    E_ta = bk.embed(coh["ta_imgs"])

    def vols(meta): return np.array([m[0] for m in meta])
    def slis(meta): return np.array([m[1] for m in meta])

    out_path = os.path.join(args.out, f"cache_{cfg['name']}.npz")
    np.savez_compressed(
        out_path,
        ref_vit=E_ref["vit_cls"], ref_cnn=E_ref["cnn_img"], ref_vitp=E_ref["vit_patch"],
        tn_vit=E_tn["vit_cls"], tn_cnn=E_tn["cnn_img"], tn_vitp=E_tn["vit_patch"],
        ta_vit=E_ta["vit_cls"], ta_cnn=E_ta["cnn_img"], ta_vitp=E_ta["vit_patch"],
        ref_vol=vols(coh["ref_meta"]), ref_slice=slis(coh["ref_meta"]),
        tn_vol=vols(coh["tn_meta"]), tn_slice=slis(coh["tn_meta"]),
        ta_vol=vols(coh["ta_meta"]), ta_slice=slis(coh["ta_meta"]),
        k=cfg.get("k", 5), bank_subsample=cfg.get("bank_subsample", 20000),
        seed=cfg.get("seed", 0),
    )
    print("saved cache ->", out_path)


if __name__ == "__main__":
    main()
