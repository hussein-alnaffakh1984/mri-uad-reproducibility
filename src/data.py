"""Cohort construction from a per-volume split and slice-level annotations.

A cohort is built deterministically (seeded) at the volume level:
  * abnormal slices are exactly the annotated slices of abnormal volumes,
    kept in (volume, slice) order;
  * normal slices are central, non-annotated slices of normal volumes,
    sampled per volume and then globally shuffled;
  * the normal pool is split into a reference bank and a normal test set.

Every kept slice carries its source volume id and slice index so that
downstream statistics can resample whole volumes.
"""
import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F


def seq_of(fname, sequences=("AXT1POST", "AXT1PRE", "AXFLAIR", "AXT2", "AXT1")):
    for s in sequences:
        if f"_{s}_" in fname:
            return s
    return "OTHER"


def load_volume(path, search_dirs=()):
    """Load a volume as [Z, H, W]. Uses a reconstruction if present, otherwise
    a root-sum-of-squares of the inverse FFT of k-space."""
    p = path.strip()
    if not os.path.exists(p):
        for d in search_dirs:
            alt = os.path.join(d, os.path.basename(p))
            if os.path.exists(alt):
                p = alt
                break
    with h5py.File(p, "r") as h:
        if "reconstruction_rss" in h:
            vol = h["reconstruction_rss"][:]
        else:
            ks = h["kspace"][:]
            img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ks, (-2, -1))), (-2, -1))
            vol = np.sqrt((np.abs(img) ** 2).sum(1))
    return vol.astype(np.float32)


def norm01(s, img_size):
    """Clip the top intensity percentile, scale to [0, 1], resize to a square."""
    s = np.asarray(s, dtype=np.float32)
    hi = np.percentile(s, 99.5)
    s = np.clip(s, 0, hi) / (hi + 1e-8)
    t = torch.from_numpy(s)[None, None]
    t = F.interpolate(t, size=img_size, mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def annotated_slices(annotation_csv):
    """Map volume base name -> set of annotated (abnormal) slice indices."""
    ann = pd.read_csv(annotation_csv)
    ann.columns = [c.lower().strip() for c in ann.columns]
    ann["base"] = ann["file"].astype(str).str.replace(".h5", "", regex=False)
    ann["slice_i"] = pd.to_numeric(ann["slice"], errors="coerce")
    boxcols = [c for c in ["x", "y", "width", "height"] if c in ann.columns]
    if boxcols:
        has_box = ann[boxcols].notna().any(axis=1)
    elif "label" in ann.columns:
        has_box = ~ann["label"].astype(str).str.lower().isin(["", "nan", "normal", "none", "no"])
    else:
        has_box = pd.Series(True, index=ann.index)
    abn = ann[has_box & ann["slice_i"].notna()].copy()
    abn["slice_i"] = abn["slice_i"].astype(int)
    return abn.groupby("base")["slice_i"].apply(lambda s: set(int(x) for x in s)).to_dict()


def build_cohort(cfg, search_dirs=()):
    """Return image lists and per-slice (volume, slice) metadata for a sequence.

    Output dict keys:
        ref_imgs, ref_meta            reference (bank) normals
        tn_imgs,  tn_meta             normal test slices
        ta_imgs,  ta_meta             abnormal test slices
    where *_meta is a list of (volume_base, slice_index).
    """
    seq = cfg["sequence"]
    img_size = cfg.get("img_size", 224)
    lo_hi = cfg.get("slice_frac", [0.30, 0.85])
    per_vol = cfg.get("normal_per_vol", 12)
    n_ref = cfg.get("n_ref", 400)
    n_test_norm = cfg.get("n_test_normal", 256)
    seed = cfg.get("seed", 0)
    rng = np.random.default_rng(seed)

    split = pd.read_csv(cfg["split_csv"])
    split.columns = [c.strip() for c in split.columns]
    split["seq"] = split["seq"].str.strip()
    split["status"] = split["status"].str.strip()
    sub = split[split["seq"] == seq].copy()
    abn_idx = annotated_slices(cfg["annotation_csv"])

    normal, abnormal = [], []   # each entry: (image, (base, z))
    for _, row in sub.iterrows():
        base = row["file"].strip()
        try:
            vol = load_volume(row["path"], search_dirs)
        except Exception as e:
            print("skip", base, e)
            continue
        n = vol.shape[0]
        aset = {z for z in abn_idx.get(base, set()) if 0 <= z < n}
        for z in sorted(aset):
            abnormal.append((norm01(vol[z], img_size), (base, int(z))))
        a, b = int(n * lo_hi[0]), int(n * lo_hi[1])
        cand = [z for z in range(a, b) if z not in aset]
        rng.shuffle(cand)
        for z in cand[:per_vol]:
            normal.append((norm01(vol[z], img_size), (base, int(z))))

    order = np.arange(len(normal))
    rng.shuffle(order)
    normal = [normal[i] for i in order]
    n_test_norm = min(n_test_norm, max(20, len(normal) // 3))
    n_ref = min(n_ref, len(normal) - n_test_norm)
    ref = normal[:n_ref]
    tn = normal[n_ref:n_ref + n_test_norm]

    def split_pair(pairs):
        return [p[0] for p in pairs], [p[1] for p in pairs]

    ref_imgs, ref_meta = split_pair(ref)
    tn_imgs, tn_meta = split_pair(tn)
    ta_imgs, ta_meta = split_pair(abnormal)
    assert len(ta_imgs) >= 8 and len(ref_imgs) >= 20, "insufficient cohort size"
    return dict(ref_imgs=ref_imgs, ref_meta=ref_meta,
                tn_imgs=tn_imgs, tn_meta=tn_meta,
                ta_imgs=ta_imgs, ta_meta=ta_meta)
