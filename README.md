# Training-Free Anomaly Detection for Volumetric MRI

Reproducibility code for a label-free (unsupervised) anomaly-detection pipeline
that uses **frozen, pre-trained vision encoders** and a nearest-neighbour memory
bank. The pipeline scores a query slice by its distance to a bank of normal
reference features, at two representation levels (whole-image descriptors and
dense patch tokens). It performs **no training** of the encoders; the only
fitted objects are simple statistics of the normal reference set.

This repository contains the feature extraction, scoring, statistical
evaluation, and plotting code needed to regenerate every numerical artifact
(feature banks, metrics tables, and figures) from raw MRI volumes plus
bounding-box annotations.

## What this code does

1. Builds a cohort from a per-volume split file and a slice-level annotation
   file (normal vs. abnormal slices).
2. Extracts frozen features with two backbones (a Wide Residual Network and a
   self-supervised Vision Transformer) at image level and patch level, and
   caches them together with the volume and slice identifier of every slice.
3. Scores slices with k-nearest-neighbour distance, Local Outlier Factor,
   distance-to-centroid, and a patch memory bank, and fuses backbone scores by
   standardised z-score.
4. Evaluates with AUROC, sensitivity/specificity at a fixed normality
   threshold, and precision-at-k, with **patient/volume-level cluster
   bootstrap** confidence intervals and paired tests, plus an equivalence
   (TOST) test.
5. Regenerates all figures and tables from the saved result files.

## Quick start

```bash
pip install -r requirements.txt

# 1. extract frozen features + memory banks (the reusable artifacts)
python scripts/01_extract_features.py --config configs/example_cohort.yaml

# 2. detection metrics with cluster-bootstrap CIs
python scripts/02_run_detection.py --cache out/cache_example.npz --out out/

# 3. image-level vs patch-level test (5-fold CV + DeLong + cluster paired + TOST)
python scripts/03_level_test.py --cache out/cache_example.npz --out out/

# 4. ablations (k sensitivity, kNN vs LOF, backbone ablation)
python scripts/04_ablations.py --cache out/cache_example.npz --out out/

# 5. bounding-box focality measure
python scripts/05_focality_measure.py --config configs/example_cohort.yaml --out out/

# 6/7. regenerate figures and tables from the result JSONs
python scripts/06_make_figures.py --results out/ --out figures/
python scripts/07_make_tables.py  --results out/ --out tables/
```

## Data

The code expects standard public MRI data; it does **not** redistribute any
imaging data or model weights. See [docs/DATA.md](docs/DATA.md) for how to
obtain the datasets and where to place the split and annotation files. The
encoder weights are downloaded automatically from their public model hubs at
first run.

## Reproducibility notes

- All randomness is controlled by a single `--seed` (default 0). Cohort
  construction, bank subsampling, and bootstrap resampling are all seeded.
- Splits are constructed at the **volume (patient) level**: every slice of a
  given acquisition is assigned to exactly one of the reference or test sets.
- Confidence intervals and paired tests resample **whole volumes**, not
  individual slices, so they respect within-patient correlation.

## Layout

```
src/        feature extraction, detectors, statistics
scripts/    end-to-end stages (numbered)
configs/    cohort definitions (YAML)
docs/       data acquisition notes
```

## License

MIT. See [LICENSE](LICENSE).
