# Data acquisition

This code does not redistribute imaging data or model weights. Obtain the
datasets from their original providers under their data-use agreements, then
point the config files at the local copies.

## Volumes

The pipeline reads volume files in HDF5 format. Each file should contain either
a `reconstruction_rss` array of shape `[slices, H, W]`, or a `kspace` array
from which a root-sum-of-squares image is computed. Place the files anywhere and
either list their full paths in the split file or pass extra search directories
with `--search-dirs`.

Public sources of brain and knee MRI in this format, and tumour-segmentation
volumes for an external check, are available from the respective dataset
websites (for example large-scale raw MRI reconstruction challenges and public
medical-segmentation collections). Follow each provider's registration and
license terms.

## Split file (`split_csv`)

One row per volume:

| column | meaning |
|--------|---------|
| file   | volume base name (without extension) |
| seq    | acquisition string (the config matches `_<seq>_`) |
| path   | path to the HDF5 file |
| status | `normal` or `abnormal` |

## Annotation file (`annotation_csv`)

One row per bounding box on an abnormal slice:

| column | meaning |
|--------|---------|
| file   | volume base name |
| slice  | slice index |
| x, y   | top-left corner of the box |
| width, height | box size |

A slice is treated as abnormal if it carries at least one box. Slices of normal
volumes without boxes form the normal pool.

## Encoder weights

The two encoders are downloaded automatically from their public model hubs the
first time `scripts/01_extract_features.py` runs. No weights are stored in this
repository.

## Reproducing the feature banks

Running stage 1 writes `out/cache_<name>.npz`. This file is the reusable
artifact (the normal-reference memory bank plus the test features and their
volume/slice identifiers). Stages 2-5 read it; stages 6-7 read the resulting
JSON files. Delete `out/` to rebuild everything from scratch.
