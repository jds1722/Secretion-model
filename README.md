# Cell Secretor Classifier

This project prepares multi-layer nanowell TIFF images for a binary classifier:

- `high secretor`
- `low secretor`

The model input is restricted to morphology channels only:

- Layer 0: brightfield (BF)
- Layer 1: nuclear fluorescence

Bead fluorescence channels are used only to find and remove bead regions from
the model input:

- Layer 5: INS capture beads
- Layer 7: GCG capture beads

Layers 2, 3, 4, and 6 are ignored.

## Recommended Data Layout

Put manually curated TIFF images into this structure:

```text
data/
  raw/
    high/
      sample_001.tif
      sample_002.tif
    low/
      sample_101.tif
      sample_102.tif
```

Then create a manifest:

```bash
python scripts/make_manifest.py \
  --data-root data/raw \
  --out data/manifest.csv
```

Preview preprocessing without training:

```bash
python scripts/preview_preprocess.py \
  --manifest data/manifest.csv \
  --out-dir previews \
  --max-images 50
```

Train later on the workstation/GPU machine:

```bash
python train.py \
  --manifest data/manifest.csv \
  --output-dir runs/resnet18_bf_nucleus
```

## Important Leakage Rule

Do not pass bead channels to the classifier. Layers 5 and 7 should only be used
to create bead masks and quality-control overlays.

If a bead overlaps the nucleus or covers a large part of the cell, exclude that
image during manual curation or keep it in a separate QC folder.

