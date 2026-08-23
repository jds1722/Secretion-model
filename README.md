# Cell Secretor Classifier

This project prepares multi-layer nanowell TIFF images for binary classifiers:

- insulin `high` / `low`
- glucagon `high` / `low`

Insulin and glucagon are trained as independent models. Use the same pipeline
twice with different `--target` values.

The model input is restricted to morphology-derived channels only:

- Layer 0: brightfield (BF)
- Layer 1: nuclear fluorescence
- Channel 2: a nucleus-guided BF cell mask (derived, not a TIFF layer)

The nucleus channel identifies the primary nucleus. Its centroid seeds a BF
watershed segmentation, so distant BF structures cannot become the target cell.
The selected UI ROI size is a maximum search/crop size; the final square crop is
automatically tightened from nucleus size and the segmented-cell bounding box.
Background outside the candidate cell is smoothly dimmed instead of hard-zeroed.
Bead pixels are excluded from the nucleus-guided cell mask and replaced again in
the final BF tensor, so a bead attached to a cell cannot be treated as cell body.

Autofluorescence and bead fluorescence channels are used only to find ignored
regions in the model input:

- Layer 2: bead/nanowell autofluorescence
- Layer 5: INS capture beads
- Layer 7: GCG capture beads

Layers 3, 4, and 6 are ignored.

## Recommended Data Layout

Put manually curated TIFF images into this structure:

```text
data/
  raw/
    insulin/
      high/
        sample_001.tif
      low/
        sample_002.tif
    glucagon/
      high/
        sample_101.tif
      low/
        sample_102.tif
```

Then create a manifest:

```bash
python scripts/make_manifest.py \
  --data-root data/raw \
  --out data/manifest.csv
```

The manifest split is made by original image group, not by individual augmented
files. Augmented variants such as `_orig`, `_rot90`, `_rot180`, `_rot270`, and
`_flip_lr` stay in the same split so validation does not reuse training images.
By default, about one sixth of each high/low group is held out for validation.

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
  --target insulin \
  --output-dir runs/insulin_resnet18_bf_nucleus \
  --preview-images 50

python train.py \
  --manifest data/manifest.csv \
  --target glucagon \
  --output-dir runs/glucagon_resnet18_bf_nucleus \
  --preview-images 50
```

Training outputs include model metrics, plots, and visual QC examples:

```text
runs/insulin_resnet18_bf_nucleus/
  best_model.pt
  history.csv
  test_metrics.json
  training_curves.png
  test_confusion_matrix.png
  test_roc_curve.png
  test_precision_recall_curve.png
  validation_threshold_curve.png
  validation_thresholds.csv
  preprocess_attention_examples/
    examples.csv
    000_sample_attention.png
    001_sample_attention.png
```

Each model-focus example shows the preprocessed BF channel, preprocessed nucleus
channel, nucleus-guided cell mask, Grad-CAM model-focus map, and overlay for the
model's predicted class.

The high/low decision threshold is selected from validation predictions and
stored in `best_model.pt`. Test metrics and `predict.py` use this stored
threshold instead of always using a fixed 0.5 cutoff. Use
`--threshold-mode high_recall --min-high-recall 0.9` when missing high cells is
more costly than false positives.

## Local UI

Run the Streamlit UI from the project folder:

```bash
python -m streamlit run app.py
```

The UI provides pages for data folder checks, manifest creation, training,
evaluation plots, Grad-CAM preview images, and prediction CSV generation.

## GPU Setup

This project uses GPU automatically when PyTorch can see CUDA:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

For this workstation, CUDA 13.2 PyTorch was installed with:

```bash
python -m pip install --force-reinstall torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132
```

Check GPU availability:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Predict with one model:

```bash
python predict.py \
  --checkpoint runs/insulin_resnet18_bf_nucleus/best_model.pt \
  --input data/new_tiffs \
  --out predictions_insulin.csv
```

Predict with both independent models into one CSV:

```bash
python predict.py \
  --insulin-checkpoint runs/insulin_resnet18_bf_nucleus/best_model.pt \
  --glucagon-checkpoint runs/glucagon_resnet18_bf_nucleus/best_model.pt \
  --input data/new_tiffs \
  --out predictions_both.csv
```

## Important Leakage Rule

Do not pass bead or autofluorescence channels to the classifier. Layers 5 and 7
create bead masks that are median-filled in BF/nucleus. Layer 2 creates an
autofluorescence/nanowell mask that is zeroed out in BF/nucleus, not
median-filled. Layer 2 uses a low-level signal threshold based on the near-black
background and high-signal range, rather than only masking the brightest top
1% pixels. It is tuned to remove broad autofluorescent/nanowell regions while
preserving pixels with clear Layer 1 nucleus signal.

If a bead overlaps the nucleus or covers a large part of the cell, exclude that
image during manual curation or keep it in a separate QC folder.

