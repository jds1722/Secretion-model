from __future__ import annotations

import numpy as np
from skimage import morphology

from .config import LAYERS, PREPROCESS, LayerConfig, PreprocessConfig
from .tiff_io import robust_normalize


def _safe_layer(stack: np.ndarray, layer_index: int) -> np.ndarray:
    if layer_index >= stack.shape[0]:
        raise IndexError(
            f"TIFF has {stack.shape[0]} layers, but layer {layer_index} was requested."
        )
    return stack[layer_index]


def make_bead_mask(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> np.ndarray:
    """Detect INS/GCG bead regions from layers 5 and 7."""
    ins = _safe_layer(stack, layer_config.bead_ins).astype(np.float32)
    gcg = _safe_layer(stack, layer_config.bead_gcg).astype(np.float32)

    bead_signal = np.maximum(ins, gcg)
    threshold = np.percentile(bead_signal, preprocess_config.bead_threshold_percentile)
    mask = bead_signal > threshold

    mask = morphology.remove_small_objects(mask, min_size=4)
    if preprocess_config.bead_dilation_px > 0:
        footprint = morphology.disk(preprocess_config.bead_dilation_px)
        mask = morphology.binary_dilation(mask, footprint)
    return mask.astype(bool)


def make_nucleus_mask(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> np.ndarray:
    """Create a rough nucleus mask for QC, not for final scientific quantification."""
    nucleus = _safe_layer(stack, layer_config.nucleus).astype(np.float32)
    threshold = np.percentile(nucleus, preprocess_config.bead_threshold_percentile)
    mask = nucleus > threshold
    mask = morphology.remove_small_objects(mask, min_size=8)
    mask = morphology.binary_closing(mask, morphology.disk(1))
    return mask.astype(bool)


def bead_nucleus_overlap_fraction(stack: np.ndarray) -> float:
    """Return area(bead_mask intersect nucleus_mask) / area(nucleus_mask)."""
    bead_mask = make_bead_mask(stack)
    nucleus_mask = make_nucleus_mask(stack)
    nucleus_area = int(nucleus_mask.sum())
    if nucleus_area == 0:
        return 0.0
    return float((bead_mask & nucleus_mask).sum() / nucleus_area)


def fill_masked_regions(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str = "median",
) -> np.ndarray:
    """Replace masked bead pixels without exposing bead signal to the model."""
    image = image.astype(np.float32, copy=True)
    if not mask.any():
        return image

    valid = image[~mask]
    if valid.size == 0:
        fill_value = float(np.median(image))
    elif mode == "zero":
        fill_value = 0.0
    elif mode == "mean":
        fill_value = float(np.mean(valid))
    elif mode == "median":
        fill_value = float(np.median(valid))
    else:
        raise ValueError(f"Unknown fill mode: {mode}")

    image[mask] = fill_value
    return image


def build_model_input(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return a 2-channel morphology input: BF and nucleus, with beads removed."""
    bead_mask = make_bead_mask(stack, layer_config, preprocess_config)

    bf = _safe_layer(stack, layer_config.bf)
    nucleus = _safe_layer(stack, layer_config.nucleus)

    bf = fill_masked_regions(bf, bead_mask, preprocess_config.fill_mode)
    nucleus = fill_masked_regions(nucleus, bead_mask, preprocess_config.fill_mode)

    bf = robust_normalize(
        bf,
        preprocess_config.lower_percentile,
        preprocess_config.upper_percentile,
    )
    nucleus = robust_normalize(
        nucleus,
        preprocess_config.lower_percentile,
        preprocess_config.upper_percentile,
    )

    model_input = np.stack([bf, nucleus], axis=0).astype(np.float32)
    masks = {"bead": bead_mask}
    return model_input, masks
