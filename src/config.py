from dataclasses import dataclass


@dataclass(frozen=True)
class LayerConfig:
    bf: int = 0
    nucleus: int = 1
    bead_ins: int = 5
    bead_gcg: int = 7


@dataclass(frozen=True)
class PreprocessConfig:
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5
    bead_threshold_percentile: float = 99.0
    bead_dilation_px: int = 3
    fill_mode: str = "median"


LAYERS = LayerConfig()
PREPROCESS = PreprocessConfig()

