from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .preprocess import build_model_input
from .tiff_io import read_multilayer_tiff


LABEL_TO_INDEX = {"low": 0, "high": 1}
INDEX_TO_LABEL = {0: "low", 1: "high"}


class SecretorTiffDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        split: str | None = None,
        image_size: int = 224,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        df = pd.read_csv(self.manifest_csv)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        if df.empty:
            raise ValueError(f"No rows found for split={split!r} in {manifest_csv}")
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[index]
        path = Path(row["path"])
        stack = read_multilayer_tiff(path)
        image, _ = build_model_input(stack)

        x = torch.from_numpy(image)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        label_name = str(row["label"]).lower()
        if label_name not in LABEL_TO_INDEX:
            raise ValueError(f"Unknown label {label_name!r} in {path}")
        y = torch.tensor(LABEL_TO_INDEX[label_name], dtype=torch.long)

        return {"image": x, "label": y, "path": str(path)}

