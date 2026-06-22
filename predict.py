from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from src.dataset import INDEX_TO_LABEL
from src.model import create_resnet18_binary
from src.preprocess import build_model_input
from src.tiff_io import read_multilayer_tiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="A TIFF file or folder of TIFF files.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def iter_tiffs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(list(path.glob("*.tif")) + list(path.glob("*.tiff")))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_resnet18_binary(in_channels=2, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = []
    for path in iter_tiffs(Path(args.input)):
        stack = read_multilayer_tiff(path)
        image, _ = build_model_input(stack)
        x = torch.from_numpy(image).unsqueeze(0)
        x = F.interpolate(
            x,
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        ).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())

        rows.append(
            {
                "path": str(path),
                "predicted_label": INDEX_TO_LABEL[pred_idx],
                "prob_low": float(probs[0].item()),
                "prob_high": float(probs[1].item()),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote predictions to {out}")


if __name__ == "__main__":
    main()

