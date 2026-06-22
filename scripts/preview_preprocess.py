from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.preprocess import build_model_input
from src.tiff_io import read_multilayer_tiff, robust_normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-images", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest).head(args.max_images)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        path = Path(row["path"])
        stack = read_multilayer_tiff(path)
        model_input, masks = build_model_input(stack)

        bf_raw = robust_normalize(stack[0])
        nucleus_raw = robust_normalize(stack[1])
        bead_mask = masks["bead"]

        fig, axes = plt.subplots(1, 5, figsize=(14, 3))
        axes[0].imshow(bf_raw, cmap="gray")
        axes[0].set_title("BF raw")
        axes[1].imshow(nucleus_raw, cmap="gray")
        axes[1].set_title("Nucleus raw")
        axes[2].imshow(bead_mask, cmap="magma")
        axes[2].set_title("Bead mask")
        axes[3].imshow(model_input[0], cmap="gray")
        axes[3].set_title("BF masked")
        axes[4].imshow(model_input[1], cmap="gray")
        axes[4].set_title("Nucleus masked")

        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{row['label']} | {path.name}", fontsize=10)
        fig.tight_layout()

        out_path = out_dir / f"{path.stem}_preview.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    print(f"Wrote previews to {out_dir}")


if __name__ == "__main__":
    main()

