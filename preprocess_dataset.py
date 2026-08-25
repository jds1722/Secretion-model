from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocess_v2 import ObjectPreprocessConfig, build_model_input_v2
from src.tiff_io import read_multilayer_tiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess TIFFs once into independent NPZ tensors."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Source TIFF manifest CSV")
    source.add_argument(
        "--input-dir",
        help="Folder to search recursively for .tif and .tiff files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--out-manifest", default=None)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--bead-min-diameter", type=float, default=5.0)
    parser.add_argument("--bead-max-diameter", type=float, default=20.0)
    parser.add_argument("--bead-dilation-px", type=int, default=2)
    parser.add_argument("--nucleus-min-diameter", type=float, default=10.0)
    parser.add_argument("--nucleus-max-diameter", type=float, default=100.0)
    parser.add_argument("--cell-opening-px", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-on-qc",
        action="store_true",
        help="Exclude samples without a valid nucleus from the trainable manifest.",
    )
    return parser.parse_args()


def cache_name(source: Path) -> str:
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{source.stem}_{digest}.npz"


def discover_tiffs(input_dir: Path) -> list[Path]:
    """Return all TIFF files below a folder in deterministic path order."""
    if not input_dir.is_dir():
        raise ValueError(f"Input folder does not exist: {input_dir}")
    return sorted(
        path.resolve()
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = (
        Path(args.out_manifest)
        if args.out_manifest
        else output_dir / "preprocessed_manifest.csv"
    )
    if args.manifest:
        source_manifest = Path(args.manifest)
        df = pd.read_csv(source_manifest)
    else:
        paths = discover_tiffs(Path(args.input_dir))
        if not paths:
            raise ValueError(f"No TIFF files found under: {args.input_dir}")
        df = pd.DataFrame({"path": [str(path) for path in paths]})
        source_manifest = output_dir / "source_manifest.csv"
        df.to_csv(source_manifest, index=False)
        print(f"Found {len(df)} TIFF files; source manifest: {source_manifest}")
    if "path" not in df.columns:
        raise ValueError("Source manifest must contain a path column")

    object_config = ObjectPreprocessConfig(
        bead_min_diameter=args.bead_min_diameter,
        bead_max_diameter=args.bead_max_diameter,
        bead_dilation_px=args.bead_dilation_px,
        nucleus_min_diameter=args.nucleus_min_diameter,
        nucleus_max_diameter=args.nucleus_max_diameter,
        cell_opening_px=args.cell_opening_px,
    )
    if object_config.bead_min_diameter > object_config.bead_max_diameter:
        raise ValueError("bead minimum diameter cannot exceed maximum diameter")
    if object_config.nucleus_min_diameter > object_config.nucleus_max_diameter:
        raise ValueError("nucleus minimum diameter cannot exceed maximum diameter")
    (output_dir / "preprocess_run_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    output_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="preprocessing"):
        result = row.to_dict()
        source = Path(row["path"])
        destination = output_dir / "tensors" / cache_name(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            qc_path = destination.with_suffix(".qc.json")
            if args.overwrite or not destination.exists() or not qc_path.exists():
                stack = read_multilayer_tiff(source)
                image, masks, qc = build_model_input_v2(
                    stack, object_config=object_config, roi_size=args.roi_size
                )
                np.savez_compressed(
                    destination,
                    image=image,
                    ignore_mask=masks["ignore"].astype(np.uint8),
                    cell_mask=masks["cell"].astype(np.uint8),
                    nucleus_mask=masks["primary_nucleus"].astype(np.uint8),
                )
                qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
            else:
                qc = json.loads(qc_path.read_text(encoding="utf-8"))
            result.update(qc)
            result["preprocessed_path"] = str(destination.resolve())
            result["preprocess_status"] = (
                "excluded_qc"
                if args.fail_on_qc and not qc["nucleus_valid"]
                else "ok"
            )
            result["preprocess_error"] = ""
        except Exception as error:
            result["preprocessed_path"] = ""
            result["preprocess_status"] = "error"
            result["preprocess_error"] = str(error)
        output_rows.append(result)

    full_df = pd.DataFrame(output_rows)
    trainable_df = full_df[full_df["preprocess_status"] == "ok"].copy()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    trainable_df.to_csv(out_manifest, index=False)
    full_df.to_csv(output_dir / "preprocess_qc_all.csv", index=False)
    print(f"Wrote {len(trainable_df)} trainable rows to {out_manifest}")
    print(f"Full QC report: {output_dir / 'preprocess_qc_all.csv'}")


if __name__ == "__main__":
    main()
