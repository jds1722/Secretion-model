from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Folder with high/ and low/ TIFFs.")
    parser.add_argument("--out", required=True, help="Output manifest CSV path.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    rows = []

    for label in ("high", "low"):
        for path in sorted((data_root / label).glob("*.tif*")):
            rows.append({"path": str(path.resolve()), "label": label})

    if not rows:
        raise SystemExit(f"No TIFF files found under {data_root}/high and {data_root}/low")

    df = pd.DataFrame(rows)
    train_val, test = train_test_split(
        df,
        test_size=args.test_fraction,
        stratify=df["label"],
        random_state=args.seed,
    )
    val_size = args.val_fraction / (1.0 - args.test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val["label"],
        random_state=args.seed,
    )

    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")
    out_df = pd.concat([train, val, test], ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} rows to {out}")
    print(out_df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()

