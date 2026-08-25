from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import MODEL_LABELS, MODEL_NAMES, MODEL_TRAINING_DEFAULTS


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)
AUGMENTATION_SUFFIX_RE = re.compile(
    r"_(?:orig|rot90|rot180|rot270)(?:_flip_lr)?$"
)

st.set_page_config(
    page_title="Secretion NPZ Dashboard",
    page_icon="🧫",
    layout="wide",
)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _set_path_value(key: str, value: str) -> None:
    st.session_state[f"{key}_typed"] = value


def _set_browser_directory(key: str, value: str) -> None:
    st.session_state[f"{key}_browser_dir"] = value


def path_browser(
    label: str,
    key: str,
    *,
    kind: str = "folder",
    default: str = "",
    extensions: tuple[str, ...] = (),
) -> str:
    """Provide direct path entry plus a server-side Windows file browser."""
    typed_key = f"{key}_typed"
    browser_key = f"{key}_browser_dir"
    if typed_key not in st.session_state:
        st.session_state[typed_key] = default
    typed = st.text_input(label, key=typed_key)

    initial = Path(typed).expanduser() if typed else ROOT
    if kind == "file" and initial.is_file():
        initial = initial.parent
    if not initial.is_dir():
        initial = ROOT
    if browser_key not in st.session_state:
        st.session_state[browser_key] = str(initial.resolve())

    with st.expander(f"Browse: {label}"):
        current = Path(st.session_state[browser_key])
        if not current.is_dir():
            current = ROOT
            st.session_state[browser_key] = str(current)
        st.caption(f"Current folder: `{current}`")

        drives = [
            Path(f"{letter}:\\")
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if Path(f"{letter}:\\").is_dir()
        ] if os.name == "nt" else [Path("/")]
        nav1, nav2, nav3 = st.columns([2, 1, 1])
        selected_drive = nav1.selectbox(
            "Drive",
            drives,
            format_func=str,
            key=f"{key}_drive",
        )
        nav2.button(
            "Open drive",
            key=f"{key}_open_drive",
            on_click=_set_browser_directory,
            args=(key, str(selected_drive)),
        )
        nav3.button(
            "Up",
            key=f"{key}_up",
            disabled=current.parent == current,
            on_click=_set_browser_directory,
            args=(key, str(current.parent)),
        )

        try:
            children = sorted(
                [path for path in current.iterdir() if path.is_dir()],
                key=lambda path: path.name.lower(),
            )
            browse_error = ""
        except (OSError, PermissionError) as error:
            children = []
            browse_error = str(error)
        selected_folder = st.selectbox(
            "Subfolder",
            children,
            format_func=lambda path: path.name,
            index=None,
            placeholder="Select a subfolder",
            key=f"{key}_subfolder",
        )
        folder1, folder2 = st.columns(2)
        folder1.button(
            "Open selected folder",
            key=f"{key}_open_subfolder",
            disabled=selected_folder is None,
            on_click=_set_browser_directory,
            args=(key, str(selected_folder) if selected_folder else str(current)),
        )
        if kind == "folder":
            folder2.button(
                "Use current folder",
                key=f"{key}_use_folder",
                on_click=_set_path_value,
                args=(key, str(current)),
                type="primary",
            )
        else:
            normalized_extensions = {suffix.lower() for suffix in extensions}
            try:
                files = sorted(
                    [
                        path for path in current.iterdir()
                        if path.is_file()
                        and (
                            not normalized_extensions
                            or path.suffix.lower() in normalized_extensions
                        )
                    ],
                    key=lambda path: path.name.lower(),
                )
            except (OSError, PermissionError):
                files = []
            selected_file = st.selectbox(
                "File",
                files,
                format_func=lambda path: path.name,
                index=None,
                placeholder="Select a file",
                key=f"{key}_file",
            )
            folder2.button(
                "Use selected file",
                key=f"{key}_use_file",
                disabled=selected_file is None,
                on_click=_set_path_value,
                args=(key, str(selected_file) if selected_file else ""),
                type="primary",
            )
        if browse_error:
            st.warning(browse_error)
    return typed


def run_dirs() -> list[Path]:
    return sorted(
        [path for path in RUNS_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def manifest_report(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    if not path.is_file():
        return None, ["Manifest file does not exist."]
    try:
        df = pd.read_csv(path)
    except Exception as error:
        return None, [f"Cannot read manifest: {error}"]
    errors = []
    required = {"preprocessed_path", "label", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        errors.append("Missing columns: " + ", ".join(missing))
    if "split" in df:
        found = set(df["split"].astype(str).str.lower())
        missing_splits = sorted({"train", "val", "test"} - found)
        if missing_splits:
            errors.append("Missing splits: " + ", ".join(missing_splits))
    return df, errors


def start_background(command: list[str]) -> int:
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    kwargs: dict[str, object] = {
        "cwd": ROOT,
        "env": environment,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs).pid


def tail(path: Path, lines: int = 35) -> str:
    try:
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        )
    except OSError:
        return "Log has not been created yet."


def preprocessing_page() -> None:
    st.header("Pre-processing status")
    st.caption(
        "This page only monitors outputs. It never stops or restarts a running pre-processing process."
    )
    output_text = path_browser(
        "Pre-processing output folder",
        "preprocess_output",
        kind="folder",
        default=str(ROOT / "data" / "preprocessed_v2"),
    )
    output = Path(output_text).expanduser()
    if not output.is_dir():
        st.info("Enter an existing output folder to monitor it.")
        return

    source_manifest = output / "source_manifest.csv"
    tensors = output / "tensors"
    total = 0
    if source_manifest.exists():
        try:
            total = len(pd.read_csv(source_manifest, usecols=["path"]))
        except Exception:
            total = 0
    npz_files = list(tensors.glob("*.npz")) if tensors.is_dir() else []
    qc_files = list(tensors.glob("*.qc.json")) if tensors.is_dir() else []
    complete = len(npz_files)
    percent = min(complete / total, 1.0) if total else 0.0
    latest = max((path.stat().st_mtime for path in npz_files), default=0)
    latest_text = (
        datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S")
        if latest else "-"
    )
    active = bool(latest and datetime.now().timestamp() - latest < 60)

    cols = st.columns(5)
    cols[0].metric("Source TIFFs", f"{total:,}" if total else "-")
    cols[1].metric("NPZ complete", f"{complete:,}")
    cols[2].metric("QC files", f"{len(qc_files):,}")
    cols[3].metric("Progress", f"{percent:.1%}" if total else "-")
    cols[4].metric("Activity", "Running" if active else "Idle / complete")
    if total:
        st.progress(percent, text=f"{complete:,} / {total:,} images")
    st.caption(f"Latest tensor update: {latest_text}")
    if st.button("Refresh status"):
        st.rerun()

    config = load_json(output / "preprocess_run_config.json")
    if config:
        with st.expander("Pre-processing configuration"):
            st.json(config)

    final_report = output / "preprocess_qc_all.csv"
    if final_report.exists():
        qc = pd.read_csv(final_report)
        st.success("Pre-processing completed and the final QC report is available.")
        status_counts = qc.get(
            "preprocess_status", pd.Series(dtype=str)
        ).value_counts()
        status_cols = st.columns(max(len(status_counts), 1))
        for column, (status, count) in zip(status_cols, status_counts.items()):
            column.metric(str(status), f"{int(count):,}")
        numeric = [
            column
            for column in (
                "ins_bead_count",
                "gcg_bead_count",
                "nucleus_area",
                "bf_blur_laplacian_variance",
            )
            if column in qc
        ]
        if numeric:
            st.subheader("QC distribution")
            st.dataframe(qc[numeric].describe().T, use_container_width=True)
        st.dataframe(qc.head(200), use_container_width=True, hide_index=True)
    else:
        st.info(
            "The final QC report and preprocessed manifest are written after all TIFFs finish."
        )


def source_id(path: str) -> str:
    return AUGMENTATION_SUFFIX_RE.sub("", Path(path).stem)


def assign_manifest_splits(
    dataframe: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Assign augmented source groups to deterministic, leakage-free splits."""
    assignments: list[dict[str, str]] = []
    groups = dataframe[["hormone", "label", "source_id"]].drop_duplicates()
    for (hormone, label), group in groups.groupby(["hormone", "label"]):
        identifiers = group["source_id"].astype(str).tolist()
        identifiers.sort(
            key=lambda item: hashlib.sha1(
                f"{seed}:{hormone}:{label}:{item}".encode("utf-8")
            ).hexdigest()
        )
        count = len(identifiers)
        if count < 3:
            raise ValueError(
                f"{hormone}/{label} needs at least 3 independent source groups; found {count}."
            )
        test_count = max(1, int(round(count * test_fraction)))
        validation_count = max(1, int(round(count * validation_fraction)))
        while test_count + validation_count >= count:
            if test_count >= validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                raise ValueError(
                    f"Cannot create train/val/test splits for {hormone}/{label}."
                )
        for index, identifier in enumerate(identifiers):
            split = (
                "test" if index < test_count
                else "val" if index < test_count + validation_count
                else "train"
            )
            assignments.append(
                {
                    "hormone": str(hormone),
                    "label": str(label),
                    "source_id": identifier,
                    "split": split,
                }
            )
    assignment_df = pd.DataFrame(assignments)
    return dataframe.drop(columns=["split"], errors="ignore").merge(
        assignment_df,
        on=["hormone", "label", "source_id"],
        how="left",
        validate="many_to_one",
    )


def _apply_dataset_root(root_text: str) -> None:
    root = Path(root_text).expanduser()
    for hormone in ("insulin", "glucagon"):
        for label in ("high", "low"):
            st.session_state[f"manifest_{hormone}_{label}_typed"] = str(
                root / hormone / label
            )


def manifest_builder_page() -> None:
    st.header("Training Manifest Builder")
    st.caption(
        "Combine completed preprocessing folders, add hormone/high-low labels, "
        "and assign augmentation-safe train/validation/test splits."
    )
    dataset_root_text = path_browser(
        "Preprocessed dataset root",
        "manifest_dataset_root",
        kind="folder",
        default=str(ROOT / "data" / "preprocessed_dataset"),
    )
    st.button(
        "Apply root as insulin/high, insulin/low, glucagon/high, glucagon/low",
        on_click=_apply_dataset_root,
        args=(dataset_root_text,),
    )
    targets = st.multiselect(
        "Hormones to include",
        ("insulin", "glucagon"),
        default=("insulin", "glucagon"),
    )

    category_paths: dict[tuple[str, str], Path] = {}
    for hormone in targets:
        st.subheader(hormone.capitalize())
        columns = st.columns(2)
        for column, label in zip(columns, ("high", "low")):
            with column:
                path_text = path_browser(
                    f"{hormone.capitalize()} {label} preprocessing folder",
                    f"manifest_{hormone}_{label}",
                    kind="folder",
                    default=str(Path(dataset_root_text) / hormone / label),
                )
                category_paths[(hormone, label)] = Path(path_text).expanduser()

    output_folder_text = path_browser(
        "Manifest output folder",
        "manifest_output_folder",
        kind="folder",
        default=dataset_root_text,
    )
    output_name = st.text_input(
        "Manifest filename", value="training_manifest.csv"
    ).strip()
    c1, c2, c3 = st.columns(3)
    validation_fraction = float(
        c1.number_input("Validation fraction", 0.01, 0.45, 0.15, 0.01)
    )
    test_fraction = float(
        c2.number_input("Test fraction", 0.01, 0.45, 0.15, 0.01)
    )
    seed = int(c3.number_input("Split seed", 0, 1_000_000, 17))

    rows = []
    status_rows = []
    errors = []
    for (hormone, label), folder in category_paths.items():
        manifest_path = folder / "preprocessed_manifest.csv"
        if not manifest_path.is_file():
            errors.append(f"Missing: {manifest_path}")
            status_rows.append(
                {
                    "hormone": hormone,
                    "label": label,
                    "folder": str(folder),
                    "rows": 0,
                    "status": "missing manifest",
                }
            )
            continue
        try:
            category = pd.read_csv(manifest_path)
        except Exception as error:
            errors.append(f"Cannot read {manifest_path}: {error}")
            continue
        required = {"path", "preprocessed_path"}
        missing_columns = sorted(required - set(category.columns))
        if missing_columns:
            errors.append(
                f"{manifest_path} missing: {', '.join(missing_columns)}"
            )
            continue
        category = category.copy()
        category["hormone"] = hormone
        category["label"] = label
        category["source_id"] = category["path"].astype(str).map(source_id)
        rows.append(category)
        status_rows.append(
            {
                "hormone": hormone,
                "label": label,
                "folder": str(folder),
                "rows": len(category),
                "source_groups": category["source_id"].nunique(),
                "status": "ready",
            }
        )
    if status_rows:
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
    for error in errors:
        st.warning(error)

    invalid_output = (
        not output_name.lower().endswith(".csv")
        or Path(output_name).name != output_name
    )
    if invalid_output:
        st.error("Manifest filename must be a plain .csv filename.")
    if validation_fraction + test_fraction >= 1.0:
        st.error("Validation and test fractions must sum to less than 1.")
        invalid_output = True

    output_path = Path(output_folder_text).expanduser() / output_name
    if st.button(
        "Build training manifest",
        type="primary",
        disabled=bool(errors) or not rows or invalid_output,
    ):
        try:
            combined = pd.concat(rows, ignore_index=True)
            combined = assign_manifest_splits(
                combined, validation_fraction, test_fraction, seed
            )
            missing_npz = [
                path for path in combined["preprocessed_path"].astype(str)
                if not Path(path).is_file()
            ]
            if missing_npz:
                raise ValueError(
                    f"{len(missing_npz):,} NPZ files referenced by the inputs are missing."
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(output_path, index=False)
            st.session_state["training_manifest_typed"] = str(output_path)
            st.success(f"Wrote {len(combined):,} rows to {output_path}")
        except Exception as error:
            st.error(str(error))

    if output_path.is_file():
        built = pd.read_csv(output_path)
        st.subheader("Current output manifest")
        summary = (
            built.groupby(["hormone", "label", "split"])
            .size()
            .rename("images")
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.dataframe(built.head(200), use_container_width=True, hide_index=True)


def duration_text(seconds: object) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "-"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m" if hours else f"{minutes:d}m {secs:02d}s"


def training_status() -> None:
    batch_files = sorted(
        RUNS_DIR.glob("*_batch_progress.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    batch_rows = []
    active_batches: list[tuple[Path, dict]] = []
    for path in batch_files:
        batch = load_json(path)
        if not batch:
            continue
        total = int(batch.get("total_combinations", 0) or 0)
        completed = int(batch.get("completed_combinations", 0) or 0)
        current_progress = 0.0
        current_output = batch.get("current_output_dir")
        if current_output:
            run_progress = load_json(Path(str(current_output)) / "progress.json")
            current_progress = float(run_progress.get("overall_progress", 0.0) or 0.0)
        overall = min((completed + current_progress) / total, 1.0) if total else 0.0
        name = path.name.removesuffix("_batch_progress.json")
        batch_rows.append(
            {
                "run group": name,
                "status": batch.get("status", "unknown"),
                "combinations": f"{completed}/{total}",
                "current": (
                    f"{batch.get('current_model')} / {batch.get('current_target')}"
                    if batch.get("current_model") else "-"
                ),
                "overall": f"{overall:.1%}",
                "updated": batch.get("updated_at", "-"),
            }
        )
        if batch.get("status") == "running":
            active_batches.append((path, batch))
    if batch_rows:
        st.subheader("Multi-run queues")
        st.dataframe(pd.DataFrame(batch_rows), width="stretch", hide_index=True)

    if active_batches:
        labels = [path.name.removesuffix("_batch_progress.json") for path, _ in active_batches]
        selected_label = st.selectbox("Active run group", labels)
        selected_index = labels.index(selected_label)
        _batch_path, batch = active_batches[selected_index]
        total = int(batch.get("total_combinations", 0) or 0)
        completed = int(batch.get("completed_combinations", 0) or 0)
        current_output = Path(str(batch.get("current_output_dir")))
        progress = load_json(current_output / "progress.json")
        current_fraction = float(progress.get("overall_progress", 0.0) or 0.0)
        overall_fraction = min(
            (completed + current_fraction) / total, 1.0
        ) if total else 0.0
        st.progress(
            overall_fraction,
            text=(
                f"Overall combinations: {completed}/{total} complete | "
                f"current {batch.get('current_model')} / {batch.get('current_target')}"
            ),
        )
        metrics = st.columns(6)
        metrics[0].metric(
            "Combination", f"{batch.get('current_combination', 0)}/{total}"
        )
        metrics[1].metric(
            "Epoch",
            f"{progress.get('epoch', 0)}/{progress.get('total_epochs', 0)}",
        )
        metrics[2].metric(
            "Batch",
            f"{progress.get('batch', 0)}/{progress.get('total_batches', 0)}",
        )
        val_auroc = progress.get("val_auroc")
        metrics[3].metric(
            "Validation AUROC",
            f"{float(val_auroc):.4f}" if val_auroc is not None else "-",
        )
        best_score = progress.get("best_validation_score")
        metrics[4].metric(
            "Best score",
            f"{float(best_score):.4f}" if best_score is not None else "-",
        )
        metrics[5].metric("ETA", duration_text(progress.get("eta_seconds")))
        epoch_fraction = float(progress.get("epoch_progress", 0.0) or 0.0)
        st.progress(
            min(max(epoch_fraction, 0.0), 1.0),
            text=f"Current epoch: {epoch_fraction:.1%} | status={progress.get('status', 'waiting')}",
        )
        device_text = progress.get("gpu") or progress.get("device") or "waiting"
        st.caption(
            f"Device: {device_text} | elapsed: "
            f"{duration_text(progress.get('elapsed_seconds'))} | "
            f"updated: {progress.get('updated_at', batch.get('updated_at', '-'))}"
        )
        jobs = pd.DataFrame(batch.get("jobs", []))
        if not jobs.empty:
            st.dataframe(jobs, width="stretch", hide_index=True)
        with st.expander("Detailed training log", expanded=True):
            st.code(tail(current_output / "train.log", 45), language="text")

    rows = []
    for directory in run_dirs():
        config = load_json(directory / "run_config.json")
        progress = load_json(directory / "progress.json")
        history_path = directory / "history.csv"
        try:
            history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        except Exception:
            history = pd.DataFrame()
        completed = (
            int(history["epoch"].max())
            if not history.empty and "epoch" in history else 0
        )
        total = int(config.get("epochs", 0) or 0)
        if progress:
            status = str(progress.get("status", "unknown"))
        elif (directory / "training.lock").exists():
            status = "running"
        elif (directory / "test_metrics.json").exists():
            status = "complete"
        elif config:
            status = "stopped / failed"
        else:
            continue
        rows.append(
            {
                "run": directory.name,
                "status": status,
                "target": config.get("target", "label"),
                "model": MODEL_LABELS.get(
                    config.get("model"), config.get("model", "-")
                ),
                "epoch": f"{completed}/{total}" if total else str(completed),
                "batch": (
                    f"{progress.get('batch', 0)}/{progress.get('total_batches', 0)}"
                    if progress else "-"
                ),
                "progress": (
                    f"{float(progress.get('overall_progress', 0.0) or 0.0):.1%}"
                    if progress else "-"
                ),
                "ETA": duration_text(progress.get("eta_seconds")) if progress else "-",
                "NPZ only": bool(config.get("preprocessed", False)),
            }
        )
    if not rows:
        return
    st.subheader("Training jobs")
    status_df = pd.DataFrame(rows)
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    active_statuses = {"initializing", "loading_data", "running", "validating", "evaluating_test"}
    active = status_df.loc[status_df["status"].isin(active_statuses), "run"].tolist()
    if active:
        selected = st.selectbox("Live log", active)
        st.code(tail(RUNS_DIR / selected / "train.log"), language="text")


@st.fragment(run_every="3s")
def live_training_status() -> None:
    training_status()


def training_page() -> None:
    st.header("Independent NPZ training")
    st.success(
        "Every command launched here uses --preprocessed. Source TIFF files are not opened."
    )
    manifest_text = path_browser(
        "preprocessed_manifest.csv",
        "training_manifest",
        kind="file",
        extensions=(".csv",),
    )
    manifest = Path(manifest_text).expanduser() if manifest_text else Path()
    df, errors = (
        manifest_report(manifest) if manifest_text
        else (None, ["Select a completed training manifest."])
    )
    if df is not None:
        missing_npz = (
            sum(not Path(str(path)).is_file() for path in df["preprocessed_path"])
            if "preprocessed_path" in df else len(df)
        )
        split_counts = df.get("split", pd.Series(dtype=str)).value_counts().to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Train", f"{split_counts.get('train', 0):,}")
        c3.metric("Validation", f"{split_counts.get('val', 0):,}")
        c4.metric("Missing NPZ", f"{missing_npz:,}")
        if missing_npz:
            errors.append(f"{missing_npz:,} NPZ files are missing.")
    for error in errors:
        st.warning(error)

    st.subheader("1. Targets")
    target_columns = st.columns(2)
    selected_targets = []
    if target_columns[0].checkbox("Insulin", value=True, key="train_target_insulin"):
        selected_targets.append("insulin")
    if target_columns[1].checkbox("Glucagon", value=True, key="train_target_glucagon"):
        selected_targets.append("glucagon")

    st.subheader("2. Training mode")
    mode_columns = st.columns(2)
    recommended_mode = mode_columns[0].checkbox(
        "Recommended schedule per model",
        value=True,
        help="Use the validated epoch, batch size, learning rate, and patience defaults for each architecture.",
    )
    custom_mode = mode_columns[1].checkbox(
        "Shared custom schedule",
        value=False,
        help="Apply one custom schedule to every selected architecture.",
    )
    mode_valid = recommended_mode != custom_mode
    if not mode_valid:
        st.error("Select exactly one training mode.")
    schedule = "recommended" if recommended_mode else "custom"

    st.subheader("3. Models")
    model_columns = st.columns(len(MODEL_NAMES))
    selected_models = []
    for index, model_name in enumerate(MODEL_NAMES):
        if model_columns[index].checkbox(
            MODEL_LABELS[model_name],
            value=model_name == "resnet18",
            key=f"train_model_{model_name}",
        ):
            selected_models.append(model_name)

    run_name = st.text_input(
        "Run group name", value=f"npz_batch_{datetime.now():%Y%m%d_%H%M}"
    ).strip()

    defaults = MODEL_TRAINING_DEFAULTS["resnet18"]
    st.subheader("4. Schedule and output settings")
    s1, s2, s3, s4 = st.columns(4)
    epochs = int(
        s1.number_input(
            "Maximum epochs", 1, 1000, int(defaults["max_epochs"]),
            disabled=recommended_mode,
        )
    )
    batch_size = int(
        s2.number_input(
            "Batch size", 1, 512, int(defaults["batch_size"]),
            disabled=recommended_mode,
        )
    )
    lr = float(
        s3.number_input(
            "Learning rate", 0.000001, 1.0,
            float(defaults["learning_rate"]), format="%.6f",
            disabled=recommended_mode,
        )
    )
    patience = int(
        s4.number_input(
            "Early-stop patience", 0, 100, int(defaults["patience"]),
            disabled=recommended_mode,
        )
    )
    if recommended_mode and selected_models:
        schedule_rows = [
            {
                "model": MODEL_LABELS[name],
                "epochs": MODEL_TRAINING_DEFAULTS[name]["max_epochs"],
                "batch_size": MODEL_TRAINING_DEFAULTS[name]["batch_size"],
                "learning_rate": MODEL_TRAINING_DEFAULTS[name]["learning_rate"],
                "patience": MODEL_TRAINING_DEFAULTS[name]["patience"],
            }
            for name in selected_models
        ]
        st.dataframe(pd.DataFrame(schedule_rows), width="stretch", hide_index=True)
    a1, a2, a3, a4 = st.columns(4)
    image_size = int(a1.number_input("Image size", 32, 1024, 224, 16))
    weight_decay = float(
        a2.number_input("Weight decay", 0.0, 1.0, 0.0001, format="%.6f")
    )
    workers = int(a3.number_input("Workers", 0, 32, 0))
    preview_images = int(a4.number_input("Grad-CAM previews", 0, 500, 25))
    pretrained = st.checkbox("Use ImageNet pretrained weights", value=False)

    combination_count = len(selected_targets) * len(selected_models)
    st.info(
        f"One click will run {combination_count} target/model combination(s) "
        "sequentially. This avoids loading multiple large models onto the GPU at once."
    )
    if not selected_targets:
        st.warning("Select at least one target.")
    if not selected_models:
        st.warning("Select at least one model.")
    if df is not None and "hormone" in df.columns:
        available_targets = set(df["hormone"].dropna().astype(str).str.lower())
        unavailable = sorted(set(selected_targets) - available_targets)
        if unavailable:
            unavailable_message = (
                "Selected targets are not present in the manifest: "
                + ", ".join(unavailable)
            )
            errors.append(unavailable_message)
            st.warning(unavailable_message)

    invalid_name = (
        not run_name
        or Path(run_name).name != run_name
        or any(character in run_name for character in '<>:"/\\|?*')
    )
    if invalid_name:
        st.error("Run name contains invalid path characters.")
    output_prefix = RUNS_DIR / run_name
    command = [
        sys.executable,
        "train_preprocessed_batch.py",
        "--manifest", str(manifest),
        "--output-prefix", str(output_prefix),
        "--targets", *selected_targets,
        "--models", *selected_models,
        "--schedule", schedule,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--patience", str(patience),
        "--image-size", str(image_size),
        "--weight-decay", str(weight_decay),
        "--num-workers", str(workers),
        "--preview-images", str(preview_images),
    ]
    expected = [
        RUNS_DIR / f"{run_name}_{model_name}_{target}"
        for model_name in selected_models
        for target in selected_targets
    ]
    if pretrained:
        command.append("--pretrained")

    existing = [path for path in expected if path.exists()]
    if existing:
        st.error(
            "Run output already exists: "
            + ", ".join(path.name for path in existing)
        )
    with st.expander("Command preview"):
        st.code(subprocess.list2cmdline(command), language="powershell")
    can_start = (
        df is not None
        and not errors
        and not existing
        and not invalid_name
        and mode_valid
        and bool(selected_targets)
        and bool(selected_models)
    )
    if st.button(
        "Start background training", type="primary", disabled=not can_start
    ):
        pid = start_background(command)
        st.session_state["last_training_pid"] = pid
        st.success(
            f"Training started in the background (PID {pid}). Refresh to follow progress."
        )

    st.subheader("Live progress dashboard")
    auto_refresh = st.checkbox(
        "Auto-refresh every 3 seconds",
        value=True,
        key="training_auto_refresh",
    )
    if auto_refresh:
        live_training_status()
    else:
        if st.button("Refresh training status"):
            st.rerun()
        training_status()


def comparison_rows() -> pd.DataFrame:
    rows = []
    for directory in run_dirs():
        metrics = load_json(directory / "test_metrics.json")
        if not metrics:
            continue
        config = load_json(directory / "run_config.json")
        summary = load_json(directory / "training_summary.json")
        rows.append(
            {
                "run": directory.name,
                "target": config.get("target", summary.get("target", "label")),
                "model": MODEL_LABELS.get(
                    config.get("model"), config.get("model", "-")
                ),
                "NPZ only": bool(config.get("preprocessed", False)),
                "best_epoch": summary.get("best_epoch"),
                "completed_epoch": summary.get("completed_epoch"),
                "stop_reason": summary.get("stop_reason", "-"),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def comparison_page() -> None:
    st.header("Model results and comparison")
    table = comparison_rows()
    if table.empty:
        st.info("No completed runs with test_metrics.json were found under runs/.")
        return
    npz_only = st.checkbox("Show independent NPZ runs only", value=True)
    if npz_only:
        table = table[table["NPZ only"]]
    if table.empty:
        st.info("No completed independent NPZ runs are available yet.")
        return
    targets = sorted(str(value) for value in table["target"].dropna().unique())
    selected_targets = st.multiselect("Targets", targets, default=targets)
    if selected_targets:
        table = table[table["target"].astype(str).isin(selected_targets)]
    metric_options = [
        column
        for column in (
            "balanced_accuracy", "auroc", "accuracy", "average_precision",
            "high_recall", "low_recall",
        )
        if column in table
    ]
    if table.empty or not metric_options:
        st.info("No matching numeric metrics are available.")
        return
    sort_metric = st.selectbox("Rank by", metric_options)
    table = table.sort_values(sort_metric, ascending=False)
    st.dataframe(table, use_container_width=True, hide_index=True)

    default_runs = table["run"].head(4).tolist()
    selected_runs = st.multiselect(
        "Runs to compare", table["run"].tolist(), default=default_runs
    )
    selected = table[table["run"].isin(selected_runs)].copy()
    metric_columns = [
        column
        for column in (
            "balanced_accuracy", "auroc", "accuracy", "high_recall", "low_recall"
        )
        if column in selected
    ]
    if not selected.empty and metric_columns:
        st.subheader("Test metrics")
        st.bar_chart(selected.set_index("run")[metric_columns])

    histories = []
    for run in selected_runs:
        history_path = RUNS_DIR / run / "history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        for metric in ("train_loss", "val_loss", "val_auroc", "val_accuracy"):
            if metric in history:
                histories.append(
                    history[["epoch", metric]]
                    .assign(run=run, metric=metric)
                    .rename(columns={metric: "value"})
                )
    if histories:
        history_df = pd.concat(histories, ignore_index=True)
        metric = st.selectbox(
            "Learning curve", sorted(history_df["metric"].unique())
        )
        curve = history_df[history_df["metric"] == metric].pivot(
            index="epoch", columns="run", values="value"
        )
        st.line_chart(curve)

    detail_run = st.selectbox("Run details", table["run"].tolist())
    detail_dir = RUNS_DIR / detail_run
    metrics = load_json(detail_dir / "test_metrics.json")
    if metrics:
        columns = st.columns(min(len(metrics), 6))
        for column, (name, value) in zip(columns, metrics.items()):
            column.metric(
                name, f"{value:.4f}" if isinstance(value, float) else value
            )
    images = [
        detail_dir / name
        for name in (
            "training_curves.png",
            "test_confusion_matrix.png",
            "test_roc_curve.png",
            "test_precision_recall_curve.png",
        )
        if (detail_dir / name).exists()
    ]
    if images:
        columns = st.columns(2)
        for index, image in enumerate(images):
            columns[index % 2].image(
                str(image), caption=image.name, use_container_width=True
            )
    predictions = detail_dir / "test_predictions.csv"
    if predictions.exists():
        with st.expander("Test predictions"):
            st.dataframe(
                pd.read_csv(predictions), use_container_width=True, hide_index=True
            )


def qc_focus_page() -> None:
    st.header("QC / Model Focus Preview")
    candidates = [
        directory
        for directory in run_dirs()
        if (directory / "best_model.pt").exists()
        and load_json(directory / "run_config.json").get("preprocessed", False)
    ]
    if not candidates:
        st.info("No NPZ-trained model runs are available yet.")
        return
    run_dir = st.selectbox(
        "Model run",
        candidates,
        format_func=lambda path: path.name,
    )
    examples_dir = run_dir / "preprocess_attention_examples"
    examples_csv = examples_dir / "examples.csv"
    if not examples_csv.exists():
        st.warning(
            "This run has no saved model-focus examples. Set Grad-CAM previews "
            "to at least 1 when starting a new training run. The dashboard default is 25."
        )
        return
    examples = pd.read_csv(examples_csv)
    c1, c2, c3 = st.columns(3)
    filter_mode = c1.radio(
        "Filter",
        ("All", "Correct only", "Wrong only"),
        horizontal=True,
    )
    probability_min = c2.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )
    page_size = int(c3.number_input("Images to show", 1, 100, 20))
    if filter_mode == "Correct only":
        examples = examples[
            examples["label"].astype(str) == examples["predicted_label"].astype(str)
        ]
    elif filter_mode == "Wrong only":
        examples = examples[
            examples["label"].astype(str) != examples["predicted_label"].astype(str)
        ]
    if "prob_high" in examples:
        confidence = examples["prob_high"].where(
            examples["predicted_label"].astype(str) == "high",
            1.0 - examples["prob_high"],
        )
        examples = examples[confidence >= probability_min]

    total = len(pd.read_csv(examples_csv))
    wrong = int(
        (
            pd.read_csv(examples_csv)["label"].astype(str)
            != pd.read_csv(examples_csv)["predicted_label"].astype(str)
        ).sum()
    )
    metrics = st.columns(3)
    metrics[0].metric("Saved previews", f"{total:,}")
    metrics[1].metric("Misclassified", f"{wrong:,}")
    metrics[2].metric("Displayed", f"{min(len(examples), page_size):,}")
    st.dataframe(examples.head(200), use_container_width=True, hide_index=True)
    columns = st.columns(2)
    for display_index, (_, row) in enumerate(examples.head(page_size).iterrows()):
        image_path = Path(str(row["example_image"]))
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        if not image_path.exists():
            continue
        caption = (
            f"{Path(str(row['path'])).name} | true={row['label']} | "
            f"pred={row['predicted_label']} | P(high)={float(row['prob_high']):.3f}"
        )
        columns[display_index % 2].image(
            str(image_path), caption=caption, use_container_width=True
        )


def checkpoint_runs(target: str) -> list[Path]:
    candidates = []
    for directory in run_dirs():
        if not (directory / "best_model.pt").exists():
            continue
        config = load_json(directory / "run_config.json")
        if not config.get("preprocessed", False):
            continue
        if str(config.get("target", "")).lower() != target:
            continue
        candidates.append(directory)
    return candidates


def checkpoint_label(directory: Path) -> str:
    config = load_json(directory / "run_config.json")
    metrics = load_json(directory / "test_metrics.json")
    model = MODEL_LABELS.get(config.get("model"), config.get("model", "-"))
    score = metrics.get("balanced_accuracy")
    score_text = f" | balanced accuracy {score:.3f}" if isinstance(score, float) else ""
    return f"{directory.name} | {model}{score_text}"


def prediction_page() -> None:
    st.header("NPZ Prediction")
    st.success(
        "Prediction reads preprocessed_path tensors only and rejects checkpoints "
        "that were trained with the legacy TIFF-on-the-fly workflow."
    )
    manifest_text = path_browser(
        "Prediction manifest CSV",
        "prediction_manifest",
        kind="file",
        extensions=(".csv",),
    )
    manifest = Path(manifest_text).expanduser() if manifest_text else Path()
    prediction_df = None
    manifest_error = ""
    if manifest_text:
        try:
            prediction_df = pd.read_csv(manifest)
            if "preprocessed_path" not in prediction_df:
                manifest_error = "Manifest must contain preprocessed_path."
            else:
                missing = sum(
                    not Path(str(path)).is_file()
                    for path in prediction_df["preprocessed_path"]
                )
                columns = st.columns(3)
                columns[0].metric("Rows", f"{len(prediction_df):,}")
                columns[1].metric("Missing NPZ", f"{missing:,}")
                columns[2].metric(
                    "Labeled", "Yes" if "label" in prediction_df else "No"
                )
                if missing:
                    manifest_error = f"{missing:,} referenced NPZ files are missing."
        except Exception as error:
            manifest_error = str(error)
    else:
        manifest_error = "Select a preprocessed manifest."
    if manifest_error:
        st.warning(manifest_error)

    selected_checkpoints: dict[str, Path] = {}
    checkpoint_columns = st.columns(2)
    for column, target in zip(checkpoint_columns, ("insulin", "glucagon")):
        with column:
            use_model = st.checkbox(
                f"Use {target} model", value=True, key=f"predict_use_{target}"
            )
            options = checkpoint_runs(target)
            selected = st.selectbox(
                f"{target.capitalize()} checkpoint",
                options,
                format_func=checkpoint_label,
                index=0 if options else None,
                placeholder=f"No completed {target} NPZ model",
                disabled=not use_model,
                key=f"predict_checkpoint_{target}",
            )
            if use_model and selected:
                selected_checkpoints[target] = selected / "best_model.pt"

    output_folder_text = path_browser(
        "Prediction output folder",
        "prediction_output_folder",
        kind="folder",
        default=str(ROOT / "predictions"),
    )
    output_filename = st.text_input(
        "Prediction CSV filename",
        value=f"npz_predictions_{datetime.now():%Y%m%d_%H%M}.csv",
    ).strip()
    output = Path(output_folder_text).expanduser() / output_filename
    command = [
        sys.executable,
        "predict_preprocessed.py",
        "--manifest",
        str(manifest),
        "--out",
        str(output),
    ]
    for target, checkpoint in selected_checkpoints.items():
        command.extend([f"--{target}-checkpoint", str(checkpoint)])
    with st.expander("Command preview"):
        st.code(subprocess.list2cmdline(command), language="powershell")
    can_run = (
        prediction_df is not None
        and not manifest_error
        and bool(selected_checkpoints)
        and bool(output_folder_text.strip())
        and output_filename.lower().endswith(".csv")
    )
    if st.button("Run NPZ prediction", type="primary", disabled=not can_run):
        with st.spinner("Running prediction from cached NPZ tensors..."):
            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        if result.returncode == 0:
            st.success(result.stdout.strip() or f"Predictions written to {output}")
        else:
            st.error(result.stderr or result.stdout or "Prediction failed.")

    if output.exists():
        results = pd.read_csv(output)
        st.subheader("Prediction results")
        if "label" in results:
            metric_columns = st.columns(max(len(selected_checkpoints), 1))
            for column, target in zip(metric_columns, selected_checkpoints):
                predicted = f"{target}_predicted_label"
                target_rows = results
                if "hormone" in results:
                    target_rows = results[
                        results["hormone"].astype(str).str.lower() == target
                    ]
                if predicted in target_rows and not target_rows.empty:
                    accuracy = (
                        target_rows["label"].astype(str).str.lower()
                        == target_rows[predicted].astype(str).str.lower()
                    ).mean()
                    column.metric(f"{target.capitalize()} accuracy", f"{accuracy:.1%}")
        st.dataframe(results, use_container_width=True, hide_index=True)


def main() -> None:
    st.sidebar.title("Secretion NPZ")
    st.sidebar.caption("Independent preprocessing and model training")
    page = st.sidebar.radio(
        "Dashboard",
        (
            "Pre-processing",
            "Manifest Builder",
            "Training",
            "Model comparison",
            "QC / Model Focus",
            "Prediction",
        ),
    )
    if page == "Pre-processing":
        preprocessing_page()
    elif page == "Manifest Builder":
        manifest_builder_page()
    elif page == "Training":
        training_page()
    elif page == "Model comparison":
        comparison_page()
    elif page == "QC / Model Focus":
        qc_focus_page()
    else:
        prediction_page()


if __name__ == "__main__":
    main()
