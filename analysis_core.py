from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly


MISSING_STATE = "missing"
DETECTED_STATE = "detected"
PREDICTED_STATE = "predicted"


@dataclass
class AnalysisBundle:
    experiment_name: str
    info_data: dict
    bee_ids: List[str]
    bin_seconds: int
    selected_metrics: Dict[str, bool]
    framewise_compare_df: pd.DataFrame
    framewise_by_source: Dict[str, pd.DataFrame]
    binned_by_source: Dict[str, pd.DataFrame]
    source_paths: Dict[str, Optional[str]]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_geom(geom_str: str):
    numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", geom_str or "")]
    if len(numbers) == 3:
        return [int(numbers[0]), int(numbers[1]), int(numbers[2])]
    if len(numbers) == 4:
        return [int(numbers[0]), int(numbers[1]), int(numbers[2]), int(numbers[3])]
    if len(numbers) > 4 and len(numbers) % 2 == 0:
        return [(int(numbers[i]), int(numbers[i + 1])) for i in range(0, len(numbers), 2)]
    return None


def get_shape_center(shape_data: dict) -> Tuple[float, float]:
    shape = shape_data.get("shape")
    geom = shape_data.get("geom")
    if not geom:
        return (math.nan, math.nan)
    if shape == "circle":
        return float(geom[0]), float(geom[1])
    if shape == "rect":
        return float(geom[0] + geom[2] / 2.0), float(geom[1] + geom[3] / 2.0)
    if shape == "poly":
        pts = np.asarray(geom, dtype=float)
        if len(pts) == 0:
            return (math.nan, math.nan)
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    return (math.nan, math.nan)


def parse_info_file(info_filepath: str) -> Optional[dict]:
    info = {
        "fps": None,
        "pixel_to_mm": 1.0,
        "frame_width": 2000,
        "frame_height": 2000,
        "start_time_str": "N/A",
        "end_time_str": "N/A",
        "arenas": {},
        "stimulus_areas": {},
        "raw_missing_rule": "",
        "filtered_prediction_rule": "",
    }

    current_section = None
    try:
        with open(info_filepath, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("#"):
                    label = line.lower()
                    if "video info" in label:
                        current_section = "video"
                    elif "scale" in label:
                        current_section = "scale"
                    elif "arenas" in label:
                        current_section = "arenas"
                    elif "stimulus areas" in label:
                        current_section = "stimulus_areas"
                    else:
                        current_section = None
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                key = parts[0].strip()
                if key == "id":
                    continue

                if current_section == "video":
                    value = parts[1].strip()
                    if key == "fps":
                        info["fps"] = float(value)
                    elif key == "frame_width":
                        info["frame_width"] = int(float(value))
                    elif key == "frame_height":
                        info["frame_height"] = int(float(value))
                    elif key == "start_time_str":
                        info["start_time_str"] = value
                    elif key == "end_time_str":
                        info["end_time_str"] = value
                    else:
                        info[key] = value
                elif current_section == "scale" and key == "pixel_to_mm":
                    info["pixel_to_mm"] = float(parts[1])
                elif current_section in {"arenas", "stimulus_areas"} and len(parts) >= 4:
                    shape_id, shape_type, category, geom_str = parts[0], parts[1], parts[2], parts[3]
                    shape_data = {
                        "id": shape_id,
                        "shape": shape_type,
                        "category": category,
                        "geom": parse_geom(geom_str),
                    }
                    shape_data["center"] = get_shape_center(shape_data)
                    info[current_section][shape_id] = shape_data
                else:
                    if len(parts) >= 2:
                        info[key] = parts[1].strip()
    except FileNotFoundError:
        return None
    except Exception:
        return None

    if info["fps"] is None:
        info["fps"] = 30.0
    return info


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _draw_shape_on_mask(mask: np.ndarray, shape_data: dict, value: int = 255) -> None:
    if not shape_data or not shape_data.get("geom"):
        return
    shape = shape_data["shape"]
    geom = shape_data["geom"]
    if shape == "circle":
        cv2.circle(mask, (geom[0], geom[1]), geom[2], value, -1)
    elif shape == "rect":
        cv2.rectangle(mask, (geom[0], geom[1]), (geom[0] + geom[2], geom[1] + geom[3]), value, -1)
    elif shape == "poly":
        pts = np.array(geom, dtype=np.int32)
        cv2.fillPoly(mask, [pts], value)


def draw_shape_outline(image: np.ndarray, shape_data: dict, color: Tuple[int, int, int], thickness: int = 2) -> None:
    if not shape_data or not shape_data.get("geom"):
        return
    shape = shape_data["shape"]
    geom = shape_data["geom"]
    if shape == "circle":
        cv2.circle(image, (geom[0], geom[1]), geom[2], color, thickness)
    elif shape == "rect":
        cv2.rectangle(image, (geom[0], geom[1]), (geom[0] + geom[2], geom[1] + geom[3]), color, thickness)
    elif shape == "poly":
        pts = np.array(geom, dtype=np.int32)
        cv2.polylines(image, [pts], True, color, thickness)


def is_point_in_shape(point: Tuple[float, float], shape_data: dict) -> bool:
    x, y = point
    if pd.isna(x) or pd.isna(y):
        return False
    shape = shape_data.get("shape")
    geom = shape_data.get("geom")
    if not geom:
        return False
    if shape == "circle":
        return (x - geom[0]) ** 2 + (y - geom[1]) ** 2 <= geom[2] ** 2
    if shape == "rect":
        return geom[0] <= x <= geom[0] + geom[2] and geom[1] <= y <= geom[1] + geom[3]
    if shape == "poly":
        pts = np.asarray(geom, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0
    return False


def points_inside_shape_vectorized(x: np.ndarray, y: np.ndarray, shape_data: dict) -> np.ndarray:
    """Vectorized equivalent of is_point_in_shape() for whole arrays at once.

    Returns a float array with 1.0/0.0 for inside/outside, and NaN wherever
    x or y is NaN. This replaces a per-row Python loop (previously the main
    performance bottleneck for large datasets) with numpy/matplotlib bulk
    operations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = np.full(x.shape, np.nan, dtype=float)

    valid = ~(np.isnan(x) | np.isnan(y))
    if not np.any(valid):
        return result

    shape = shape_data.get("shape")
    geom = shape_data.get("geom")
    if not geom:
        result[valid] = 0.0
        return result

    vx = x[valid]
    vy = y[valid]

    if shape == "circle":
        cx, cy, r = geom[0], geom[1], geom[2]
        inside = (vx - cx) ** 2 + (vy - cy) ** 2 <= r ** 2
    elif shape == "rect":
        rx, ry, rw, rh = geom
        inside = (vx >= rx) & (vx <= rx + rw) & (vy >= ry) & (vy <= ry + rh)
    elif shape == "poly":
        from matplotlib.path import Path as MplPath

        path = MplPath(np.asarray(geom, dtype=float))
        pts = np.column_stack([vx, vy])
        inside = path.contains_points(pts, radius=0.0)
    else:
        inside = np.zeros(vx.shape, dtype=bool)

    result[valid] = inside.astype(float)
    return result


def shape_intersections(info_data: dict) -> List[Tuple[str, str]]:
    arenas = info_data.get("arenas", {})
    stimuli = info_data.get("stimulus_areas", {})
    w = int(info_data.get("frame_width", 0))
    h = int(info_data.get("frame_height", 0))
    if w <= 0 or h <= 0:
        return []

    pairs = []
    for arena_id, arena in arenas.items():
        for stim_id, stim in stimuli.items():
            mask1 = np.zeros((h, w), dtype=np.uint8)
            mask2 = np.zeros((h, w), dtype=np.uint8)
            _draw_shape_on_mask(mask1, arena)
            _draw_shape_on_mask(mask2, stim)
            if cv2.countNonZero(cv2.bitwise_and(mask1, mask2)) > 0:
                pairs.append((arena_id, stim_id))
    return pairs


# ---------------------------------------------------------------------------
# Coordinate loading / validation
# ---------------------------------------------------------------------------

ID_COLUMN_RE = re.compile(r"^(ID_.+?)_(X|Y|Ang)$")


def detect_bee_ids(columns: Iterable[str]) -> List[str]:
    prefixes = set()
    for col in columns:
        match = ID_COLUMN_RE.match(str(col).strip())
        if match:
            prefixes.add(match.group(1))
    def sort_key(text: str):
        nums = re.findall(r"\d+", text)
        return int(nums[-1]) if nums else text
    return sorted(prefixes, key=sort_key)


def load_coordinate_file(coord_filepath: Optional[str]) -> Optional[pd.DataFrame]:
    if not coord_filepath:
        return None
    try:
        df = pd.read_csv(coord_filepath, sep="\t")
    except FileNotFoundError:
        return None
    except Exception:
        return None

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    bee_ids = detect_bee_ids(df.columns)
    if not bee_ids:
        raise ValueError(f"No bee ID columns found in {coord_filepath}")

    for bee_id in bee_ids:
        cols = [f"{bee_id}_X", f"{bee_id}_Y", f"{bee_id}_Ang"]
        if not all(c in df.columns for c in cols):
            raise ValueError(f"Incomplete coordinate triplet for {bee_id} in {coord_filepath}")
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        missing_triplet = (df[cols[0]] == -1) & (df[cols[1]] == -1) & (df[cols[2]] == -1)
        df.loc[missing_triplet, cols] = np.nan
    return df


# ---------------------------------------------------------------------------
# Framewise construction
# ---------------------------------------------------------------------------

def wrap_angle_diff(current: float, previous: float) -> float:
    diff = (current - previous + 180.0) % 360.0 - 180.0
    return diff


def _series_or_nan(df: Optional[pd.DataFrame], column: str, n_rows: int) -> pd.Series:
    if df is None or column not in df.columns:
        return pd.Series(np.nan, index=np.arange(n_rows), dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def build_framewise_compare_df(
    experiment_name: str,
    raw_df: Optional[pd.DataFrame],
    filtered_df: Optional[pd.DataFrame],
    info_data: dict,
) -> pd.DataFrame:
    fps = float(info_data.get("fps", 30.0))
    raw_rows = len(raw_df) if raw_df is not None else 0
    filtered_rows = len(filtered_df) if filtered_df is not None else 0
    n_rows = max(raw_rows, filtered_rows)
    if n_rows == 0:
        return pd.DataFrame()

    bee_ids = sorted(set(detect_bee_ids(raw_df.columns if raw_df is not None else [])) | set(detect_bee_ids(filtered_df.columns if filtered_df is not None else [])))
    records = []
    for bee_id in bee_ids:
        raw_x = _series_or_nan(raw_df, f"{bee_id}_X", n_rows)
        raw_y = _series_or_nan(raw_df, f"{bee_id}_Y", n_rows)
        raw_ang = _series_or_nan(raw_df, f"{bee_id}_Ang", n_rows)
        filtered_x = _series_or_nan(filtered_df, f"{bee_id}_X", n_rows)
        filtered_y = _series_or_nan(filtered_df, f"{bee_id}_Y", n_rows)
        filtered_ang = _series_or_nan(filtered_df, f"{bee_id}_Ang", n_rows)

        raw_valid = raw_x.notna() & raw_y.notna() & raw_ang.notna()
        filtered_valid = filtered_x.notna() & filtered_y.notna() & filtered_ang.notna()

        state = np.where(raw_valid, DETECTED_STATE, np.where(filtered_valid, PREDICTED_STATE, MISSING_STATE))

        bee_df = pd.DataFrame(
            {
                "experiment_name": experiment_name,
                "bee_id": bee_id,
                "frame_index": np.arange(n_rows, dtype=int),
                "time_s": np.arange(n_rows, dtype=float) / fps,
                "raw_x": raw_x.values,
                "raw_y": raw_y.values,
                "raw_ang": raw_ang.values,
                "filtered_x": filtered_x.values,
                "filtered_y": filtered_y.values,
                "filtered_ang": filtered_ang.values,
                "state": state,
            }
        )
        records.append(bee_df)

    return pd.concat(records, ignore_index=True)


def build_source_framewise_df(compare_df: pd.DataFrame, info_data: dict, source: str) -> pd.DataFrame:
    if compare_df.empty:
        return pd.DataFrame()
    if source not in {"raw", "filtered"}:
        raise ValueError("source must be 'raw' or 'filtered'")

    fps = float(info_data.get("fps", 30.0))
    pixel_to_mm = float(info_data.get("pixel_to_mm", 1.0))
    arenas = info_data.get("arenas", {})
    stimuli = info_data.get("stimulus_areas", {})

    x_col = f"{source}_x"
    y_col = f"{source}_y"
    ang_col = f"{source}_ang"

    df = compare_df[["experiment_name", "bee_id", "frame_index", "time_s", "state", x_col, y_col, ang_col]].copy()
    df = df.rename(columns={x_col: "x", y_col: "y", ang_col: "ang"})
    df["source"] = source
    valid = df[["x", "y", "ang"]].notna().all(axis=1)
    df["has_point"] = valid
    df["speed_mm_s"] = np.nan
    df["angle_change_deg"] = np.nan

    x_all = df["x"].to_numpy(dtype=float)
    y_all = df["y"].to_numpy(dtype=float)

    for shape_id, shape_data in arenas.items():
        center = shape_data.get("center", (math.nan, math.nan))
        df[f"inside_{shape_id}"] = points_inside_shape_vectorized(x_all, y_all, shape_data)
        df[f"dist_to_{shape_id}_center_mm"] = np.hypot(x_all - center[0], y_all - center[1]) * pixel_to_mm

    for shape_id, shape_data in stimuli.items():
        center = shape_data.get("center", (math.nan, math.nan))
        df[f"inside_{shape_id}"] = points_inside_shape_vectorized(x_all, y_all, shape_data)
        df[f"dist_to_{shape_id}_center_mm"] = np.hypot(x_all - center[0], y_all - center[1]) * pixel_to_mm
        df[f"entry_{shape_id}"] = np.nan
        df[f"exit_{shape_id}"] = np.nan

    # dist_to_*_center_mm should be NaN wherever the point itself is missing
    # (np.hypot on NaN already propagates NaN automatically, so no extra masking needed here).

    output_groups = []
    for bee_id, bee_df in df.groupby("bee_id", sort=False):
        bee_df = bee_df.sort_values("frame_index").copy()
        valid_mask = bee_df["has_point"].to_numpy(dtype=bool)
        x = bee_df["x"].to_numpy(dtype=float)
        y = bee_df["y"].to_numpy(dtype=float)
        ang = bee_df["ang"].to_numpy(dtype=float)

        speed = np.full(len(bee_df), np.nan, dtype=float)
        angle_change = np.full(len(bee_df), np.nan, dtype=float)

        if len(bee_df) > 1:
            pair_valid = valid_mask[1:] & valid_mask[:-1]
            dist_px = np.hypot(np.diff(x), np.diff(y))
            speed_vals = dist_px * pixel_to_mm * fps
            angle_vals = wrap_angle_diff(ang[1:], ang[:-1])
            speed[1:] = np.where(pair_valid, speed_vals, np.nan)
            angle_change[1:] = np.where(pair_valid, angle_vals, np.nan)

        bee_df["speed_mm_s"] = speed
        bee_df["angle_change_deg"] = angle_change

        for stim_id in stimuli:
            inside = bee_df[f"inside_{stim_id}"].to_numpy(dtype=float)
            entry = np.full(len(bee_df), np.nan, dtype=float)
            exit_ = np.full(len(bee_df), np.nan, dtype=float)

            valid_i = ~np.isnan(inside)
            prev = np.full(len(bee_df), np.nan, dtype=float)
            if len(bee_df) > 1:
                prev[1:] = inside[:-1]

            no_prev = valid_i & np.isnan(prev)
            entry[no_prev] = (inside[no_prev] == 1.0).astype(float)
            exit_[no_prev] = 0.0

            has_prev = valid_i & ~np.isnan(prev)
            entry[has_prev] = ((prev[has_prev] == 0.0) & (inside[has_prev] == 1.0)).astype(float)
            exit_[has_prev] = ((prev[has_prev] == 1.0) & (inside[has_prev] == 0.0)).astype(float)

            bee_df[f"entry_{stim_id}"] = entry
            bee_df[f"exit_{stim_id}"] = exit_

        output_groups.append(bee_df)

    ordered = pd.concat(output_groups, ignore_index=True)

    preferred_front = [
        "experiment_name",
        "bee_id",
        "source",
        "frame_index",
        "time_s",
        "x",
        "y",
        "ang",
        "state",
        "has_point",
        "speed_mm_s",
        "angle_change_deg",
    ]
    remaining = [c for c in ordered.columns if c not in preferred_front]
    return ordered[preferred_front + remaining]


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_source_framewise_df(
    experiment_name: str,
    framewise_df: pd.DataFrame,
    info_data: dict,
    bin_seconds: int,
    selected_metrics: Dict[str, bool],
) -> pd.DataFrame:
    if framewise_df.empty:
        return pd.DataFrame()

    fps = float(info_data.get("fps", 30.0))
    bin_frames = max(1, int(round(bin_seconds * fps)))
    arenas = list(info_data.get("arenas", {}).keys())
    stimuli = list(info_data.get("stimulus_areas", {}).keys())

    records = []
    for bee_id, bee_df in framewise_df.groupby("bee_id", sort=False):
        bee_df = bee_df.sort_values("frame_index").copy()
        total_frames = len(bee_df)
        for start in range(0, total_frames, bin_frames):
            end = min(start + bin_frames, total_frames)
            chunk = bee_df.iloc[start:end].copy()
            if chunk.empty:
                continue
            duration_s = max((end - start) / fps, 1.0 / fps)
            result = {
                "ExperimentName": experiment_name,
                "BeeID": bee_id,
                "Source": chunk["source"].iloc[0],
                "BinIndex": start // bin_frames,
                "Time_Start_s": start / fps,
                "Time_End_s": end / fps,
                "BinDuration_s": duration_s,
            }

            if selected_metrics.get("speed", True):
                valid_xy = chunk[["x", "y"]].dropna()
                if len(valid_xy) >= 2:
                    diffs = valid_xy.diff().dropna()
                    path_px = np.sqrt(diffs["x"] ** 2 + diffs["y"] ** 2).sum()
                    result["AvgSpeed_mm_s"] = path_px * info_data.get("pixel_to_mm", 1.0) / duration_s
                else:
                    result["AvgSpeed_mm_s"] = np.nan

            if selected_metrics.get("dist_arena", True):
                for arena_id in arenas:
                    col = f"dist_to_{arena_id}_center_mm"
                    result[f"AvgDistArenaCenter_{arena_id}_mm"] = chunk[col].mean(skipna=True) if col in chunk.columns else np.nan

            if selected_metrics.get("stim_duration", True):
                for stim_id in stimuli:
                    col = f"inside_{stim_id}"
                    result[f"Duration_in_Stim_{stim_id}_s"] = float(np.nansum(chunk[col].to_numpy(dtype=float))) / fps if col in chunk.columns else np.nan

            if selected_metrics.get("stim_entries", True):
                for stim_id in stimuli:
                    entry_col = f"entry_{stim_id}"
                    exit_col = f"exit_{stim_id}"
                    result[f"Entries_to_Stim_{stim_id}"] = float(np.nansum(chunk[entry_col].to_numpy(dtype=float))) if entry_col in chunk.columns else np.nan
                    result[f"Exits_from_Stim_{stim_id}"] = float(np.nansum(chunk[exit_col].to_numpy(dtype=float))) if exit_col in chunk.columns else np.nan

            if selected_metrics.get("stim_dist", True):
                for stim_id in stimuli:
                    col = f"dist_to_{stim_id}_center_mm"
                    result[f"AvgDistStimCenter_{stim_id}_mm"] = chunk[col].mean(skipna=True) if col in chunk.columns else np.nan

            records.append(result)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def build_analysis_bundle(
    experiment_name: str,
    info_data: dict,
    raw_df: Optional[pd.DataFrame],
    filtered_df: Optional[pd.DataFrame],
    bin_seconds: int,
    selected_metrics: Dict[str, bool],
    source_paths: Optional[Dict[str, Optional[str]]] = None,
    progress_callback=None,
) -> AnalysisBundle:
    def _report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    _report("Building framewise comparison table...")
    compare_df = build_framewise_compare_df(experiment_name, raw_df, filtered_df, info_data)
    bee_ids = sorted(compare_df["bee_id"].unique().tolist()) if not compare_df.empty else []

    framewise_by_source: Dict[str, pd.DataFrame] = {}
    binned_by_source: Dict[str, pd.DataFrame] = {}

    if raw_df is not None:
        _report("Processing raw coordinates (positions, speed, arena/stimulus state)...")
        framewise_by_source["raw"] = build_source_framewise_df(compare_df, info_data, "raw")
        _report("Binning raw data into time windows...")
        binned_by_source["raw"] = bin_source_framewise_df(experiment_name, framewise_by_source["raw"], info_data, bin_seconds, selected_metrics)
    if filtered_df is not None:
        _report("Processing filtered coordinates (positions, speed, arena/stimulus state)...")
        framewise_by_source["filtered"] = build_source_framewise_df(compare_df, info_data, "filtered")
        _report("Binning filtered data into time windows...")
        binned_by_source["filtered"] = bin_source_framewise_df(experiment_name, framewise_by_source["filtered"], info_data, bin_seconds, selected_metrics)

    _report("Finalizing results...")
    return AnalysisBundle(
        experiment_name=experiment_name,
        info_data=info_data,
        bee_ids=bee_ids,
        bin_seconds=bin_seconds,
        selected_metrics=selected_metrics,
        framewise_compare_df=compare_df,
        framewise_by_source=framewise_by_source,
        binned_by_source=binned_by_source,
        source_paths=source_paths or {"raw": None, "filtered": None},
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_experiment_files(directory: str) -> Optional[dict]:
    dir_path = Path(directory)
    info_files = sorted(dir_path.glob("*_info.txt"))
    if not info_files:
        return None

    if len(info_files) == 1:
        info_path = info_files[0]
        experiment_name = info_path.name[: -len("_info.txt")]
    else:
        prefixes = []
        for p in info_files:
            prefixes.append(p.name[: -len("_info.txt")])
        if len(prefixes) != 1:
            raise ValueError("Multiple experiment info files found. Please keep one experiment per folder.")
        experiment_name = prefixes[0]
        info_path = dir_path / f"{experiment_name}_info.txt"

    raw_path = dir_path / f"{experiment_name}_coordinates_raw.txt"
    filtered_path = dir_path / f"{experiment_name}_coordinates_filtered.txt"
    return {
        "experiment_name": experiment_name,
        "info_path": str(info_path),
        "raw_path": str(raw_path) if raw_path.exists() else None,
        "filtered_path": str(filtered_path) if filtered_path.exists() else None,
    }


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def export_tables(bundle: AnalysisBundle, output_directory: str) -> dict:
    out_dir = Path(output_directory)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    compare_path = tables_dir / f"{bundle.experiment_name}_framewise_compare.tsv"
    bundle.framewise_compare_df.to_csv(compare_path, sep="\t", index=False, float_format="%.6f")
    saved["framewise_compare"] = str(compare_path)

    for source, df in bundle.framewise_by_source.items():
        frame_path = tables_dir / f"{bundle.experiment_name}_framewise_{source}.tsv"
        bin_path = tables_dir / f"{bundle.experiment_name}_binned_{source}_{bundle.bin_seconds}s.tsv"
        df.to_csv(frame_path, sep="\t", index=False, float_format="%.6f")
        bundle.binned_by_source[source].to_csv(bin_path, sep="\t", index=False, float_format="%.6f")
        saved[f"framewise_{source}"] = str(frame_path)
        saved[f"binned_{source}"] = str(bin_path)
    return saved


# ---------------------------------------------------------------------------
# Static plots
# ---------------------------------------------------------------------------

def _time_plot_shapes(max_time: float) -> List[dict]:
    shapes = []
    ten_min = 600
    hour = 3600
    t = ten_min
    while t <= max_time + 1e-9:
        shapes.append(
            {
                "type": "line",
                "x0": t,
                "x1": t,
                "y0": 0,
                "y1": 1,
                "xref": "x",
                "yref": "paper",
                "line": {"color": "rgba(120,120,120,0.30)", "width": 1, "dash": "dot"},
            }
        )
        t += ten_min
    t = hour
    while t <= max_time + 1e-9:
        shapes.append(
            {
                "type": "line",
                "x0": t,
                "x1": t,
                "y0": 0,
                "y1": 1,
                "xref": "x",
                "yref": "paper",
                "line": {"color": "rgba(60,60,60,0.55)", "width": 2},
            }
        )
        t += hour
    return shapes


def create_static_plots(bundle: AnalysisBundle, output_directory: str) -> dict:
    plots_dir = Path(output_directory) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    failures = []

    arenas = list(bundle.info_data.get("arenas", {}).keys())
    stimuli = list(bundle.info_data.get("stimulus_areas", {}).keys())

    for source, binned_df in bundle.binned_by_source.items():
        source_dir = plots_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        framewise_df = bundle.framewise_by_source[source]

        for bee_id in bundle.bee_ids:
            try:
                bee_framewise = framewise_df[framewise_df["bee_id"] == bee_id].copy()
                bee_binned = binned_df[binned_df["BeeID"] == bee_id].copy()
                if bee_binned.empty:
                    continue

                subplot_specs = []
                if bundle.selected_metrics.get("speed", True):
                    subplot_specs.append("speed")
                if bundle.selected_metrics.get("dist_arena", True):
                    subplot_specs.append("dist_arena")
                if bundle.selected_metrics.get("stim_duration", True):
                    subplot_specs.append("stim_duration")
                if bundle.selected_metrics.get("stim_dist", True):
                    subplot_specs.append("stim_dist")
                if bundle.selected_metrics.get("stim_entries", True):
                    subplot_specs.append("stim_entries")

                if subplot_specs:
                    fig, axes = plt.subplots(len(subplot_specs), 1, figsize=(10, 3.0 * len(subplot_specs)), sharex=True)
                    if len(subplot_specs) == 1:
                        axes = [axes]

                    for ax, plot_type in zip(axes, subplot_specs):
                        x = bee_binned["Time_End_s"]
                        if plot_type == "speed":
                            ax.plot(x, bee_binned.get("AvgSpeed_mm_s", np.nan), marker="o")
                            ax.set_ylabel("mm/s")
                            ax.set_title("Average Speed")
                        elif plot_type == "dist_arena":
                            for arena_id in arenas:
                                col = f"AvgDistArenaCenter_{arena_id}_mm"
                                if col in bee_binned.columns and bee_binned[col].notna().any():
                                    ax.plot(x, bee_binned[col], marker="o", label=arena_id)
                            ax.set_ylabel("mm")
                            ax.set_title("Average Distance to Arena Center")
                            if len(ax.lines) > 1:
                                ax.legend(fontsize=8)
                        elif plot_type == "stim_duration":
                            for stim_id in stimuli:
                                col = f"Duration_in_Stim_{stim_id}_s"
                                if col in bee_binned.columns and bee_binned[col].notna().any():
                                    ax.plot(x, bee_binned[col], marker="o", label=stim_id)
                            ax.set_ylabel("s")
                            ax.set_title("Duration in Stimulus")
                            if len(ax.lines) > 1:
                                ax.legend(fontsize=8)
                        elif plot_type == "stim_dist":
                            for stim_id in stimuli:
                                col = f"AvgDistStimCenter_{stim_id}_mm"
                                if col in bee_binned.columns and bee_binned[col].notna().any():
                                    ax.plot(x, bee_binned[col], marker="o", label=stim_id)
                            ax.set_ylabel("mm")
                            ax.set_title("Average Distance to Stimulus Center")
                            if len(ax.lines) > 1:
                                ax.legend(fontsize=8)
                        elif plot_type == "stim_entries":
                            for stim_id in stimuli:
                                col1 = f"Entries_to_Stim_{stim_id}"
                                col2 = f"Exits_from_Stim_{stim_id}"
                                if col1 in bee_binned.columns and bee_binned[col1].notna().any():
                                    ax.plot(x, bee_binned[col1], marker="o", label=f"{stim_id} entries")
                                if col2 in bee_binned.columns and bee_binned[col2].notna().any():
                                    ax.plot(x, bee_binned[col2], marker="o", linestyle="--", label=f"{stim_id} exits")
                            ax.set_ylabel("count")
                            ax.set_title("Stimulus Entries / Exits")
                            if len(ax.lines) > 1:
                                ax.legend(fontsize=8)
                        ax.grid(True, linestyle=":", alpha=0.6)

                    axes[-1].set_xlabel("Time (s)")
                    fig.suptitle(f"{bundle.experiment_name} | {bee_id} | {source}")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                    plot_path = source_dir / f"{_sanitize_filename(bee_id)}_analysis_plots.png"
                    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
                    plt.close(fig)
                    saved[f"plot_{source}_{bee_id}"] = str(plot_path)

                if not bee_framewise.empty:
                    valid = bee_framewise[["x", "y"]].dropna()
                    if not valid.empty:
                        w = int(bundle.info_data.get("frame_width", 1920))
                        h = int(bundle.info_data.get("frame_height", 1080))
                        bins_x = max(10, min(200, int(w / 20)))
                        bins_y = max(10, min(200, int(h / 20)))
                        hist, _, _ = np.histogram2d(valid["x"], valid["y"], bins=[bins_x, bins_y], range=[[0, w], [0, h]])
                        hist = hist.T.astype(np.float32)
                        hist = cv2.GaussianBlur(hist, (11, 11), 0)
                        hist = np.sqrt(hist)
                        hist = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        heatmap = cv2.applyColorMap(hist, cv2.COLORMAP_JET)
                        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
                        for arena in bundle.info_data.get("arenas", {}).values():
                            draw_shape_outline(heatmap, arena, (255, 255, 255), 3)
                        for stim in bundle.info_data.get("stimulus_areas", {}).values():
                            draw_shape_outline(heatmap, stim, (255, 255, 255), 2)
                        heatmap_path = source_dir / f"{_sanitize_filename(bee_id)}_heatmap.png"
                        if cv2.imwrite(str(heatmap_path), heatmap):
                            saved[f"heatmap_{source}_{bee_id}"] = str(heatmap_path)
                        else:
                            failures.append(f"Failed to write heatmap for {bee_id} ({source})")
            except Exception as exc:
                failures.append(f"{bee_id} ({source}): {exc}")
            finally:
                plt.close("all")

    if failures:
        failure_path = plots_dir / "plot_failures.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        saved["failures_log"] = str(failure_path)
    return saved


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _df_to_records_json(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "[]"
    export_df = df.copy()
    export_df = export_df.replace([np.inf, -np.inf], np.nan)
    return export_df.to_json(orient="records")


def export_html_report(bundle: AnalysisBundle, output_directory: str) -> dict:
    report_dir = Path(output_directory) / "report"
    data_dir = report_dir / "data"
    report_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    css_path = report_dir / "report.css"
    html_path = report_dir / "report.html"
    js_path = report_dir / "report.js"
    data_js_path = report_dir / "report_data.js"
    plotly_path = report_dir / "plotly.min.js"

    summary = {
        "fps": bundle.info_data.get("fps"),
        "pixel_to_mm": bundle.info_data.get("pixel_to_mm"),
        "frame_width": bundle.info_data.get("frame_width"),
        "frame_height": bundle.info_data.get("frame_height"),
        "start_time_str": bundle.info_data.get("start_time_str"),
        "end_time_str": bundle.info_data.get("end_time_str"),
        "bee_count": len(bundle.bee_ids),
        "bee_ids": bundle.bee_ids,
        "bin_seconds": bundle.bin_seconds,
        "available_sources": sorted(bundle.framewise_by_source.keys()),
        "source_paths": bundle.source_paths,
        "intersections": shape_intersections(bundle.info_data),
        "default_bee": bundle.bee_ids[0] if bundle.bee_ids else None,
    }

    payload = {
        "summary": summary,
        "arenas": bundle.info_data.get("arenas", {}),
        "stimuli": bundle.info_data.get("stimulus_areas", {}),
        "selected_metrics": bundle.selected_metrics,
        "chunk_index": {"compare": {}, "framewise": {}, "binned": {}},
    }

    def write_chunk(kind: str, source: str, bee_id: str, df: pd.DataFrame) -> None:
        safe_bee = _sanitize_filename(bee_id)
        rel_path = f"data/{kind}_{source}_{safe_bee}.js"
        abs_path = report_dir / rel_path
        export_df = df.replace([np.inf, -np.inf], np.nan)
        js_name = f"{kind}|{source}|{bee_id}"
        abs_path.write_text(
            "window.REPORT_CHUNKS = window.REPORT_CHUNKS || {};\n"
            + f"window.REPORT_CHUNKS[{json.dumps(js_name)}] = {export_df.to_json(orient='records')};\n",
            encoding="utf-8",
        )
        payload["chunk_index"][kind].setdefault(source, {})[bee_id] = rel_path

    for bee_id, bee_df in bundle.framewise_compare_df.groupby("bee_id", sort=False):
        write_chunk("compare", "compare", bee_id, bee_df)
    for source, df in bundle.framewise_by_source.items():
        for bee_id, bee_df in df.groupby("bee_id", sort=False):
            write_chunk("framewise", source, bee_id, bee_df)
    for source, df in bundle.binned_by_source.items():
        for bee_id, bee_df in df.groupby("BeeID", sort=False):
            write_chunk("binned", source, bee_id, bee_df)

    payload_js = "window.REPORT_META = " + json.dumps(payload, ensure_ascii=False) + ";\nwindow.REPORT_CHUNKS = window.REPORT_CHUNKS || {};\n"

    css_text = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; font-family:Segoe UI, Arial, sans-serif; background:#111827; color:#e5e7eb; }
header { padding:20px 24px; background:#0f172a; border-bottom:1px solid #1f2937; }
main { padding:20px 24px; width:100%; }
.controls { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; align-items:end; }
.panel { background:#1f2937; border:1px solid #374151; border-radius:14px; padding:16px; margin-bottom:18px; width:100%; }
.panel h3 { margin:0 0 12px 0; }
.grid { display:flex; flex-direction:column; gap:18px; width:100%; }
.grid > .panel { margin-bottom:0; }
label { display:flex; flex-direction:column; gap:6px; font-size:14px; min-width:220px; }
select { background:#111827; color:#e5e7eb; border:1px solid #4b5563; border-radius:8px; padding:8px 10px; }
#summaryCards { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; }
.card { background:#111827; border:1px solid #374151; border-radius:12px; padding:12px; }
.metric-label { color:#93c5fd; font-size:13px; }
.metric-value { font-size:26px; font-weight:700; margin-top:6px; }
.info-row { display:flex; gap:8px; flex-wrap:wrap; }
.info-pill { background:#111827; border:1px solid #374151; border-radius:999px; padding:8px 12px; font-size:13px; }
.chart { min-height:520px; width:100%; }
#status { font-size:13px; color:#93c5fd; }
@media (max-width: 900px) {
  main { padding:16px; }
  .chart { min-height:400px; }
  label { min-width:160px; }
}
"""

    html_text = f"""<!doctype html>
<html><head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{bundle.experiment_name} report</title>
  <link rel=\"stylesheet\" href=\"report.css\" />
</head>
<body>
<header>
  <h1 style=\"margin:0 0 8px 0;\">{bundle.experiment_name} tracking report</h1>
  <div id=\"status\">Offline report. Data for each bee loads on demand.</div>
</header>
<main>
  <section class=\"controls panel\">
    <label>Bee<select id=\"beeSelect\"></select></label>
    <label>Mode<select id=\"modeSelect\"><option value=\"raw\">Raw</option><option value=\"filtered\">Filtered</option><option value=\"compare\">Compare Raw vs Filtered</option></select></label>
  </section>
  <section class=\"panel\"><div id=\"summaryCards\"></div></section>
  <section class=\"panel\"><div id=\"shapeList\" class=\"info-row\"></div></section>
  <section class=\"grid\">
    <div class=\"panel\"><h3>Trajectory</h3><div id=\"trajectoryChart\" class=\"chart\"></div></div>
    <div class=\"panel\"><h3>Raw vs Filtered</h3><div id=\"compareChart\" class=\"chart\"></div></div>
    <div class=\"panel\"><h3>Position</h3><div id=\"positionChart\" class=\"chart\"></div></div>
    <div class=\"panel\"><h3>Angle</h3><div id=\"angleChart\" class=\"chart\"></div></div>
    <div class=\"panel\"><h3>Speed</h3><div id=\"speedChart\" class=\"chart\"></div></div>
    <div class=\"panel\"><h3>Inside / State Timeline</h3><div id=\"stateChart\" class=\"chart\"></div></div>
    <div class=\"panel\"><h3>Distances</h3><div id=\"distanceChart\" class=\"chart\"></div></div>
    <div class=\"panel\"><h3>Binned Metrics</h3><div id=\"binnedChart\" class=\"chart\"></div></div>
  </section>
</main>
<script src=\"plotly.min.js\"></script>
<script src=\"report_data.js\"></script>
<script src=\"report.js\"></script>
</body></html>"""

    report_js = """
const payload = window.REPORT_META;
const beeSelect = document.getElementById('beeSelect');
const modeSelect = document.getElementById('modeSelect');
const summaryCards = document.getElementById('summaryCards');
const shapeList = document.getElementById('shapeList');
const statusEl = document.getElementById('status');

function pathFor(kind, source, beeId){ return payload.chunk_index?.[kind]?.[source]?.[beeId] || null; }
function keyFor(kind, source, beeId){ return `${kind}|${source}|${beeId}`; }
function loadScript(relPath){
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-rel=\"${relPath}\"]`);
    if (existing) { resolve(); return; }
    const s = document.createElement('script');
    s.src = relPath;
    s.dataset.rel = relPath;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`Failed to load ${relPath}`));
    document.body.appendChild(s);
  });
}
async function ensureChunk(kind, source, beeId){
  const key = keyFor(kind, source, beeId);
  if (window.REPORT_CHUNKS[key]) return window.REPORT_CHUNKS[key];
  const relPath = pathFor(kind, source, beeId);
  if (!relPath) return [];
  await loadScript(relPath);
  return window.REPORT_CHUNKS[key] || [];
}
function rulerShapes(maxTime){
  const shapes = [];
  for(let t=600; t<=maxTime+1e-9; t+=600){ shapes.push({type:'line', x0:t, x1:t, y0:0, y1:1, xref:'x', yref:'paper', line:{color:'rgba(120,120,120,0.30)', width:1, dash:'dot'}}); }
  for(let t=3600; t<=maxTime+1e-9; t+=3600){ shapes.push({type:'line', x0:t, x1:t, y0:0, y1:1, xref:'x', yref:'paper', line:{color:'rgba(60,60,60,0.55)', width:2}}); }
  return shapes;
}
function arenaStimTraces(){
  const shapes = [];
  Object.values(payload.arenas || {}).forEach(shape => {
    if(shape.shape === 'rect'){ shapes.push({type:'rect', x0:shape.geom[0], y0:shape.geom[1], x1:shape.geom[0]+shape.geom[2], y1:shape.geom[1]+shape.geom[3], line:{color:'rgba(255,255,255,0.8)', width:2}}); }
    else if(shape.shape === 'circle'){ shapes.push({type:'circle', x0:shape.geom[0]-shape.geom[2], y0:shape.geom[1]-shape.geom[2], x1:shape.geom[0]+shape.geom[2], y1:shape.geom[1]+shape.geom[2], line:{color:'rgba(255,255,255,0.8)', width:2}}); }
  });
  Object.values(payload.stimuli || {}).forEach(shape => {
    if(shape.shape === 'rect'){ shapes.push({type:'rect', x0:shape.geom[0], y0:shape.geom[1], x1:shape.geom[0]+shape.geom[2], y1:shape.geom[1]+shape.geom[3], line:{color:'rgba(255,170,0,0.9)', width:2}}); }
    else if(shape.shape === 'circle'){ shapes.push({type:'circle', x0:shape.geom[0]-shape.geom[2], y0:shape.geom[1]-shape.geom[2], x1:shape.geom[0]+shape.geom[2], y1:shape.geom[1]+shape.geom[2], line:{color:'rgba(255,170,0,0.9)', width:2}}); }
  });
  return shapes;
}
function initControls(){
  payload.summary.bee_ids.forEach(id => {
    const opt = document.createElement('option'); opt.value = id; opt.textContent = id; beeSelect.appendChild(opt);
  });
  beeSelect.value = payload.summary.default_bee || payload.summary.bee_ids[0] || '';
  if (!payload.summary.available_sources.includes('filtered')) modeSelect.value = 'raw';
}
function renderStaticSummary(compare){
  const total = compare.length;
  const detected = compare.filter(r => r.state === 'detected').length;
  const predicted = compare.filter(r => r.state === 'predicted').length;
  const missing = compare.filter(r => r.state === 'missing').length;
  summaryCards.innerHTML = [ ['Frames', total], ['Detected', detected], ['Predicted', predicted], ['Missing', missing], ['FPS', payload.summary.fps], ['Scale (mm/px)', payload.summary.pixel_to_mm] ].map(([label, value]) => `<div class=\"card\"><div class=\"metric-label\">${label}</div><div class=\"metric-value\">${value}</div></div>`).join('');
  const pills = [];
  Object.values(payload.arenas || {}).forEach(a => pills.push(`<div class=\"info-pill\"><strong>${a.id}</strong> arena</div>`));
  Object.values(payload.stimuli || {}).forEach(s => pills.push(`<div class=\"info-pill\"><strong>${s.id}</strong> stimulus</div>`));
  shapeList.innerHTML = pills.join('');
}
function transparentLayout(extra){ return Object.assign({paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:'#e5e7eb'}, margin:{l:50,r:20,t:20,b:45}}, extra||{}); }
async function refresh(){
  const beeId = beeSelect.value;
  const mode = modeSelect.value;
  statusEl.textContent = `Loading ${beeId} (${mode})...`;
  const compare = await ensureChunk('compare', 'compare', beeId);
  let primary = [];
  let raw = [];
  let filtered = [];
  if (mode === 'compare') {
    if (payload.summary.available_sources.includes('raw')) raw = await ensureChunk('framewise', 'raw', beeId);
    if (payload.summary.available_sources.includes('filtered')) filtered = await ensureChunk('framewise', 'filtered', beeId);
    primary = filtered.length ? filtered : raw;
  } else {
    primary = await ensureChunk('framewise', mode, beeId);
  }
  const binnedSource = mode === 'compare' ? (payload.summary.available_sources.includes('filtered') ? 'filtered' : 'raw') : mode;
  const binned = await ensureChunk('binned', binnedSource, beeId);
  renderStaticSummary(compare);
  statusEl.textContent = `Loaded ${beeId} (${mode}).`;
  const maxPrimary = primary.length ? primary[primary.length-1].time_s : 0;
  const maxCompare = compare.length ? compare[compare.length-1].time_s : 0;
  const traj = [];
  if (mode === 'compare') { if (raw.length) traj.push({x:raw.map(r=>r.x), y:raw.map(r=>r.y), mode:'lines', name:'Raw'}); if (filtered.length) traj.push({x:filtered.map(r=>r.x), y:filtered.map(r=>r.y), mode:'lines', name:'Filtered', line:{dash:'dot'}}); }
  else { traj.push({x:primary.map(r=>r.x), y:primary.map(r=>r.y), mode:'lines', name:mode}); }
  Plotly.newPlot('trajectoryChart', traj, transparentLayout({xaxis:{title:'X (px)', range:[0, payload.summary.frame_width]}, yaxis:{title:'Y (px)', range:[payload.summary.frame_height,0]}, shapes:arenaStimTraces()}), {responsive:true});
  const stateMap = {missing:0, predicted:1, detected:2};
  Plotly.newPlot('compareChart', [{x:compare.map(r=>r.time_s), y:compare.map(r=>r.raw_x), mode:'lines', name:'Raw X'}, {x:compare.map(r=>r.time_s), y:compare.map(r=>r.filtered_x), mode:'lines', name:'Filtered X', line:{dash:'dot'}}, {x:compare.map(r=>r.time_s), y:compare.map(r=>stateMap[r.state]), mode:'lines', name:'State', yaxis:'y2'}], transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'X position (px)'}, yaxis2:{title:'State', overlaying:'y', side:'right', tickvals:[0,1,2], ticktext:['missing','predicted','detected']}, shapes:rulerShapes(maxCompare)}), {responsive:true});
  Plotly.newPlot('positionChart', [{x:primary.map(r=>r.time_s), y:primary.map(r=>r.x), mode:'lines', name:'X'}, {x:primary.map(r=>r.time_s), y:primary.map(r=>r.y), mode:'lines', name:'Y'}], transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Position (px)'}, shapes:rulerShapes(maxPrimary)}), {responsive:true});
  Plotly.newPlot('angleChart', [{x:primary.map(r=>r.time_s), y:primary.map(r=>r.ang), mode:'lines', name:'Angle'}, {x:primary.map(r=>r.time_s), y:primary.map(r=>r.angle_change_deg), mode:'lines', name:'Angle change', yaxis:'y2'}], transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Angle (deg)'}, yaxis2:{title:'Δ angle (deg)', overlaying:'y', side:'right'}, shapes:rulerShapes(maxPrimary)}), {responsive:true});
  Plotly.newPlot('speedChart', [{x:primary.map(r=>r.time_s), y:primary.map(r=>r.speed_mm_s), mode:'lines', name:'Speed'}], transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Speed (mm/s)'}, shapes:rulerShapes(maxPrimary)}), {responsive:true});
  const insideCols = primary.length ? Object.keys(primary[0]).filter(k => k.startsWith('inside_')) : [];
  const stateTraces = insideCols.map(col => ({x:primary.map(r=>r.time_s), y:primary.map(r=>r[col]), mode:'lines', name:col, line:{shape:'hv'}}));
  stateTraces.push({x:compare.map(r=>r.time_s), y:compare.map(r=>stateMap[r.state]), mode:'lines', name:'state', yaxis:'y2', line:{dash:'dot'}});
  Plotly.newPlot('stateChart', stateTraces, transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Inside (0/1)', range:[-0.2,1.2]}, yaxis2:{title:'State', overlaying:'y', side:'right', tickvals:[0,1,2], ticktext:['missing','predicted','detected']}, shapes:rulerShapes(maxCompare)}), {responsive:true});
  const distCols = primary.length ? Object.keys(primary[0]).filter(k => k.startsWith('dist_to_')) : [];
  Plotly.newPlot('distanceChart', distCols.map(col => ({x:primary.map(r=>r.time_s), y:primary.map(r=>r[col]), mode:'lines', name:col})), transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Distance (mm)'}, shapes:rulerShapes(maxPrimary)}), {responsive:true});
  const time = binned.map(r => r.Time_End_s);
  const binnedTraces = [];
  if (payload.selected_metrics.speed && binned.length && binned[0].AvgSpeed_mm_s !== undefined) binnedTraces.push({x:time, y:binned.map(r=>r.AvgSpeed_mm_s), mode:'lines+markers', name:'AvgSpeed_mm_s'});
  if (binned.length) Object.keys(binned[0]).forEach(key => { if (key.startsWith('AvgDistArenaCenter_') || key.startsWith('Duration_in_Stim_') || key.startsWith('AvgDistStimCenter_') || key.startsWith('Entries_to_Stim_') || key.startsWith('Exits_from_Stim_')) { const trace = {x:time, y:binned.map(r=>r[key]), mode:'lines+markers', name:key}; if (key.startsWith('Exits_from_Stim_')) trace.line = {dash:'dot'}; binnedTraces.push(trace); } });
  Plotly.newPlot('binnedChart', binnedTraces, transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Binned value'}, shapes:rulerShapes(time.length ? time[time.length-1] : 0)}), {responsive:true});
}
initControls(); beeSelect.addEventListener('change', refresh); modeSelect.addEventListener('change', refresh); refresh();
"""

    css_path.write_text(css_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    js_path.write_text(report_js, encoding="utf-8")
    data_js_path.write_text(payload_js, encoding="utf-8")
    plotly_path.write_text(plotly.offline.get_plotlyjs(), encoding="utf-8")
    return {
        "html_report": str(html_path),
        "report_css": str(css_path),
        "report_js": str(js_path),
        "report_data_js": str(data_js_path),
        "plotly_js": str(plotly_path),
        "report_data_dir": str(data_dir),
    }
