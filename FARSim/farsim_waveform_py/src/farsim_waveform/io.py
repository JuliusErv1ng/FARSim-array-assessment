from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .synthesizer import WaveformResult


def save_results_npz(path: str | Path, results: Sequence[WaveformResult]) -> Path:
    """Save synthesized traces to a compact NumPy archive."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        raise ValueError("No waveform results to save.")
    station_ids = np.array([str(result.station_id) for result in results], dtype=object)
    traces = np.stack([result.trace_uv_m for result in results], axis=0)
    metadata = np.array([json.dumps(result.metadata, ensure_ascii=False) for result in results], dtype=object)
    np.savez_compressed(path, t_ns=results[0].t_ns, station_ids=station_ids, traces_uv_m=traces, metadata=metadata)
    return path


def save_rfchain_long_csv(path: str | Path, results: Sequence[WaveformResult]) -> Path:
    """Save traces as a long table: station_id, t_ns, Ex, Ey, Ez."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["station_id", "t_ns", "Ex_uV_m", "Ey_uV_m", "Ez_uV_m"])
        for result in results:
            for t_ns, (ex, ey, ez) in zip(result.t_ns, result.trace_uv_m, strict=True):
                writer.writerow([result.station_id, f"{t_ns:.6g}", f"{ex:.9g}", f"{ey:.9g}", f"{ez:.9g}"])
    return path
