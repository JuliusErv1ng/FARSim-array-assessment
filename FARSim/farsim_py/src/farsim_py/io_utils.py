from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


def load_mat_file(path: Path) -> dict[str, Any]:
    """Load a MATLAB file, including simple v7.3 fallback support."""
    try:
        data = loadmat(path, squeeze_me=True, simplify_cells=True)
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except NotImplementedError:
        return _load_hdf5_mat(path)
    except ValueError as exc:
        if "Unknown mat file type" not in str(exc):
            raise
        return _load_hdf5_mat(path)


def _load_hdf5_mat(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            out[key] = np.array(handle[key]).T
    return out


def read_layout_positions(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        frame = pd.read_excel(path, header=None)
        numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        pos = numeric.to_numpy(dtype=float)
    else:
        rows: list[list[float]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) >= 2:
                rows.append(values)
        pos = np.asarray(rows, dtype=float)
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError(f"Layout file {path} must contain at least two columns.")
    return pos[:, :2]


def read_flux_curve(path: Path) -> np.ndarray:
    frame = pd.read_excel(path, header=None)
    return frame.to_numpy(dtype=float)


def read_numeric_table(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float)


def save_mat_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, payload, do_compression=True)


def format_threshold_label(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}uV"
    return f"{value}".replace("-", "m").replace(".", "p") + "uV"


def build_output_path(layout_path: Path, output_root: Path, threshold_uv_m: float, num_antennas: int) -> Path:
    layout_stem = layout_path.stem
    threshold_label = format_threshold_label(threshold_uv_m)
    filename = f"{layout_stem}_thr{threshold_label}_ant{num_antennas}_recons.mat"
    return output_root / layout_stem / filename
