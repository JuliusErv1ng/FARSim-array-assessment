from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d

from .io_utils import read_layout_positions


DEFAULT_DATA_ROOT = Path("/Volumes/T7/FARSim/from mapx/StshpOutbox_XDS")


@dataclass(slots=True)
class FastRateConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    layout: Path = DEFAULT_DATA_ROOT / "layout" / "gp80" / "elliptic3.txt"
    trigger_library: Path = DEFAULT_DATA_ROOT / "threshold75uV_1118.mat"
    threshold_uv_m: float = 75.0
    required_antennas: int = 5
    num_core_samples: int = 5000
    trigger_radius_m: float = 750.0
    seed: int = 12345
    energy_start: float = 16.0
    energy_stop: float = 18.0
    energy_step: float = 0.1
    theta_start: int = 45
    theta_stop: int = 89
    phi_start: int = 0
    phi_stop: int = 350
    phi_step: int = 10
    output_dir: Path = Path("/Volumes/T7/FARSim/farsim_py/outputs/gp80_fast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast FARSim field-threshold event-rate calculator.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--layout", type=Path, default=DEFAULT_DATA_ROOT / "layout" / "gp80" / "elliptic3.txt")
    parser.add_argument("--trigger-library", type=Path, default=DEFAULT_DATA_ROOT / "threshold75uV_1118.mat")
    parser.add_argument("--threshold-uvm", type=float, default=75.0)
    parser.add_argument("--required-antennas", type=int, default=5)
    parser.add_argument("--num-core-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=Path("/Volumes/T7/FARSim/farsim_py/outputs/gp80_fast"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FastRateConfig(
        data_root=args.data_root,
        layout=args.layout,
        trigger_library=args.trigger_library,
        threshold_uv_m=args.threshold_uvm,
        required_antennas=args.required_antennas,
        num_core_samples=args.num_core_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    result = run_fast_event_rate(config)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(result["mat_path"])
    print(result["csv_path"])


def run_fast_event_rate(config: FastRateConfig) -> dict[str, object]:
    start = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    positions_xy = read_layout_positions(config.layout).astype(np.float64)
    trigger_library = loadmat(config.trigger_library, squeeze_me=True, simplify_cells=True)
    trigger_regions = np.asarray(trigger_library["trigger_region"], dtype=object)

    energy_values = _inclusive_float_grid(config.energy_start, config.energy_stop, config.energy_step)
    theta_values = np.arange(config.theta_start, config.theta_stop + 1, dtype=int)
    phi_values = np.arange(config.phi_start, config.phi_stop + 1, config.phi_step, dtype=int)

    yita = np.zeros((len(energy_values), len(theta_values), len(phi_values)), dtype=np.float64)
    fire_num = np.zeros_like(yita)
    fire_num_triggered = np.zeros_like(yita)

    rng = np.random.default_rng(config.seed)
    sampling_radius = config.trigger_radius_m + float(np.max(np.linalg.norm(positions_xy, axis=1)))

    # Compile the Numba kernels before timing the full loop.
    warm_centers = _matlab_like_core_positions(sampling_radius, 8, rng)
    warm_poly = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
    _trigger_stats_triggered_mean_numba(positions_xy, warm_poly, warm_centers, config.required_antennas)

    evaluated = 0
    nonempty = 0
    for energy_idx, _energy in enumerate(energy_values):
        for theta_idx, _theta in enumerate(theta_values):
            for phi_eval_idx, phi in enumerate(phi_values):
                evaluated += 1
                polygon = _coerce_polygon(trigger_regions[energy_idx, theta_idx, int(phi)])
                if polygon.size == 0:
                    continue
                nonempty += 1
                centers = _matlab_like_core_positions(sampling_radius, config.num_core_samples, rng)
                efficiency, mean_fired, mean_fired_triggered = _trigger_stats_triggered_mean_numba(
                    positions_xy,
                    polygon,
                    centers,
                    config.required_antennas,
                )
                yita[energy_idx, theta_idx, phi_eval_idx] = efficiency
                fire_num[energy_idx, theta_idx, phi_eval_idx] = mean_fired
                fire_num_triggered[energy_idx, theta_idx, phi_eval_idx] = mean_fired_triggered

    trg_rate = compute_event_rate(config, positions_xy, yita, energy_values, theta_values, phi_values)
    yita_legacy, fire_num_legacy, trg_rate_legacy = _legacy_phi_arrays(yita, fire_num, trg_rate, phi_values)
    _, fire_num_triggered_legacy, _ = _legacy_phi_arrays(yita, fire_num_triggered, trg_rate, phi_values)

    elapsed_s = time.perf_counter() - start
    layout_stem = config.layout.stem
    out_stem = f"{layout_stem}_thr{config.threshold_uv_m:g}uVm_ant{config.required_antennas}_core{config.num_core_samples}_fast"
    mat_path = config.output_dir / f"{out_stem}_recons.mat"
    csv_path = config.output_dir / f"{out_stem}_summary.csv"

    summary = {
        "layout": layout_stem,
        "threshold_uv_m": float(config.threshold_uv_m),
        "required_antennas": int(config.required_antennas),
        "num_core_samples": int(config.num_core_samples),
        "sampling_radius_m": float(sampling_radius),
        "total_daily_rate": float(np.nansum(trg_rate)),
        "mean_efficiency": float(np.mean(yita)),
        "mean_fired_antennas": float(np.mean(fire_num)),
        "mean_fired_antennas_triggered": float(np.nanmean(fire_num_triggered[fire_num_triggered > 0]))
        if np.any(fire_num_triggered > 0)
        else 0.0,
        "evaluated_bins": int(evaluated),
        "nonempty_trigger_bins": int(nonempty),
        "elapsed_s": float(elapsed_s),
        "seed": int(config.seed),
        "trigger_library": str(config.trigger_library),
    }

    payload = {
        "trg_rate": trg_rate_legacy,
        "yita": yita_legacy,
        "fire_num": fire_num_legacy,
        "fire_num_triggered": fire_num_triggered_legacy,
        "trg_rate_compact": trg_rate,
        "yita_compact": yita,
        "fire_num_compact": fire_num,
        "fire_num_triggered_compact": fire_num_triggered,
        "energy_values": energy_values,
        "theta_values": theta_values,
        "phi_values": phi_values,
        "summary_json": np.array([json.dumps(summary, ensure_ascii=False)], dtype=object),
        "config_json": np.array([json.dumps(_config_for_json(config), ensure_ascii=False)], dtype=object),
    }
    savemat(mat_path, payload, do_compression=True)
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    return {"mat_path": str(mat_path), "csv_path": str(csv_path), "summary": summary}


def compute_event_rate(
    config: FastRateConfig,
    positions_xy: np.ndarray,
    yita: np.ndarray,
    energy_values: np.ndarray,
    theta_values: np.ndarray,
    phi_values: np.ndarray,
) -> np.ndarray:
    flux = pd.read_excel(config.data_root / "Flux.xlsx", header=None).to_numpy(dtype=np.float64)
    flux[:, 1] *= 1.0e24
    xi = np.round(np.arange(15.33, 18.25 + 1.0e-9, 0.01), 2)
    yi = interp1d(flux[:, 0], flux[:, 1], kind="cubic", fill_value="extrapolate")(xi)
    xi_to_idx = {float(value): idx for idx, value in enumerate(xi)}

    r_d = float(np.max(np.linalg.norm(positions_xy, axis=1)))
    dd = config.trigger_radius_m + r_d
    trg_rate = np.zeros_like(yita)

    for energy_idx, energy in enumerate(energy_values):
        energy_low = round(float(energy), 2)
        energy_high = round(float(energy + config.energy_step), 2)
        low_idx = xi_to_idx[energy_low]
        high_idx = xi_to_idx[energy_high]
        flux_factor = (
            yi[high_idx] / (10.0**energy_high) ** 3
            + yi[low_idx] / (10.0**energy_low) ** 3
        ) * (10.0**energy_high - 10.0**energy_low) * 0.5

        for theta_idx, theta in enumerate(theta_values):
            area = np.pi * dd**2 * np.cos(np.deg2rad(theta))
            sr = (np.cos(np.deg2rad(theta)) - np.cos(np.deg2rad(theta + 1))) * np.pi / 180.0
            n_events = flux_factor * area * 86400.0 * sr
            for phi_eval_idx, phi in enumerate(phi_values):
                trg_rate[energy_idx, theta_idx, phi_eval_idx] = (
                    yita[energy_idx, theta_idx, phi_eval_idx] * n_events * config.phi_step
                )
    return trg_rate


def _matlab_like_core_positions(radius_m: float, count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw core positions uniformly in area inside the sampling disk.

    The original MATLAB-style implementation sampled x uniformly and then y
    uniformly along the chord.  That does not produce an area-uniform disk
    distribution, while the aperture calculation below multiplies the measured
    efficiency by pi R^2.  Keep the historical private function name so older
    driver scripts pick up the corrected sampling automatically.
    """
    radius = float(radius_m) * np.sqrt(rng.random(count))
    angle = 2.0 * np.pi * rng.random(count)
    center_x = radius * np.cos(angle)
    center_y = radius * np.sin(angle)
    return np.column_stack((center_x, center_y)).astype(np.float64)


def _coerce_polygon(value: object) -> np.ndarray:
    polygon = np.asarray(value, dtype=np.float64)
    if polygon.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    polygon = np.atleast_2d(polygon)
    if polygon.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    polygon = polygon[:, :2]
    finite = np.isfinite(polygon).all(axis=1)
    polygon = polygon[finite]
    if len(polygon) < 3:
        return np.empty((0, 2), dtype=np.float64)
    return np.ascontiguousarray(polygon, dtype=np.float64)


def _legacy_phi_arrays(
    yita: np.ndarray,
    fire_num: np.ndarray,
    trg_rate: np.ndarray,
    phi_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    legacy_shape = (yita.shape[0], yita.shape[1], int(phi_values.max()) + 1)
    yita_legacy = np.zeros(legacy_shape, dtype=np.float64)
    fire_num_legacy = np.zeros(legacy_shape, dtype=np.float64)
    trg_rate_legacy = np.zeros(legacy_shape, dtype=np.float64)
    for compact_idx, phi in enumerate(phi_values):
        yita_legacy[:, :, int(phi)] = yita[:, :, compact_idx]
        fire_num_legacy[:, :, int(phi)] = fire_num[:, :, compact_idx]
        trg_rate_legacy[:, :, int(phi)] = trg_rate[:, :, compact_idx]
    return yita_legacy, fire_num_legacy, trg_rate_legacy


def _inclusive_float_grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step)) + 1
    return np.round(start + step * np.arange(count, dtype=np.float64), 10)


def _config_for_json(config: FastRateConfig) -> dict[str, object]:
    out = asdict(config)
    for key, value in list(out.items()):
        if isinstance(value, Path):
            out[key] = str(value)
    return out


@njit(fastmath=True, cache=True)
def _trigger_stats_numba(
    positions_xy: np.ndarray,
    polygon_xy: np.ndarray,
    centers_xy: np.ndarray,
    required_antennas: int,
) -> tuple[float, float]:
    n_centers = centers_xy.shape[0]
    n_antennas = positions_xy.shape[0]
    xmin = np.min(polygon_xy[:, 0])
    xmax = np.max(polygon_xy[:, 0])
    ymin = np.min(polygon_xy[:, 1])
    ymax = np.max(polygon_xy[:, 1])

    triggered = 0
    fired_sum = 0.0
    for center_idx in range(n_centers):
        cx = centers_xy[center_idx, 0]
        cy = centers_xy[center_idx, 1]
        fired = 0
        for ant_idx in range(n_antennas):
            x = positions_xy[ant_idx, 0] + cx
            y = positions_xy[ant_idx, 1] + cy
            if x < xmin or x > xmax or y < ymin or y > ymax:
                continue
            if _point_in_polygon_numba(x, y, polygon_xy):
                fired += 1
        fired_sum += fired
        if fired >= required_antennas:
            triggered += 1
    return triggered / n_centers, fired_sum / n_centers


@njit(fastmath=True, cache=True)
def _trigger_stats_triggered_mean_numba(
    positions_xy: np.ndarray,
    polygon_xy: np.ndarray,
    centers_xy: np.ndarray,
    required_antennas: int,
) -> tuple[float, float, float]:
    n_centers = centers_xy.shape[0]
    n_antennas = positions_xy.shape[0]
    xmin = np.min(polygon_xy[:, 0])
    xmax = np.max(polygon_xy[:, 0])
    ymin = np.min(polygon_xy[:, 1])
    ymax = np.max(polygon_xy[:, 1])

    triggered = 0
    fired_sum = 0.0
    fired_sum_triggered = 0.0
    for center_idx in range(n_centers):
        cx = centers_xy[center_idx, 0]
        cy = centers_xy[center_idx, 1]
        fired = 0
        for ant_idx in range(n_antennas):
            x = positions_xy[ant_idx, 0] + cx
            y = positions_xy[ant_idx, 1] + cy
            if x < xmin or x > xmax or y < ymin or y > ymax:
                continue
            if _point_in_polygon_numba(x, y, polygon_xy):
                fired += 1
        fired_sum += fired
        if fired >= required_antennas:
            triggered += 1
            fired_sum_triggered += fired
    efficiency = triggered / n_centers
    mean_fired = fired_sum / n_centers
    mean_fired_triggered = fired_sum_triggered / triggered if triggered > 0 else 0.0
    return efficiency, mean_fired, mean_fired_triggered


@njit(fastmath=True, cache=True)
def _point_in_polygon_numba(x: float, y: float, polygon_xy: np.ndarray) -> bool:
    inside = False
    n_vertices = polygon_xy.shape[0]
    j = n_vertices - 1
    for i in range(n_vertices):
        xi = polygon_xy[i, 0]
        yi = polygon_xy[i, 1]
        xj = polygon_xy[j, 0]
        yj = polygon_xy[j, 1]
        if (yi > y) != (yj > y):
            x_intersection = (xj - xi) * (y - yi) / (yj - yi + 1.0e-300) + xi
            if x < x_intersection:
                inside = not inside
        j = i
    return inside


if __name__ == "__main__":
    main()
