from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
from numba import njit

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
from scipy.io import savemat
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from farsim_py.fast_event_rate import (
    FastRateConfig,
    _matlab_like_core_positions,
    _trigger_stats_triggered_mean_numba,
    compute_event_rate,
)
from farsim_py.io_utils import read_layout_positions
from farsim_py.physics import largest_contour_polygon
from farsim_waveform import FARSimAPI, adc_filter_metadata, apply_adc_iir_fir, build_adc_noise_library, default_l1_config
from farsim_waveform.voltage_trigger import aera_voltage_trigger


FARSIM_ROOT = Path(os.environ.get("FARSIM_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = Path(os.environ.get("FARSIM_DATA_ROOT", FARSIM_ROOT / "data" / "StshpOutbox_XDS"))
DEFAULT_LAYOUT_PATH = DATA_ROOT / "layout" / "gp65" / "gp65_centered_mean.txt"
OUTPUT_DIR = FARSIM_ROOT / "farsim_waveform_py" / "outputs" / "gp65_fullbin_adaptive_modes"
REFERENCE_EEV = 0.316

_WORKER: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GP65 full-bin adaptive voltage-threshold and noisy-L1 scans.")
    parser.add_argument("--coarse-grid-size", type=int, default=9)
    parser.add_argument("--grid-extent-m", type=float, default=8000.0)
    parser.add_argument("--refine-factor", type=int, default=3)
    parser.add_argument("--max-refine-points", type=int, default=160)
    parser.add_argument("--target-cell-size-m", type=float, default=250.0)
    parser.add_argument("--max-safe-sample-points", type=int, default=12000)
    parser.add_argument("--legacy-regular-contours", action="store_true")
    parser.add_argument("--l1-probability-refine", action="store_true")
    parser.add_argument("--voltage-band-fraction", type=float, default=0.30)
    parser.add_argument("--l1-boundary-level", type=float, default=0.5)
    parser.add_argument("--l1-boundary-band", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--core-samples", type=int, default=5000)
    parser.add_argument("--core-margin-m", type=float, default=750.0)
    parser.add_argument("--threshold-uv", type=float, default=5000.0)
    parser.add_argument("--required-antennas", type=int, default=5)
    parser.add_argument("--noise-library-size", type=int, default=8)
    parser.add_argument("--noise-trials", type=int, default=1)
    parser.add_argument("--adc-digital-filter", choices=["iir_fir", "none"], default="iir_fir")
    parser.add_argument("--l1-threshold-t1-uv", type=float, default=6100.0)
    parser.add_argument("--l1-threshold-t2-uv", type=float, default=5500.0)
    parser.add_argument("--l1-trigger-channel", type=int, default=1, help="Zero-based ADC channel index; 1 selects y.")
    parser.add_argument("--l1-trigger-channels", default=None, help="Comma-separated zero-based channel indices; e.g. 0,1 for x OR y.")
    parser.add_argument("--l1-station-jitter-std-uv", type=float, default=545.0)
    parser.add_argument("--l1-station-jitter-seed", type=int, default=24680)
    parser.add_argument("--l1-station-jitter-file", type=Path, default=None)
    parser.add_argument("--filter-vpeak", action="store_true", help="Use post-ADC filtered waveform for Vpeak threshold maps.")
    parser.add_argument("--vpeak-trigger-channel", type=int, default=1, help="Zero-based ADC channel for filtered Vpeak; negative uses all channels.")
    parser.add_argument("--vpeak-trigger-channels", default=None, help="Comma-separated zero-based channel indices for filtered Vpeak.")
    parser.add_argument("--theta-start", type=int, default=45)
    parser.add_argument("--theta-stop", type=int, default=89)
    parser.add_argument("--phi-start", type=int, default=0)
    parser.add_argument("--phi-stop", type=int, default=350)
    parser.add_argument("--phi-step", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--layout-path", type=Path, default=DEFAULT_LAYOUT_PATH)
    parser.add_argument("--layout-name", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _parse_trigger_channels(channel_list: str | None, fallback_channel: int) -> tuple[int, ...] | None:
    if channel_list is None or not str(channel_list).strip():
        return None if int(fallback_channel) < 0 else (int(fallback_channel),)
    channels = tuple(int(part.strip()) for part in str(channel_list).split(",") if part.strip())
    if not channels:
        raise ValueError("Trigger channel list is empty.")
    bad = [channel for channel in channels if channel < 0]
    if bad:
        raise ValueError(f"Trigger channel indices must be non-negative: {bad}")
    return tuple(dict.fromkeys(channels))


def _load_or_create_station_offsets(args: argparse.Namespace, station_count: int) -> np.ndarray:
    if args.l1_station_jitter_file is not None and Path(args.l1_station_jitter_file).exists():
        offsets = np.loadtxt(args.l1_station_jitter_file, dtype=float)
        offsets = np.asarray(offsets, dtype=np.float64).reshape(-1)
        if offsets.size != station_count:
            raise ValueError(
                f"Expected {station_count} station threshold offsets, got {offsets.size} from "
                f"{args.l1_station_jitter_file}."
            )
        return offsets

    rng = np.random.default_rng(int(args.l1_station_jitter_seed))
    offsets = rng.normal(0.0, float(args.l1_station_jitter_std_uv), size=station_count).astype(np.float64)
    if args.l1_station_jitter_file is not None:
        Path(args.l1_station_jitter_file).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.l1_station_jitter_file, offsets, fmt="%.12g")
    return offsets


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = args.output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for old_partial in partial_dir.glob("theta*_phi*.npz"):
            old_partial.unlink()

    layout_path = Path(args.layout_path)
    layout_name = args.layout_name or layout_path.stem
    positions_xy = read_layout_positions(layout_path).astype(np.float64)
    station_l1_offsets_uv = _load_or_create_station_offsets(args, positions_xy.shape[0])
    sampling_radius = float(args.core_margin_m) + float(np.max(np.linalg.norm(positions_xy, axis=1)))

    energy_log = np.round(np.arange(16.0, 18.0 + 1e-9, 0.1), 10)
    energy_eev = 10.0**energy_log / 1.0e18
    theta_values = np.arange(int(args.theta_start), int(args.theta_stop) + 1, dtype=int)
    phi_values = np.arange(int(args.phi_start), int(args.phi_stop) + 1, int(args.phi_step), dtype=int)
    all_jobs = [(int(theta), int(phi)) for theta in theta_values for phi in phi_values]
    completed = _completed_jobs(partial_dir) if args.resume else set()
    jobs = [job for job in all_jobs if job not in completed]

    l1_config = default_l1_config()
    l1_config.threshold_t1_uv = float(args.l1_threshold_t1_uv)
    l1_config.threshold_t2_uv = float(args.l1_threshold_t2_uv)
    l1_channels = _parse_trigger_channels(args.l1_trigger_channels, int(args.l1_trigger_channel))
    vpeak_channels = _parse_trigger_channels(args.vpeak_trigger_channels, int(args.vpeak_trigger_channel))
    l1_config.trigger_channel = l1_channels

    config = {
        "layout_path": str(layout_path),
        "layout_name": layout_name,
        "method": (
            f"{layout_name} full-bin adaptive scan. For each theta/phi, signal-only RFChain traces are computed at "
            f"{REFERENCE_EEV:g} EeV and scaled linearly with energy. Voltage threshold contours use "
            "adaptive refinement around all scaled threshold levels. L1 maps add reusable RFChain ADC-noise "
            "traces to scaled signal traces before applying the AERA-like trigger."
        ),
        "reference_eev": REFERENCE_EEV,
        "energy_log10_eV": energy_log.tolist(),
        "theta_values": theta_values.tolist(),
        "phi_values": phi_values.tolist(),
        "theta_start": int(args.theta_start),
        "theta_stop": int(args.theta_stop),
        "phi_start": int(args.phi_start),
        "phi_stop": int(args.phi_stop),
        "phi_step": int(args.phi_step),
        "coarse_grid_size": int(args.coarse_grid_size),
        "grid_extent_m": float(args.grid_extent_m),
        "refine_factor": int(args.refine_factor),
        "max_refine_points": int(args.max_refine_points),
        "target_cell_size_m": float(args.target_cell_size_m),
        "max_safe_sample_points": int(args.max_safe_sample_points),
        "legacy_regular_contours": bool(args.legacy_regular_contours),
        "l1_probability_refine": bool(args.l1_probability_refine),
        "voltage_band_fraction": float(args.voltage_band_fraction),
        "l1_boundary_level": float(args.l1_boundary_level),
        "l1_boundary_band": float(args.l1_boundary_band),
        "threshold_uv": float(args.threshold_uv),
        "required_antennas": int(args.required_antennas),
        "core_samples": int(args.core_samples),
        "core_margin_m": float(args.core_margin_m),
        "sampling_radius_m": float(sampling_radius),
        "seed": int(args.seed),
        "noise_library_size": int(args.noise_library_size),
        "noise_trials": int(args.noise_trials),
        "l1_threshold_t1_uv": float(args.l1_threshold_t1_uv),
        "l1_threshold_t2_uv": float(args.l1_threshold_t2_uv),
        "l1_station_jitter_std_uv": float(args.l1_station_jitter_std_uv),
        "l1_station_jitter_seed": int(args.l1_station_jitter_seed),
        "l1_station_threshold_offsets_uv": station_l1_offsets_uv.tolist(),
        "l1_station_threshold_t1_uv": (float(args.l1_threshold_t1_uv) + station_l1_offsets_uv).tolist(),
        "l1_station_threshold_t2_uv": (float(args.l1_threshold_t2_uv) + station_l1_offsets_uv).tolist(),
        "l1_config": asdict(l1_config),
        "adc_digital_filter": adc_filter_metadata() if args.adc_digital_filter == "iir_fir" else None,
        "filter_vpeak": bool(args.filter_vpeak),
        "vpeak_trigger_channel": vpeak_channels,
        "workers": int(args.workers),
    }
    (args.output_dir / "gp65_fullbin_adaptive_modes_settings.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    start = time.time()
    print(json.dumps({**config, "remaining_jobs": len(jobs), "completed_jobs": len(completed)}, ensure_ascii=False, indent=2), flush=True)
    noise_vadc = _build_shared_noise_library(args.noise_library_size, args.adc_digital_filter)
    if jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(
                positions_xy,
                energy_eev,
                args.coarse_grid_size,
                args.grid_extent_m,
                args.refine_factor,
                args.max_refine_points,
                args.target_cell_size_m,
                args.max_safe_sample_points,
                args.legacy_regular_contours,
                args.l1_probability_refine,
                args.voltage_band_fraction,
                args.l1_boundary_level,
                args.l1_boundary_band,
                args.threshold_uv,
                args.required_antennas,
                args.core_samples,
                sampling_radius,
                noise_vadc,
                args.noise_trials,
                args.seed,
                args.adc_digital_filter,
                args.l1_threshold_t1_uv,
                args.l1_threshold_t2_uv,
                station_l1_offsets_uv,
                l1_channels,
                args.filter_vpeak,
                vpeak_channels,
            ),
        ) as pool:
            for index, result in enumerate(pool.imap_unordered(_run_direction, jobs, chunksize=1), start=1):
                _write_partial(partial_dir, result)
                if index == 1 or index % 5 == 0 or index == len(jobs):
                    elapsed = time.time() - start
                    print(
                        f"directions {index}/{len(jobs)} done, theta={result['theta']} phi={result['phi']} "
                        f"points={result['sample_count']} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

    results = [_read_partial(path) for path in sorted(partial_dir.glob("theta*_phi*.npz"))]
    if len(results) != len(all_jobs):
        missing = sorted(set(all_jobs) - {(int(r["theta"]), int(r["phi"])) for r in results})
        raise RuntimeError(f"Missing {len(missing)} direction results, first missing: {missing[:5]}")

    shape = (len(energy_log), len(theta_values), len(phi_values))
    yita_voltage = np.zeros(shape, dtype=np.float64)
    fire_voltage = np.zeros(shape, dtype=np.float64)
    fire_voltage_triggered = np.zeros(shape, dtype=np.float64)
    yita_l1 = np.zeros(shape, dtype=np.float64)
    fire_l1 = np.zeros(shape, dtype=np.float64)
    fire_l1_triggered = np.zeros(shape, dtype=np.float64)
    sample_count = np.zeros((len(theta_values), len(phi_values)), dtype=np.int32)
    elapsed_direction = np.zeros((len(theta_values), len(phi_values)), dtype=np.float64)

    theta_to_idx = {int(v): i for i, v in enumerate(theta_values)}
    phi_to_idx = {int(v): i for i, v in enumerate(phi_values)}
    for result in results:
        ti = theta_to_idx[int(result["theta"])]
        pi = phi_to_idx[int(result["phi"])]
        yita_voltage[:, ti, pi] = result["yita_voltage"]
        fire_voltage[:, ti, pi] = result["fire_voltage"]
        fire_voltage_triggered[:, ti, pi] = result.get("fire_voltage_triggered", np.nan)
        yita_l1[:, ti, pi] = result["yita_l1"]
        fire_l1[:, ti, pi] = result["fire_l1"]
        fire_l1_triggered[:, ti, pi] = result.get("fire_l1_triggered", np.nan)
        sample_count[ti, pi] = int(result["sample_count"])
        elapsed_direction[ti, pi] = float(result["elapsed_s"])

    rate_config = FastRateConfig(
        data_root=DATA_ROOT,
        layout=layout_path,
        trigger_library=DATA_ROOT / "UNUSED_voltage_fullbin_no_threshold_library.mat",
        threshold_uv_m=75.0,
        required_antennas=args.required_antennas,
        num_core_samples=args.core_samples,
        trigger_radius_m=float(args.core_margin_m),
        seed=args.seed,
        energy_start=16.0,
        energy_stop=18.0,
        energy_step=0.1,
        theta_start=45,
        theta_stop=89,
        phi_start=int(args.phi_start),
        phi_stop=int(args.phi_stop),
        phi_step=int(args.phi_step),
        output_dir=args.output_dir,
    )
    trg_rate_voltage = compute_event_rate(rate_config, positions_xy, yita_voltage, energy_log, theta_values, phi_values)
    trg_rate_l1 = compute_event_rate(rate_config, positions_xy, yita_l1, energy_log, theta_values, phi_values)

    summary = {
        **config,
        "elapsed_s": float(time.time() - start),
        "voltage_threshold_daily_rate": float(np.nansum(trg_rate_voltage)),
        "l1_noisy_daily_rate": float(np.nansum(trg_rate_l1)),
        "voltage_mean_efficiency": float(np.mean(yita_voltage)),
        "l1_mean_efficiency": float(np.mean(yita_l1)),
        "mean_sample_count_per_direction": float(np.mean(sample_count)),
    }

    out_stem = (
        f"{layout_name}_adaptive_cg{args.coarse_grid_size}_rf{args.refine_factor}_rp{args.max_refine_points}_"
        f"thr{args.threshold_uv:g}uV_ant{args.required_antennas}_core{args.core_samples}"
    )
    mat_path = args.output_dir / f"{out_stem}.mat"
    csv_path = args.output_dir / f"{out_stem}_summary.csv"
    json_path = args.output_dir / f"{out_stem}_summary.json"
    savemat(
        mat_path,
        {
            "trg_rate_voltage_threshold": trg_rate_voltage,
            "trg_rate_l1_noisy": trg_rate_l1,
            "yita_voltage_threshold": yita_voltage,
            "fire_voltage_threshold": fire_voltage,
            "fire_voltage_threshold_triggered": fire_voltage_triggered,
            "yita_l1_noisy": yita_l1,
            "fire_l1_noisy": fire_l1,
            "fire_l1_noisy_triggered": fire_l1_triggered,
            "energy_log10_eV": energy_log,
            "energy_eev": energy_eev,
            "theta_values": theta_values,
            "phi_values": phi_values,
            "l1_station_threshold_offsets_uv": station_l1_offsets_uv,
            "l1_station_threshold_t1_uv": float(args.l1_threshold_t1_uv) + station_l1_offsets_uv,
            "l1_station_threshold_t2_uv": float(args.l1_threshold_t2_uv) + station_l1_offsets_uv,
            "sample_count": sample_count,
            "elapsed_direction_s": elapsed_direction,
            "summary_json": np.array([json.dumps(summary, ensure_ascii=False)], dtype=object),
        },
        do_compression=True,
    )
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(mat_path, flush=True)
    print(csv_path, flush=True)
    print(json_path, flush=True)


def _init_worker(
    positions_xy: np.ndarray,
    energy_eev: np.ndarray,
    coarse_grid_size: int,
    grid_extent_m: float,
    refine_factor: int,
    max_refine_points: int,
    target_cell_size_m: float,
    max_safe_sample_points: int,
    legacy_regular_contours: bool,
    l1_probability_refine: bool,
    voltage_band_fraction: float,
    l1_boundary_level: float,
    l1_boundary_band: float,
    threshold_uv: float,
    required_antennas: int,
    core_samples: int,
    sampling_radius: float,
    noise_vadc: np.ndarray,
    noise_trials: int,
    seed: int,
    adc_digital_filter: str,
    l1_threshold_t1_uv: float,
    l1_threshold_t2_uv: float,
    station_l1_offsets_uv: np.ndarray,
    l1_trigger_channel: tuple[int, ...] | None,
    filter_vpeak: bool,
    vpeak_trigger_channel: tuple[int, ...] | None,
) -> None:
    api = FARSimAPI()
    l1_config = default_l1_config()
    l1_config.threshold_t1_uv = float(l1_threshold_t1_uv)
    l1_config.threshold_t2_uv = float(l1_threshold_t2_uv)
    l1_config.trigger_channel = None if l1_trigger_channel is None else tuple(int(channel) for channel in l1_trigger_channel)
    axis = np.linspace(-float(grid_extent_m), float(grid_extent_m), int(coarse_grid_size))
    x_grid, y_grid = np.meshgrid(axis, axis)
    xy = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    _WORKER.clear()
    _WORKER.update(
        {
            "api": api,
            "predictor": api._predictor_cached(),
            "rfchain": api._rfchain_signal_cached(),
            "noise_vadc": np.asarray(noise_vadc, dtype=float),
            "l1_threshold_t1_uv": float(l1_threshold_t1_uv),
            "l1_threshold_t2_uv": float(l1_threshold_t2_uv),
            "station_l1_offsets_uv": np.asarray(station_l1_offsets_uv, dtype=np.float64),
            "l1_config": l1_config,
            "adc_digital_filter": str(adc_digital_filter),
            "filter_vpeak": bool(filter_vpeak),
            "vpeak_trigger_channel": None if vpeak_trigger_channel is None else tuple(int(channel) for channel in vpeak_trigger_channel),
            "positions_xy": np.asarray(positions_xy, dtype=np.float64),
            "energy_eev": np.asarray(energy_eev, dtype=np.float64),
            "axis": axis,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "xy": xy,
            "coarse_grid_size": int(coarse_grid_size),
            "grid_extent_m": float(grid_extent_m),
            "refine_factor": int(refine_factor),
            "max_refine_points": int(max_refine_points),
            "target_cell_size_m": float(target_cell_size_m),
            "max_safe_sample_points": int(max_safe_sample_points),
            "legacy_regular_contours": bool(legacy_regular_contours),
            "l1_probability_refine": bool(l1_probability_refine),
            "voltage_band_fraction": float(voltage_band_fraction),
            "l1_boundary_level": float(l1_boundary_level),
            "l1_boundary_band": float(l1_boundary_band),
            "threshold_uv": float(threshold_uv),
            "required_antennas": int(required_antennas),
            "core_samples": int(core_samples),
            "sampling_radius": float(sampling_radius),
            "noise_trials": int(noise_trials),
            "seed": int(seed),
        }
    )
    _trigger_stats_triggered_mean_numba(
        np.asarray(positions_xy, dtype=np.float64),
        np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float64),
        np.zeros((4, 2), dtype=np.float64),
        int(required_antennas),
    )
    _trigger_stats_station_polygons_numba(
        np.asarray(positions_xy, dtype=np.float64),
        np.zeros(positions_xy.shape[0], dtype=np.int64),
        np.zeros(positions_xy.shape[0], dtype=np.int64),
        np.empty((0, 2), dtype=np.float64),
        np.zeros((positions_xy.shape[0], 4), dtype=np.float64),
        np.zeros((4, 2), dtype=np.float64),
        int(required_antennas),
    )


def _run_direction(job: tuple[int, int]) -> dict[str, Any]:
    theta, phi = job
    start = time.time()
    energy_eev = _WORKER["energy_eev"]
    scales = energy_eev / REFERENCE_EEV

    if _WORKER["legacy_regular_contours"]:
        coarse_xy = _WORKER["xy"]
        coarse_peak, coarse_l1, _groups = _evaluate_points(coarse_xy, theta=theta, phi=phi, scales=scales, seed_offset=0)
        refine_points = _combined_refine_points(coarse_peak, coarse_l1, scales)
        if len(refine_points):
            refine_peak, refine_l1, _ = _evaluate_points(
                refine_points,
                theta=theta,
                phi=phi,
                scales=scales,
                seed_offset=1000003,
            )
            sample_xy = np.vstack((coarse_xy, refine_points))
            sample_peak = np.concatenate((coarse_peak, refine_peak))
            sample_l1 = np.concatenate((coarse_l1, refine_l1), axis=2 if coarse_l1.ndim == 3 else 1)
        else:
            sample_xy = coarse_xy
            sample_peak = coarse_peak
            sample_l1 = coarse_l1

        fine_size = (_WORKER["coarse_grid_size"] - 1) * _WORKER["refine_factor"] + 1
        fine_axis = np.linspace(-_WORKER["grid_extent_m"], _WORKER["grid_extent_m"], fine_size)
        x_fine, y_fine = np.meshgrid(fine_axis, fine_axis)
        peak_interp = _interpolate_to_grid(sample_xy, sample_peak, x_fine, y_fine)
        contour_mode = "legacy_regular"
        refine_count = int(len(refine_points))
    else:
        sample_xy, sample_peak, sample_l1, refine_count = _safe_adaptive_samples(
            theta=theta,
            phi=phi,
            scales=scales,
        )
        x_fine = y_fine = peak_interp = None
        contour_mode = "adaptive_tricontour"

    yita_voltage = np.zeros(len(energy_eev), dtype=np.float64)
    fire_voltage = np.zeros(len(energy_eev), dtype=np.float64)
    fire_voltage_triggered = np.zeros(len(energy_eev), dtype=np.float64)
    yita_l1 = np.zeros(len(energy_eev), dtype=np.float64)
    fire_l1 = np.zeros(len(energy_eev), dtype=np.float64)
    fire_l1_triggered = np.zeros(len(energy_eev), dtype=np.float64)

    for energy_idx, scale in enumerate(scales):
        seed = int(_WORKER["seed"] + energy_idx * 100000 + (theta - 45) * 1000 + int(phi))
        rng = np.random.default_rng(seed)
        centers = _matlab_like_core_positions(_WORKER["sampling_radius"], _WORKER["core_samples"], rng)

        if _WORKER["legacy_regular_contours"]:
            voltage_grid = peak_interp * float(scale)
            polygon_voltage = largest_contour_polygon(x_fine, y_fine, voltage_grid, _WORKER["threshold_uv"])
        else:
            polygon_voltage = _largest_scattered_contour_polygon(
                sample_xy,
                sample_peak * float(scale),
                _WORKER["threshold_uv"],
                max_edge_m=_WORKER["target_cell_size_m"] * 2.25,
            )
        if polygon_voltage.size:
            yita_voltage[energy_idx], fire_voltage[energy_idx], fire_voltage_triggered[energy_idx] = (
                _trigger_stats_triggered_mean_numba(
                _WORKER["positions_xy"],
                np.ascontiguousarray(polygon_voltage, dtype=np.float64),
                centers,
                _WORKER["required_antennas"],
            )
            )

        if sample_l1.ndim == 3:
            station_polygons = _station_l1_polygons(sample_xy, np.clip(sample_l1[:, energy_idx, :], 0.0, 1.0))
            if station_polygons[2].size:
                yita_l1[energy_idx], fire_l1[energy_idx], fire_l1_triggered[energy_idx] = (
                    _trigger_stats_station_polygons_numba(
                        _WORKER["positions_xy"],
                        station_polygons[0],
                        station_polygons[1],
                        station_polygons[2],
                        station_polygons[3],
                        centers,
                        _WORKER["required_antennas"],
                    )
                )
            continue

        if _WORKER["legacy_regular_contours"]:
            l1_grid = _interpolate_to_grid(sample_xy, sample_l1[energy_idx], x_fine, y_fine)
            l1_grid = np.clip(l1_grid, 0.0, 1.0)
            polygon_l1 = largest_contour_polygon(x_fine, y_fine, l1_grid, _WORKER["l1_boundary_level"])
        else:
            polygon_l1 = _largest_scattered_contour_polygon(
                sample_xy,
                np.clip(sample_l1[energy_idx], 0.0, 1.0),
                _WORKER["l1_boundary_level"],
                max_edge_m=_WORKER["target_cell_size_m"] * 2.25,
            )
        if polygon_l1.size:
            yita_l1[energy_idx], fire_l1[energy_idx], fire_l1_triggered[energy_idx] = (
                _trigger_stats_triggered_mean_numba(
                _WORKER["positions_xy"],
                np.ascontiguousarray(polygon_l1, dtype=np.float64),
                centers,
                _WORKER["required_antennas"],
            )
            )

    return {
        "theta": int(theta),
        "phi": int(phi),
        "yita_voltage": yita_voltage,
        "fire_voltage": fire_voltage,
        "fire_voltage_triggered": fire_voltage_triggered,
        "yita_l1": yita_l1,
        "fire_l1": fire_l1,
        "fire_l1_triggered": fire_l1_triggered,
        "sample_count": int(len(sample_xy)),
        "refine_count": int(refine_count),
        "contour_mode": contour_mode,
        "elapsed_s": float(time.time() - start),
    }


def _evaluate_points(
    xy: np.ndarray,
    *,
    theta: int,
    phi: int,
    scales: np.ndarray,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    predictions = _WORKER["predictor"].predict_fixed_shower_points(
        energy_eev=REFERENCE_EEV,
        theta_deg=float(theta),
        phi_deg=float(phi),
        xy_m=xy,
        t_peak_ns=0.0,
    )
    time_axis = predictions[0].t_ns
    traces = np.stack([prediction.trace_uv_m for prediction in predictions], axis=0)
    groups = _WORKER["rfchain"].forward_traces_vadc_groups(
        t_ns=time_axis,
        electric_traces_uv_m=traces,
        theta_deg=float(theta),
        phi_deg=float(phi),
    )
    peak_ref = np.zeros(len(xy), dtype=np.float64)
    station_offsets = np.asarray(_WORKER["station_l1_offsets_uv"], dtype=np.float64)
    l1_prob = np.zeros((station_offsets.size, len(scales), len(xy)), dtype=np.float64)
    noise_rng = np.random.default_rng(int(_WORKER["seed"] + seed_offset + (theta - 45) * 1009 + int(phi)))
    for trace_indices, t_ds, vadc in groups:
        vadc_l1 = apply_adc_iir_fir(t_ds, vadc) if _WORKER["adc_digital_filter"] == "iir_fir" else vadc
        if _WORKER["filter_vpeak"]:
            peak_source = vadc_l1
            channel = _WORKER["vpeak_trigger_channel"]
            if channel is None:
                peak_ref[trace_indices] = np.max(np.abs(peak_source), axis=(1, 2))
            else:
                peak_ref[trace_indices] = np.max(np.abs(peak_source[:, :, list(channel)]), axis=(1, 2))
        else:
            peak_ref[trace_indices] = np.max(np.abs(vadc), axis=(1, 2))
        gate_channel = _WORKER["l1_config"].trigger_channel
        if gate_channel is None:
            signal_gate_peaks = np.max(np.abs(vadc_l1), axis=(1, 2))
        else:
            signal_gate_peaks = np.max(np.abs(vadc_l1[:, :, list(gate_channel)]), axis=(1, 2))
        l1_signal_gate_uv = max(0.0, float(_WORKER["l1_threshold_t2_uv"]) + float(np.min(station_offsets)))
        for energy_idx, scale in enumerate(scales):
            scaled = vadc_l1 * float(scale)
            active_local = np.flatnonzero(signal_gate_peaks * float(scale) >= l1_signal_gate_uv)
            if active_local.size == 0:
                continue
            triggered = np.zeros((station_offsets.size, len(trace_indices)), dtype=np.float64)
            for _trial in range(max(1, _WORKER["noise_trials"])):
                for local_idx in active_local:
                    noise = _draw_noise_segment(_WORKER["noise_vadc"], scaled.shape[1], noise_rng)
                    combined = scaled[local_idx] + noise
                    triggered[:, local_idx] += _station_l1_trigger_vector(t_ds, combined)
            l1_prob[:, energy_idx, trace_indices] = triggered / max(1, _WORKER["noise_trials"])
    return peak_ref, l1_prob, groups


def _station_l1_trigger_vector(t_ns: np.ndarray, combined_vadc_uv: np.ndarray) -> np.ndarray:
    offsets = np.asarray(_WORKER["station_l1_offsets_uv"], dtype=np.float64)
    cfg = _WORKER["l1_config"]
    base_t1 = float(_WORKER["l1_threshold_t1_uv"])
    base_t2 = float(_WORKER["l1_threshold_t2_uv"])
    old_t1 = float(cfg.threshold_t1_uv)
    old_t2 = float(cfg.threshold_t2_uv if cfg.threshold_t2_uv is not None else 0.75 * cfg.threshold_t1_uv)
    out = np.zeros(offsets.size, dtype=np.float64)
    try:
        for station_idx, delta in enumerate(offsets):
            cfg.threshold_t1_uv = max(0.0, base_t1 + float(delta))
            cfg.threshold_t2_uv = max(0.0, base_t2 + float(delta))
            out[station_idx] = float(aera_voltage_trigger(t_ns, combined_vadc_uv, cfg).triggered)
    finally:
        cfg.threshold_t1_uv = old_t1
        cfg.threshold_t2_uv = old_t2
    return out


def _safe_adaptive_samples(
    *,
    theta: int,
    phi: int,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Recursively refine cells that can contain a Vpeak or L1 boundary.

    The old production path sampled a very coarse ``9x9`` grid over ``+-8000 m``
    and then drew contours on a regular interpolated image.  For compact
    low-zenith footprints this creates artificial large contours.  Here we only
    triangulate cells that have been refined down to a controlled physical cell
    size, and the same samples are used for both deterministic Vpeak and L1.
    """

    axis = np.asarray(_WORKER["axis"], dtype=np.float64)
    coarse_xy = _WORKER["xy"]
    coarse_peak, coarse_l1, _groups = _evaluate_points(coarse_xy, theta=theta, phi=phi, scales=scales, seed_offset=0)

    point_xy: dict[tuple[int, int], tuple[float, float]] = {}
    point_peak: dict[tuple[int, int], float] = {}
    point_l1: dict[tuple[int, int], np.ndarray] = {}
    for idx, (xy, peak) in enumerate(zip(coarse_xy, coarse_peak, strict=True)):
        l1_col = coarse_l1[:, :, idx] if coarse_l1.ndim == 3 else coarse_l1[:, idx]
        key = _point_key(float(xy[0]), float(xy[1]))
        point_xy[key] = (float(xy[0]), float(xy[1]))
        point_peak[key] = float(peak)
        point_l1[key] = np.asarray(l1_col, dtype=np.float64)

    cells = [
        (float(axis[ix]), float(axis[ix + 1]), float(axis[iy]), float(axis[iy + 1]))
        for iy in range(len(axis) - 1)
        for ix in range(len(axis) - 1)
    ]
    target = float(_WORKER["target_cell_size_m"])
    max_points = int(_WORKER["max_safe_sample_points"])
    refine_count = 0
    pass_index = 0
    while cells:
        pass_index += 1
        next_cells: list[tuple[float, float, float, float]] = []
        new_points: dict[tuple[int, int], tuple[float, float]] = {}
        for cell in cells:
            x0, x1, y0, y1 = cell
            if max(abs(x1 - x0), abs(y1 - y0)) <= target:
                continue
            if not _cell_needs_refine(cell, point_peak, point_l1, scales):
                continue
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)
            for y in (y0, ym, y1):
                for x in (x0, xm, x1):
                    key = _point_key(x, y)
                    if key not in point_xy:
                        new_points[key] = (float(x), float(y))
            next_cells.extend(
                [
                    (x0, xm, y0, ym),
                    (xm, x1, y0, ym),
                    (x0, xm, ym, y1),
                    (xm, x1, ym, y1),
                ]
            )

        if not new_points:
            break
        if len(point_xy) + len(new_points) > max_points:
            raise RuntimeError(
                f"Safe adaptive contour sampling would require {len(point_xy) + len(new_points)} points "
                f"for theta={theta}, phi={phi}; increase --max-safe-sample-points or --target-cell-size-m."
            )
        new_xy = np.asarray(list(new_points.values()), dtype=np.float64)
        new_peak, new_l1, _groups = _evaluate_points(
            new_xy,
            theta=theta,
            phi=phi,
            scales=scales,
            seed_offset=1000003 + pass_index * 131071,
        )
        for idx, (xy, peak) in enumerate(zip(new_xy, new_peak, strict=True)):
            l1_col = new_l1[:, :, idx] if new_l1.ndim == 3 else new_l1[:, idx]
            key = _point_key(float(xy[0]), float(xy[1]))
            point_xy[key] = (float(xy[0]), float(xy[1]))
            point_peak[key] = float(peak)
            point_l1[key] = np.asarray(l1_col, dtype=np.float64)
        refine_count += len(new_points)
        cells = next_cells

    keys = list(point_xy)
    sample_xy = np.asarray([point_xy[key] for key in keys], dtype=np.float64)
    sample_peak = np.asarray([point_peak[key] for key in keys], dtype=np.float64)
    first_l1 = point_l1[keys[0]]
    sample_l1 = np.stack([point_l1[key] for key in keys], axis=first_l1.ndim)
    return sample_xy, sample_peak, sample_l1, refine_count


def _cell_needs_refine(
    cell: tuple[float, float, float, float],
    point_peak: dict[tuple[int, int], float],
    point_l1: dict[tuple[int, int], np.ndarray],
    scales: np.ndarray,
) -> bool:
    x0, x1, y0, y1 = cell
    keys = [_point_key(x0, y0), _point_key(x1, y0), _point_key(x0, y1), _point_key(x1, y1)]
    peak_values = np.asarray([point_peak[key] for key in keys], dtype=np.float64)
    peak_min = float(np.nanmin(peak_values))
    peak_max = float(np.nanmax(peak_values))
    signal_thresholds = [float(_WORKER["threshold_uv"])]
    l1_config = _WORKER.get("l1_config")
    if l1_config is not None:
        offsets = np.asarray(_WORKER.get("station_l1_offsets_uv", np.zeros(1)), dtype=np.float64)
        base_t2 = float(_WORKER.get("l1_threshold_t2_uv", l1_config.threshold_t2_uv))
        signal_thresholds.append(max(0.0, base_t2 + float(np.min(offsets))))
        signal_thresholds.append(max(0.0, base_t2 + float(np.max(offsets))))
    thresholds_ref = np.unique(np.asarray(signal_thresholds, dtype=np.float64)[:, None] / np.asarray(scales, dtype=np.float64)[None, :])
    band_fraction = float(_WORKER["voltage_band_fraction"])
    for threshold_ref in thresholds_ref:
        threshold_ref = float(threshold_ref)
        band = max(abs(threshold_ref) * band_fraction, 1.0)
        if peak_min <= threshold_ref <= peak_max:
            return True
        if min(abs(peak_min - threshold_ref), abs(peak_max - threshold_ref)) <= band:
            return True

    if not _WORKER["l1_probability_refine"]:
        return False

    first_l1 = point_l1[keys[0]]
    l1_values = np.stack([point_l1[key] for key in keys], axis=first_l1.ndim)
    l1_level = float(_WORKER["l1_boundary_level"])
    l1_band = float(_WORKER["l1_boundary_band"])
    l1_min = np.nanmin(l1_values, axis=first_l1.ndim)
    l1_max = np.nanmax(l1_values, axis=first_l1.ndim)
    crosses = (l1_min <= l1_level) & (l1_level <= l1_max)
    close = np.minimum(np.abs(l1_min - l1_level), np.abs(l1_max - l1_level)) <= l1_band
    return bool(np.any(crosses | close))


def _largest_scattered_contour_polygon(
    sample_xy: np.ndarray,
    sample_values: np.ndarray,
    level: float,
    *,
    max_edge_m: float,
) -> np.ndarray:
    values = np.asarray(sample_values, dtype=np.float64)
    xy = np.asarray(sample_xy, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    if np.count_nonzero(finite) < 3:
        return np.empty((0, 2), dtype=np.float64)
    xy = xy[finite]
    values = values[finite]
    if not (np.nanmin(values) <= level <= np.nanmax(values)):
        return np.empty((0, 2), dtype=np.float64)

    try:
        triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1])
    except Exception:
        return np.empty((0, 2), dtype=np.float64)
    triangles = triangulation.triangles
    if triangles.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = xy[triangles]
    edge01 = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
    edge12 = np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1)
    edge20 = np.linalg.norm(pts[:, 2] - pts[:, 0], axis=1)
    triangulation.set_mask(np.maximum.reduce([edge01, edge12, edge20]) > float(max_edge_m))

    fig, ax = plt.subplots()
    try:
        contour = ax.tricontour(triangulation, values, levels=[level])
        polygons: list[np.ndarray] = []
        for collection in contour.collections:
            for path in collection.get_paths():
                vertices = np.asarray(path.vertices, dtype=np.float64)
                if len(vertices) >= 3:
                    polygons.append(vertices)
        if not polygons:
            return np.empty((0, 2), dtype=np.float64)
        return max(polygons, key=_polygon_area)
    except Exception:
        return np.empty((0, 2), dtype=np.float64)
    finally:
        plt.close(fig)


def _station_l1_polygons(sample_xy: np.ndarray, station_l1_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    polygons: list[np.ndarray] = []
    starts: list[int] = []
    lengths: list[int] = []
    bounds: list[tuple[float, float, float, float]] = []
    cursor = 0
    for station_idx in range(station_l1_values.shape[0]):
        polygon = _largest_scattered_contour_polygon(
            sample_xy,
            np.asarray(station_l1_values[station_idx], dtype=np.float64),
            _WORKER["l1_boundary_level"],
            max_edge_m=_WORKER["target_cell_size_m"] * 2.25,
        )
        starts.append(cursor)
        if polygon.size:
            polygon = np.ascontiguousarray(polygon, dtype=np.float64)
            polygons.append(polygon)
            length = int(polygon.shape[0])
            lengths.append(length)
            bounds.append(
                (
                    float(np.min(polygon[:, 0])),
                    float(np.max(polygon[:, 0])),
                    float(np.min(polygon[:, 1])),
                    float(np.max(polygon[:, 1])),
                )
            )
            cursor += length
        else:
            lengths.append(0)
            bounds.append((0.0, -1.0, 0.0, -1.0))
    vertices = np.vstack(polygons) if polygons else np.empty((0, 2), dtype=np.float64)
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(lengths, dtype=np.int64),
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.asarray(bounds, dtype=np.float64),
    )


@njit(fastmath=True, cache=True)
def _trigger_stats_station_polygons_numba(
    positions_xy: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    polygon_vertices: np.ndarray,
    bounds: np.ndarray,
    centers_xy: np.ndarray,
    required_antennas: int,
) -> tuple[float, float, float]:
    n_centers = centers_xy.shape[0]
    n_antennas = positions_xy.shape[0]
    triggered = 0
    fired_sum = 0.0
    fired_sum_triggered = 0.0
    for center_idx in range(n_centers):
        cx = centers_xy[center_idx, 0]
        cy = centers_xy[center_idx, 1]
        fired = 0
        for ant_idx in range(n_antennas):
            length = lengths[ant_idx]
            if length <= 0:
                continue
            x = positions_xy[ant_idx, 0] + cx
            y = positions_xy[ant_idx, 1] + cy
            if x < bounds[ant_idx, 0] or x > bounds[ant_idx, 1] or y < bounds[ant_idx, 2] or y > bounds[ant_idx, 3]:
                continue
            if _point_in_flat_polygon_numba(x, y, polygon_vertices, starts[ant_idx], length):
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
def _point_in_flat_polygon_numba(x: float, y: float, vertices: np.ndarray, start: int, length: int) -> bool:
    inside = False
    j = length - 1
    for i in range(length):
        xi = vertices[start + i, 0]
        yi = vertices[start + i, 1]
        xj = vertices[start + j, 0]
        yj = vertices[start + j, 1]
        if (yi > y) != (yj > y):
            x_intersection = (xj - xi) * (y - yi) / (yj - yi + 1.0e-300) + xi
            if x < x_intersection:
                inside = not inside
        j = i
    return inside


def _combined_refine_points(coarse_peak: np.ndarray, coarse_l1: np.ndarray, scales: np.ndarray) -> np.ndarray:
    x_axis = _WORKER["axis"]
    y_axis = _WORKER["axis"]
    peak_grid = coarse_peak.reshape(len(y_axis), len(x_axis))
    l1_cube = coarse_l1.reshape(len(scales), len(y_axis), len(x_axis))
    existing = {(_round_key(x), _round_key(y)) for x, y in _WORKER["xy"]}
    candidates: list[tuple[float, float, float]] = []
    thresholds_ref = _WORKER["threshold_uv"] / np.asarray(scales, dtype=float)
    voltage_band = max(_WORKER["threshold_uv"] * _WORKER["voltage_band_fraction"], 1.0)
    l1_level = float(_WORKER["l1_boundary_level"])
    l1_band = float(_WORKER["l1_boundary_band"])

    for iy in range(len(y_axis) - 1):
        for ix in range(len(x_axis) - 1):
            peak_corners = peak_grid[iy : iy + 2, ix : ix + 2].ravel()
            peak_min = float(np.nanmin(peak_corners))
            peak_max = float(np.nanmax(peak_corners))
            voltage_hit = False
            for threshold_ref in thresholds_ref:
                band_ref = voltage_band / max(float(threshold_ref), 1.0) * float(threshold_ref)
                brackets = peak_min <= threshold_ref <= peak_max
                close = min(abs(peak_min - threshold_ref), abs(peak_max - threshold_ref)) <= band_ref
                if brackets or close:
                    voltage_hit = True
                    break

            l1_hit = False
            for energy_idx in range(l1_cube.shape[0]):
                corners = l1_cube[energy_idx, iy : iy + 2, ix : ix + 2].ravel()
                cmin = float(np.nanmin(corners))
                cmax = float(np.nanmax(corners))
                if cmin <= l1_level <= cmax or min(abs(cmin - l1_level), abs(cmax - l1_level)) <= l1_band:
                    l1_hit = True
                    break

            if not voltage_hit and not l1_hit:
                continue
            x_local = np.linspace(x_axis[ix], x_axis[ix + 1], _WORKER["refine_factor"] + 1)
            y_local = np.linspace(y_axis[iy], y_axis[iy + 1], _WORKER["refine_factor"] + 1)
            center_x = 0.5 * (x_axis[ix] + x_axis[ix + 1])
            center_y = 0.5 * (y_axis[iy] + y_axis[iy + 1])
            for y in y_local:
                for x in x_local:
                    key = (_round_key(x), _round_key(y))
                    if key in existing:
                        continue
                    existing.add(key)
                    candidates.append((float(np.hypot(x - center_x, y - center_y)), float(x), float(y)))
    if not candidates:
        return np.empty((0, 2), dtype=np.float64)
    candidates.sort(key=lambda item: item[0])
    candidates = candidates[: max(0, int(_WORKER["max_refine_points"]))]
    return np.asarray([(x, y) for _dist, x, y in candidates], dtype=np.float64)


def _interpolate_to_grid(sample_xy: np.ndarray, sample_values: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    linear = LinearNDInterpolator(sample_xy, sample_values, fill_value=np.nan)
    nearest = NearestNDInterpolator(sample_xy, sample_values)
    values = np.asarray(linear(x_grid, y_grid), dtype=float)
    missing = ~np.isfinite(values)
    if np.any(missing):
        values[missing] = nearest(x_grid[missing], y_grid[missing])
    return values


def _draw_noise_segment(noise_traces: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if noise_traces.shape[0] == 0:
        return np.zeros((n_samples, 3), dtype=float)
    trace = noise_traces[int(rng.integers(0, noise_traces.shape[0]))]
    if trace.shape[0] == n_samples:
        return trace
    if trace.shape[0] > n_samples:
        start = int(rng.integers(0, trace.shape[0] - n_samples + 1))
        return trace[start : start + n_samples]
    repeats = int(np.ceil(n_samples / trace.shape[0]))
    return np.tile(trace, (repeats, 1))[:n_samples]


def _build_shared_noise_library(size: int, adc_digital_filter: str) -> np.ndarray:
    if int(size) <= 0:
        return np.empty((0, 1, 3), dtype=float)
    api = FARSimAPI()
    library = build_adc_noise_library(
        t_ns=api.config.t_axis_ns,
        rfchain=api._rfchain_noise_cached(),
        size=int(size),
    )
    vadc = np.asarray(library.vadc_uv, dtype=float)
    if str(adc_digital_filter) == "iir_fir":
        vadc = apply_adc_iir_fir(library.t_ns, vadc)
    return vadc


def _write_partial(partial_dir: Path, result: dict[str, Any]) -> None:
    path = partial_dir / f"theta{int(result['theta']):02d}_phi{int(result['phi']):03d}.npz"
    np.savez_compressed(path, **result)


def _read_partial(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key].item() if data[key].shape == () else data[key] for key in data.files}


def _completed_jobs(partial_dir: Path) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for path in partial_dir.glob("theta*_phi*.npz"):
        try:
            data = np.load(path, allow_pickle=True)
            if "fire_voltage_triggered" not in data.files or "fire_l1_triggered" not in data.files:
                continue
        except Exception:
            continue
        stem = path.stem
        theta = int(stem.split("_")[0].replace("theta", ""))
        phi = int(stem.split("_")[1].replace("phi", ""))
        out.add((theta, phi))
    return out


def _round_key(value: float) -> int:
    return int(round(float(value) * 1_000_000.0))


def _point_key(x: float, y: float) -> tuple[int, int]:
    return (_round_key(x), _round_key(y))


def _polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


if __name__ == "__main__":
    main()
