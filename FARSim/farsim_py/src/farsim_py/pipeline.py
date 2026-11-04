from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.interpolate import interp1d

from .config import PipelineConfig
from .efficiency import trigger_efficiency
from .io_utils import build_output_path, load_mat_file, read_flux_curve, read_layout_positions, save_mat_file
from .physics import FieldLibrary, build_field_library, largest_contour_polygon, rotate_xmax, shower_to_ground


def run_pipeline(config: PipelineConfig, seed: int = 12345) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    positions_xy = read_layout_positions(config.layout_path)

    cache = load_mat_file(config.position_efield_cache)
    xmax_data = np.asarray(cache["xmax_total"], dtype=float)
    position_efield_total = np.asarray(cache["position_efield_total"], dtype=float)
    field_library = build_field_library(position_efield_total, xmax_data)

    theta_values = np.arange(config.grid.theta_start, config.grid.theta_stop + 1, 1, dtype=int)
    phi_values = np.arange(config.grid.phi_start, config.grid.phi_stop + 1, 1, dtype=int)
    eval_phi_values = np.arange(config.grid.phi_start, config.grid.phi_stop + 1, config.grid.phi_step, dtype=int)
    phi_to_index = {int(phi): idx for idx, phi in enumerate(phi_values)}
    energy_values = np.arange(config.grid.energy_start, config.grid.energy_stop + 1e-9, config.grid.energy_step)

    trigger_region = np.empty((len(energy_values), len(theta_values), len(phi_values)), dtype=object)
    xmax_grid = np.empty((len(energy_values), len(theta_values), len(phi_values), 3), dtype=float)

    for energy_idx, energy in enumerate(energy_values):
        ratio_energy = (10.0**energy) / (0.316 * 1e18)
        for theta_idx, theta in enumerate(theta_values):
            lib_angle = np.array(
                [
                    np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(45.0)),
                    np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(45.0)),
                    np.cos(np.deg2rad(theta)),
                ]
            )
            anglelib2b = angle_between(lib_angle, field_library.b_field)
            # Keep the reference Xmax geometry fixed with energy.  The fast
            # model scales only the radio amplitude with primary energy; moving
            # Xmax by an empirical energy ratio would also move the footprint
            # projection and mix energy scaling with geometry.
            xmax_reference = field_library.xmax_total[theta - 45]
            rq_now = field_library.rq[:, theta - 45]
            ea_now = field_library.ea_total[:, theta - 45]

            for phi_idx, phi in enumerate(phi_values):
                now_angle = np.array(
                    [
                        np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                        np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
                        np.cos(np.deg2rad(theta)),
                    ]
                )
                anglenow2b = angle_between(now_angle, field_library.b_field)
                ratio_ga = np.sin(np.deg2rad(anglenow2b)) / np.sin(np.deg2rad(anglelib2b))
                xmax_xyz = rotate_xmax(xmax_reference, phi)
                eg_now = field_library.eg_total[:, theta - 45] * ratio_ga
                x_ground, y_ground, etotal = shower_to_ground(
                    xmax_xyz=xmax_xyz,
                    b_field=field_library.b_field,
                    ea_now=ea_now,
                    eg_now=eg_now,
                    rq_now=rq_now,
                    signed_superposition=True,
                )
                polygon = largest_contour_polygon(
                    x_grid=x_ground,
                    y_grid=y_ground,
                    field_grid=etotal * ratio_energy,
                    threshold_uv_m=config.trigger.threshold_uv_m,
                )
                trigger_region[energy_idx, theta_idx, phi_idx] = polygon
                xmax_grid[energy_idx, theta_idx, phi_idx] = xmax_xyz

    yita = np.zeros((len(energy_values), len(theta_values), len(eval_phi_values)), dtype=float)
    fire_num = np.zeros_like(yita)
    error_total = np.empty_like(yita, dtype=object)

    for energy_idx, energy in enumerate(energy_values):
        for theta_idx, theta in enumerate(theta_values):
            for phi_eval_idx, phi in enumerate(eval_phi_values):
                phi_grid_idx = phi_to_index[int(phi)]
                polygon = trigger_region[energy_idx, theta_idx, phi_grid_idx]
                xmax_xyz = xmax_grid[energy_idx, theta_idx, phi_grid_idx]
                result = trigger_efficiency(
                    positions_xy=positions_xy,
                    trigger_region_xy=polygon,
                    required_antennas=config.trigger.num_antennas,
                    num_core_samples=config.runtime.num_core_samples,
                    trigger_radius_m=config.runtime.trigger_radius_m,
                    use_polygon_radius=config.runtime.use_polygon_radius,
                    xmax_xyz=xmax_xyz,
                    theta_deg=float(theta),
                    phi_deg=float(phi),
                    timing_jitter_ns=config.runtime.timing_jitter_ns,
                    rng=rng,
                )
                yita[energy_idx, theta_idx, phi_eval_idx] = result.efficiency
                fire_num[energy_idx, theta_idx, phi_eval_idx] = result.mean_fired_antennas
                error_total[energy_idx, theta_idx, phi_eval_idx] = result.errors_deg

    trg_rate = compute_event_rate(config, positions_xy, yita, energy_values, theta_values, eval_phi_values)
    output_path = build_output_path(
        layout_path=config.layout_path,
        output_root=config.runtime.output_dir,
        threshold_uv_m=config.trigger.threshold_uv_m,
        num_antennas=config.trigger.num_antennas,
    )

    payload = {
        "trg_rate": trg_rate,
        "yita": yita,
        "fire_num": fire_num,
        "error_total": error_total,
        "energy_values": energy_values,
        "theta_values": theta_values,
        "phi_values": eval_phi_values,
        "layout_path": str(config.layout_path),
        "threshold_uv_m": config.trigger.threshold_uv_m,
        "num_antennas": config.trigger.num_antennas,
        "config": np.array([str(asdict(config))], dtype=object),
    }
    save_mat_file(output_path, payload)
    return {"output_path": output_path, **payload}


def compute_event_rate(
    config: PipelineConfig,
    positions_xy: np.ndarray,
    yita: np.ndarray,
    energy_values: np.ndarray,
    theta_values: np.ndarray,
    phi_values: np.ndarray,
) -> np.ndarray:
    flux = read_flux_curve(config.flux_path)
    flux[:, 1] = flux[:, 1] * 1e24
    xi = np.arange(15.33, 18.25 + 1e-9, 0.01)
    yi = interp1d(flux[:, 0], flux[:, 1], kind="cubic", fill_value="extrapolate")(xi)

    r_d = np.max(np.linalg.norm(positions_xy, axis=1))
    dd = config.runtime.trigger_radius_m + r_d
    trg_rate = np.zeros_like(yita)

    for energy_idx, energy in enumerate(energy_values):
        energy_high = np.round(energy + config.grid.energy_step, 2)
        energy_low = np.round(energy, 2)
        high_idx = np.where(np.isclose(xi, energy_high))[0][0]
        low_idx = np.where(np.isclose(xi, energy_low))[0][0]
        for phi_idx, _phi in enumerate(phi_values):
            for theta_idx, theta in enumerate(theta_values):
                area = np.pi * dd**2 * np.cos(np.deg2rad(theta))
                sr = (np.cos(np.deg2rad(theta)) - np.cos(np.deg2rad(theta + 1))) * np.pi / 180.0
                n_events = (
                    (yi[high_idx] / (10.0**energy_high) ** 3 + yi[low_idx] / (10.0**energy_low) ** 3)
                    * (10.0**energy_high - 10.0**energy_low)
                    * 0.5
                    * area
                    * config.runtime.day_seconds
                    * sr
                )
                trg_rate[energy_idx, theta_idx, phi_idx] = (
                    yita[energy_idx, theta_idx, phi_idx] * n_events * config.grid.phi_step
                )
    return trg_rate


def angle_between(vec1: np.ndarray, vec2: np.ndarray) -> float:
    cos_angle = np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))
