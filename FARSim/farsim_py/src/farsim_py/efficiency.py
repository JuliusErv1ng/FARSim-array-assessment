from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

from .reconstruction import get_zhs_effective_refractive_index, plane_wave_reconstruction_error


@dataclass(slots=True)
class EfficiencyResult:
    efficiency: float
    errors_deg: np.ndarray
    mean_fired_antennas: float


def random_core_positions(radius_m: float, count: int, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    radius = radius_m * np.sqrt(rng.uniform(0.0, 1.0, count))
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))


def trigger_efficiency(
    positions_xy: np.ndarray,
    trigger_region_xy: np.ndarray,
    required_antennas: int,
    num_core_samples: int,
    trigger_radius_m: float,
    use_polygon_radius: bool,
    xmax_xyz: np.ndarray,
    theta_deg: float,
    phi_deg: float,
    timing_jitter_ns: float,
    rng: np.random.Generator,
) -> EfficiencyResult:
    if trigger_region_xy.size == 0:
        return EfficiencyResult(0.0, np.empty(0, dtype=float), 0.0)

    polygon_radius = np.max(np.linalg.norm(trigger_region_xy, axis=1))
    h_r = polygon_radius if use_polygon_radius else trigger_radius_m
    r_d = np.max(np.linalg.norm(positions_xy, axis=1))
    dd = h_r + r_d

    centers = random_core_positions(dd, num_core_samples, rng)
    trigger_path = Path(trigger_region_xy)
    trigger_flags = np.zeros(num_core_samples, dtype=bool)
    fired_counts = np.zeros(num_core_samples, dtype=float)
    errors: list[float] = []

    n_eff = get_zhs_effective_refractive_index(0.0, 0.0, 0.0, xmax_xyz[0], xmax_xyz[1], xmax_xyz[2])
    c_eff = 299_792_458.0 / n_eff

    for idx, center in enumerate(centers):
        shifted_polygon = trigger_region_xy - center
        inside = Path(shifted_polygon).contains_points(positions_xy)
        count = int(np.count_nonzero(inside))
        fired_counts[idx] = count
        if count < required_antennas:
            continue

        trigger_flags[idx] = True
        triggered = positions_xy[inside]
        xmax_shifted = np.array([xmax_xyz[0] - center[0], xmax_xyz[1] - center[1], xmax_xyz[2]], dtype=float)
        distance = np.sqrt(
            (triggered[:, 0] - xmax_shifted[0]) ** 2
            + (triggered[:, 1] - xmax_shifted[1]) ** 2
            + xmax_shifted[2] ** 2
        )
        times = distance / c_eff
        times += rng.uniform(-timing_jitter_ns, timing_jitter_ns, size=times.shape[0]) * 1e-9
        points_with_time = np.column_stack((triggered, np.zeros(triggered.shape[0]), times))
        errors.append(plane_wave_reconstruction_error(points_with_time, n_eff, theta_deg, phi_deg))

    return EfficiencyResult(
        efficiency=float(np.mean(trigger_flags)),
        errors_deg=np.asarray(errors, dtype=float),
        mean_fired_antennas=float(np.mean(fired_counts)),
    )
