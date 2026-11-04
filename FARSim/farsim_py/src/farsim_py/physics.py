from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator, griddata


@dataclass(slots=True)
class FieldLibrary:
    """Reference lateral field templates.

    The geomagnetic/Askaryan component profiles are tabulated on the off-axis
    angle ``psi_q_deg``.  ``rq`` is kept as the equivalent shower-plane radius
    for each zenith bin because the ground-projection code still needs a
    physical radius grid.
    """

    rq: np.ndarray
    psi_q_deg: np.ndarray
    eg_total: np.ndarray
    ea_total: np.ndarray
    xmax_total: np.ndarray
    b_field: np.ndarray


def local_magnetic_field() -> np.ndarray:
    return np.array(
        [
            np.sin(np.deg2rad(151.91)) * np.cos(np.deg2rad(-0.06)),
            np.sin(np.deg2rad(151.91)) * np.sin(np.deg2rad(-0.06)),
            np.cos(np.deg2rad(151.91)),
        ],
        dtype=float,
    )


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def shower_basis(inc: np.ndarray, b_field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = -_normalize(inc)
    x_axis = _normalize(np.cross(z_axis, b_field))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return x_axis, y_axis, z_axis


def c2c(position_efield_total: np.ndarray, b_field: np.ndarray, inc_zenith: int, inc: np.ndarray) -> np.ndarray:
    x_axis, y_axis, z_axis = shower_basis(inc, b_field)
    xyz = np.column_stack((x_axis, y_axis, z_axis))

    plane = np.asarray(position_efield_total[inc_zenith - 45], dtype=float)
    site_xyz = plane[:, :3]
    field_xyz = plane[:, 3:6]

    site_pie = site_xyz @ np.linalg.pinv(xyz).T
    phi_pie = np.mod(np.rad2deg(np.arctan2(site_pie[:, 1], site_pie[:, 0])), 360.0)
    r_pie = np.round(np.hypot(site_pie[:, 0], site_pie[:, 1]))

    ex_pie = field_xyz @ x_axis
    ey_pie = field_xyz @ y_axis
    ez_pie = field_xyz @ z_axis
    return np.column_stack((r_pie, np.round(phi_pie), ex_pie, ey_pie, ez_pie))


def rho_to_off_axis_deg(rho_m: np.ndarray | float, distance_xmax_m: float) -> np.ndarray:
    """Convert shower-plane radius to off-axis angle relative to Xmax."""

    distance = _positive_distance_xmax(distance_xmax_m)
    return np.rad2deg(np.arctan2(np.asarray(rho_m, dtype=float), distance))


def off_axis_deg_to_rho(psi_deg: np.ndarray | float, distance_xmax_m: float) -> np.ndarray:
    """Convert off-axis angle to the equivalent shower-plane radius."""

    distance = _positive_distance_xmax(distance_xmax_m)
    return distance * np.tan(np.deg2rad(np.asarray(psi_deg, dtype=float)))


def _positive_distance_xmax(distance_xmax_m: float) -> float:
    distance = float(distance_xmax_m)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_xmax_m must be finite and positive.")
    return distance


def extract_ga_components(
    position_efield_pie: np.ndarray,
    distance_xmax_m: float,
    radial_bins: int = 500,
    min_ring_points: int = 6,
    radius_tolerance_m: float = 5.0,
    angular_tolerance_deg: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract geomagnetic and Askaryan profiles on an off-axis-angle axis."""

    samples = np.asarray(position_efield_pie, dtype=float)
    samples = select_structured_ring_samples(
        samples,
        min_ring_points=min_ring_points,
        radius_tolerance_m=radius_tolerance_m,
        angular_tolerance_deg=angular_tolerance_deg,
    )

    core_sample = samples[np.argmin(samples[:, 0]) : np.argmin(samples[:, 0]) + 1, :]
    extra = np.repeat(core_sample, 359, axis=0)
    extra[:, 1] = np.arange(1, 360)
    samples = np.vstack((samples, extra))

    wrap = samples[samples[:, 1] == 360].copy()
    wrap[:, 1] = 0
    samples = np.vstack((samples, wrap))

    psi_samples = samples.copy()
    psi_samples[:, 0] = rho_to_off_axis_deg(psi_samples[:, 0], distance_xmax_m)
    psi_limit = float(psi_samples[:, 0].max())
    psi_q = np.linspace(0.0, psi_limit, int(radial_bins) + 1)
    rq = off_axis_deg_to_rho(psi_q, distance_xmax_m)
    phiq = np.arange(0.0, 361.0, 1.0)
    psi_grid, phiq_grid = np.meshgrid(psi_q, phiq)

    exq = _grid_fill(psi_samples[:, :2], psi_samples[:, 2], psi_grid, phiq_grid)
    eyq = _grid_fill(psi_samples[:, :2], psi_samples[:, 3], psi_grid, phiq_grid)

    eg = (exq[0, :] + exq[90, :] + exq[180, :] + exq[270, :]) / 4.0
    ea = (eyq[90, :] - eyq[270, :]) / 2.0
    return rq, psi_q, eg, ea


def select_structured_ring_samples(
    samples: np.ndarray,
    *,
    min_ring_points: int = 6,
    radius_tolerance_m: float = 5.0,
    angular_tolerance_deg: float = 10.0,
) -> np.ndarray:
    """Select the polar-ring calibration antennas without relying on row order.

    The ZHAireS reference layouts contain an eight-azimuth polar calibration grid
    plus a core antenna, followed by extra/random antennas.  The original MATLAB
    implementation kept the first 161 rows, which silently assumes this file
    order.  Here we identify the calibration grid geometrically: keep the core
    and radius clusters that contain enough azimuth samples to represent a ring.
    """

    if samples.ndim != 2 or samples.shape[1] < 5:
        raise ValueError("position_efield_pie must be an Nx5 array.")

    radius = np.asarray(samples[:, 0], dtype=float)
    order = np.argsort(radius)
    keep_indices: list[int] = []

    core_radius = max(1.0, float(radius_tolerance_m))
    core_candidates = np.flatnonzero(radius <= core_radius)
    if len(core_candidates) == 0:
        raise ValueError("Could not identify the core antenna sample.")
    keep_indices.append(int(core_candidates[np.argmin(radius[core_candidates])]))

    sorted_radius = radius[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_radius[end] - sorted_radius[start] <= radius_tolerance_m:
            end += 1
        group_idx = order[start:end]
        if np.mean(radius[group_idx]) > core_radius and len(group_idx) >= min_ring_points:
            keep_indices.extend(
                _ring_representative_indices(
                    samples,
                    group_idx,
                    min_ring_points=min_ring_points,
                    angular_tolerance_deg=angular_tolerance_deg,
                )
            )
        start = end

    selected = samples[np.unique(keep_indices)].copy()
    if selected.shape[0] < min_ring_points + 1:
        raise ValueError("Could not identify enough structured ring samples.")
    return selected[np.lexsort((selected[:, 1], selected[:, 0]))]


def _ring_representative_indices(
    samples: np.ndarray,
    group_idx: np.ndarray,
    *,
    min_ring_points: int,
    angular_tolerance_deg: float,
) -> list[int]:
    """Choose at most one sample near each nominal 45-degree calibration angle."""

    canonical = np.arange(0.0, 360.0, 45.0)
    phi = np.mod(samples[group_idx, 1], 360.0)
    chosen: list[int] = []
    used: set[int] = set()

    for angle in canonical:
        distance = np.abs(((phi - angle + 180.0) % 360.0) - 180.0)
        order = np.argsort(distance)
        for local_idx in order:
            original_idx = int(group_idx[local_idx])
            if original_idx in used:
                continue
            if float(distance[local_idx]) <= angular_tolerance_deg:
                chosen.append(original_idx)
                used.add(original_idx)
            break

    if len(chosen) < min_ring_points:
        return []
    return chosen


def _grid_fill(points: np.ndarray, values: np.ndarray, rq_grid: np.ndarray, phiq_grid: np.ndarray) -> np.ndarray:
    linear = griddata(points, values, (rq_grid, phiq_grid), method="linear")
    nearest = griddata(points, values, (rq_grid, phiq_grid), method="nearest")
    return np.where(np.isnan(linear), nearest, linear)


def build_field_library(position_efield_total: np.ndarray, xmax_total: np.ndarray) -> FieldLibrary:
    b_field = local_magnetic_field()
    eg_total = []
    ea_total = []
    rq_total = []
    psi_total = []

    for zenith in range(45, 90):
        xmax_now = np.asarray(xmax_total[zenith - 45], dtype=float)
        inc = _normalize(xmax_now)
        position_efield_pie = c2c(position_efield_total, b_field, zenith, inc)
        rq, psi_q, eg, ea = extract_ga_components(
            position_efield_pie,
            distance_xmax_m=float(np.linalg.norm(xmax_now)),
        )
        rq_total.append(rq)
        psi_total.append(psi_q)
        eg_total.append(eg)
        ea_total.append(ea)

    return FieldLibrary(
        rq=np.column_stack(rq_total),
        psi_q_deg=np.column_stack(psi_total),
        eg_total=np.column_stack(eg_total),
        ea_total=np.column_stack(ea_total),
        xmax_total=np.asarray(xmax_total, dtype=float),
        b_field=b_field,
    )


MIXED_TEMPLATE_NAME = "farsim_equal_prior_four_arm_50_200MHz.mat"


def field_library_from_cache(cache: dict) -> FieldLibrary:
    """Load a radial p/Fe template, or a legacy station cache explicitly supplied by the caller."""
    if "composition_weights" not in cache and "position_efield_total" in cache:
        return build_field_library(cache["position_efield_total"], cache["xmax_total"])

    theta = np.asarray(cache["theta_values"]).ravel()
    weights = np.asarray(cache["composition_weights"]).ravel()
    radius, egeo, ece, xmax = (
        np.asarray(cache[key], dtype=float)
        for key in ("radius_total", "egeo_total", "ece_total", "xmax_total")
    )
    if (not np.array_equal(theta, np.arange(45, 90))
        or not np.array_equal(weights, [0.5, 0.5])
        or radius.ndim != 2 or radius.shape[0] != theta.size
        or radius.shape != egeo.shape or radius.shape != ece.shape
        or xmax.shape != (theta.size, 3)
        or not all(np.isfinite(a).all() for a in (radius, egeo, ece, xmax))
        or np.any(np.diff(radius, axis=1) <= 0)
        or np.any(np.linalg.norm(xmax, axis=1) <= 0)):
        raise ValueError("Invalid equal-weight proton/iron reference template")
    distance = np.linalg.norm(xmax, axis=1)
    psi = np.rad2deg(np.arctan2(radius, distance[:, None]))
    return FieldLibrary(radius.T, psi.T, egeo.T, ece.T, xmax, local_magnetic_field())


def rotate_xmax(xmax: np.ndarray, phi_deg: float) -> np.ndarray:
    phi = np.arctan2(xmax[1], xmax[0])
    rho = np.hypot(xmax[0], xmax[1])
    z = xmax[2]
    x_new = rho * np.cos(phi - np.pi / 4.0 + np.deg2rad(phi_deg))
    y_new = rho * np.sin(phi - np.pi / 4.0 + np.deg2rad(phi_deg))
    return np.array([x_new, y_new, z], dtype=float)


def shower_to_ground(
    xmax_xyz: np.ndarray,
    b_field: np.ndarray,
    ea_now: np.ndarray,
    eg_now: np.ndarray,
    rq_now: np.ndarray,
    signed_superposition: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_ground, y_ground, _z_ground, _field_vector, etotal = shower_to_ground_vector(
        xmax_xyz=xmax_xyz,
        b_field=b_field,
        ea_now=ea_now,
        eg_now=eg_now,
        rq_now=rq_now,
        signed_superposition=signed_superposition,
    )
    return x_ground, y_ground, etotal


def shower_to_ground_vector(
    xmax_xyz: np.ndarray,
    b_field: np.ndarray,
    ea_now: np.ndarray,
    eg_now: np.ndarray,
    rq_now: np.ndarray,
    signed_superposition: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project shower-plane components to the ground and return vector fields.

    Returns
    -------
    x_ground, y_ground, z_ground:
        Ground-plane coordinates for each ``(radius, azimuth)`` grid point.
    field_vector:
        Cartesian electric-field vector ``[..., (Ex, Ey, Ez)]`` on the same
        ground grid.
    etotal:
        Magnitude of ``field_vector``.  This preserves the old scalar API while
        exposing the three-polarization interface needed by RF-chain studies.
    """

    x_axis, y_axis, _ = shower_basis(xmax_xyz, b_field)
    angles_deg = np.arange(0.0, 361.0, 1.0)
    cos_phi = np.cos(np.deg2rad(angles_deg))
    sin_phi = np.sin(np.deg2rad(angles_deg))

    x_now = np.outer(rq_now, cos_phi) * x_axis[0] + np.outer(rq_now, sin_phi) * y_axis[0]
    y_now = np.outer(rq_now, cos_phi) * x_axis[1] + np.outer(rq_now, sin_phi) * y_axis[1]
    z_now = np.outer(rq_now, cos_phi) * x_axis[2] + np.outer(rq_now, sin_phi) * y_axis[2]

    if signed_superposition:
        ex_pie = -(np.outer(ea_now, cos_phi) + eg_now[:, None])
        ey_pie = -np.outer(ea_now, sin_phi)
    else:
        ex_pie = -(np.outer(np.abs(ea_now), cos_phi) + np.abs(eg_now)[:, None])
        ey_pie = -np.outer(np.abs(ea_now), sin_phi)

    t = -xmax_xyz[2] / (z_now - xmax_xyz[2])
    x_ground = xmax_xyz[0] + (x_now - xmax_xyz[0]) * t
    y_ground = xmax_xyz[1] + (y_now - xmax_xyz[1]) * t
    z_ground = xmax_xyz[2] + (z_now - xmax_xyz[2]) * t

    ground_dist = np.sqrt((x_ground - xmax_xyz[0]) ** 2 + (y_ground - xmax_xyz[1]) ** 2 + (z_ground - xmax_xyz[2]) ** 2)
    plane_dist = np.sqrt((x_now - xmax_xyz[0]) ** 2 + (y_now - xmax_xyz[1]) ** 2 + (z_now - xmax_xyz[2]) ** 2)
    ratio = np.divide(ground_dist, plane_dist, out=np.ones_like(ground_dist), where=plane_dist > 0)
    field_vector = (ex_pie[..., None] * x_axis + ey_pie[..., None] * y_axis) / ratio[..., None]
    etotal = np.linalg.norm(field_vector, axis=2)
    return x_ground, y_ground, z_ground, field_vector, etotal


def largest_contour_polygon(x_grid: np.ndarray, y_grid: np.ndarray, field_grid: np.ndarray, threshold_uv_m: float) -> np.ndarray:
    figure, axis = plt.subplots()
    try:
        contour = axis.contour(x_grid, y_grid, field_grid, levels=[threshold_uv_m])
        polygons: list[np.ndarray] = []
        for collection in contour.collections:
            for path in collection.get_paths():
                vertices = path.vertices
                if len(vertices) >= 3:
                    polygons.append(vertices)
        if not polygons:
            return np.empty((0, 2), dtype=float)
        return max(polygons, key=_polygon_area)
    finally:
        plt.close(figure)


def _polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def energy_scaling_model(energy2xmax: np.ndarray, reference_energy: float = 17.5) -> callable:
    energy_axis = np.linspace(16.0, 18.0, len(energy2xmax))
    interpolator = RegularGridInterpolator((energy_axis,), energy2xmax, bounds_error=False, fill_value=None)
    reference = float(interpolator([[reference_energy]])[0])

    def _ratio(log_energy: float) -> float:
        return float(interpolator([[log_energy]])[0]) / reference

    return _ratio
