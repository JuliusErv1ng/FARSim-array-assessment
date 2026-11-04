from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import load_mat_file
from .physics import FieldLibrary, MIXED_TEMPLATE_NAME, field_library_from_cache, rotate_xmax, shower_to_ground_vector


DEFAULT_DATA_ROOT = Path("/Volumes/T7/FARSim/from mapx/StshpOutbox_XDS")


@dataclass(slots=True)
class FieldFootprint:
    """FARSim electric-field footprint for one primary-particle geometry."""

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    field_vector_uv_m: np.ndarray
    field_peak_uv_m: np.ndarray
    xmax_xyz: np.ndarray
    metadata: dict[str, float | str]


def compute_field_footprint(
    *,
    energy_eev: float,
    theta_deg: float,
    phi_deg: float,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    position_efield_cache: str | Path | None = None,
    signed_superposition: bool = True,
) -> FieldFootprint:
    """Compute a vector electric-field footprint from the FARSim cache.

    Energy scales only the radio amplitude.  The reference Xmax geometry is
    kept fixed, matching the current FARSim pipeline used for the paper.
    """

    data_root = Path(data_root)
    cache_path = Path(position_efield_cache) if position_efield_cache is not None else data_root / MIXED_TEMPLATE_NAME
    cache = load_mat_file(cache_path)
    library = field_library_from_cache(cache)

    theta_index = int(round(theta_deg)) - 45
    if theta_index < 0 or theta_index >= library.eg_total.shape[1]:
        raise ValueError("theta_deg must be within the FARSim library range 45-89 deg.")
    energy_eev = float(energy_eev)
    if energy_eev <= 0.0:
        raise ValueError("energy_eev must be positive.")

    geomagnetic_ratio = _geomagnetic_ratio(theta_deg=float(theta_deg), phi_deg=float(phi_deg), library=library)
    xmax_xyz = rotate_xmax(library.xmax_total[theta_index], float(phi_deg))
    x_m, y_m, z_m, field_vector, field_peak = shower_to_ground_vector(
        xmax_xyz=xmax_xyz,
        b_field=library.b_field,
        ea_now=library.ea_total[:, theta_index],
        eg_now=library.eg_total[:, theta_index] * geomagnetic_ratio,
        rq_now=library.rq[:, theta_index],
        signed_superposition=signed_superposition,
    )
    amplitude_ratio = energy_eev / 0.316
    field_vector = field_vector * amplitude_ratio
    field_peak = field_peak * amplitude_ratio
    return FieldFootprint(
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        field_vector_uv_m=field_vector,
        field_peak_uv_m=field_peak,
        xmax_xyz=xmax_xyz,
        metadata={
            "energy_eev": energy_eev,
            "theta_deg": float(theta_deg),
            "phi_deg": float(phi_deg),
            "position_efield_cache": str(cache_path),
            "geomagnetic_ratio": float(geomagnetic_ratio),
        },
    )


def _geomagnetic_ratio(*, theta_deg: float, phi_deg: float, library: FieldLibrary) -> float:
    lib_direction = np.array(
        [
            np.sin(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(45.0)),
            np.sin(np.deg2rad(theta_deg)) * np.sin(np.deg2rad(45.0)),
            np.cos(np.deg2rad(theta_deg)),
        ],
        dtype=float,
    )
    now_direction = np.array(
        [
            np.sin(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(phi_deg)),
            np.sin(np.deg2rad(theta_deg)) * np.sin(np.deg2rad(phi_deg)),
            np.cos(np.deg2rad(theta_deg)),
        ],
        dtype=float,
    )
    angle_reference = _angle_between(lib_direction, library.b_field)
    angle_now = _angle_between(now_direction, library.b_field)
    denom = np.sin(np.deg2rad(angle_reference))
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.sin(np.deg2rad(angle_now)) / denom)


def _angle_between(vec1: np.ndarray, vec2: np.ndarray) -> float:
    cos_angle = np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))
