from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata

from farsim_py.io_utils import load_mat_file
from farsim_py.physics import (
    FieldLibrary,
    MIXED_TEMPLATE_NAME,
    field_library_from_cache,
    rotate_xmax,
    shower_to_ground_vector,
)
from farsim_py.reconstruction import get_zhs_effective_refractive_index

from .synthesizer import WaveformInput, WaveformResult, WaveformSynthesizer


DEFAULT_BUNDLE_ROOT = Path(os.environ.get("FARSIM_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_DATA_ROOT = Path(os.environ.get("FARSIM_DATA_ROOT", DEFAULT_BUNDLE_ROOT / "data" / "StshpOutbox_XDS"))
DEFAULT_RING_METRICS = Path(
    os.environ.get(
        "FARSIM_RING_METRICS",
        DEFAULT_BUNDLE_ROOT / "waveform_width_analysis" / "outputs" / "pulse_width_metrics_raw.csv",
    )
)


@dataclass(slots=True)
class StationPrediction:
    """Complete FARSim waveform prediction for one ground station."""

    result: WaveformResult
    waveform_input: WaveformInput
    geometry: dict[str, float | str | int]

    @property
    def t_ns(self) -> np.ndarray:
        return self.result.t_ns

    @property
    def trace_uv_m(self) -> np.ndarray:
        return self.result.trace_uv_m


class FARSimWaveformPredictor:
    """Predict three-polarization time-domain traces without ZHAireS truth traces.

    The public prediction path uses only

    1. the FARSim position-and-peak-field cache,
    2. the Xmax-energy scaling table, and
    3. the waveform coefficients embedded in :mod:`farsim_waveform`.

    Training tables and individual ``a*.trace`` files are only needed for
    fitting or validating the coefficients, not for deployment.
    """

    def __init__(
        self,
        *,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        position_efield_cache: str | Path | None = None,
        xmax2energy_path: str | Path | None = None,
        t_start_ns: float = -100.0,
        t_stop_ns: float = 250.0,
        dt_ns: float = 0.5,
        t_axis_ns: np.ndarray | None = None,
        template: str = "asymmetric_tail",
        pca_model_path: str | None = None,
        frequency_model_path: str | None = None,
        complex_model_path: str | None = None,
        kernel_model_path: str | None = None,
        mlp_model_path: str | None = None,
        signed_superposition: bool = True,
        polarity_sign: float = -1.0,
        ring_reference: str = "empirical",
        ring_metrics_path: str | Path | None = None,
    ) -> None:
        if ring_reference not in {"empirical", "theory"}:
            raise ValueError("ring_reference must be 'empirical' or 'theory'.")
        self.data_root = Path(data_root)
        self.position_efield_cache = (
            Path(position_efield_cache)
            if position_efield_cache is not None
            else self.data_root / MIXED_TEMPLATE_NAME
        )
        self.xmax2energy_path = Path(xmax2energy_path) if xmax2energy_path is not None else self.data_root / "xmax2energy.mat"
        self.signed_superposition = bool(signed_superposition)
        self.polarity_sign = float(polarity_sign)
        self.ring_reference = ring_reference
        self.ring_metrics_path = Path(ring_metrics_path) if ring_metrics_path is not None else DEFAULT_RING_METRICS
        self.template = template
        self.pca_model_path = pca_model_path
        self.frequency_model_path = frequency_model_path
        self.complex_model_path = complex_model_path
        self.kernel_model_path = kernel_model_path
        self.mlp_model_path = mlp_model_path
        self.synthesizer = WaveformSynthesizer(
            t_start_ns=t_start_ns,
            t_stop_ns=t_stop_ns,
            dt_ns=dt_ns,
            t_axis_ns=t_axis_ns,
            template=template,
            pca_model_path=pca_model_path,
            frequency_model_path=frequency_model_path,
            complex_model_path=complex_model_path,
            kernel_model_path=kernel_model_path,
            mlp_model_path=mlp_model_path,
        )
        self._field_library: FieldLibrary | None = None
        self._empirical_ring_table: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    def predict_one(
        self,
        *,
        energy_eev: float,
        theta_deg: float,
        phi_deg: float,
        x_m: float,
        y_m: float,
        z_m: float = 0.0,
        station_id: str | int | None = None,
        particle: str = "Iron",
        psi_ref_deg: float | None = None,
        t_peak_ns: float = 0.0,
        t_axis_ns: np.ndarray | None = None,
    ) -> StationPrediction:
        """Predict ``Ex, Ey, Ez`` trace at one ground position.

        Parameters
        ----------
        energy_eev:
            Primary energy in EeV.
        theta_deg, phi_deg:
            Arrival direction in degrees, using the same convention as FARSim.
        x_m, y_m, z_m:
            Ground-station coordinates relative to the shower core, in metres.
        psi_ref_deg:
            Optional reference Cherenkov angle.  If omitted, it is computed from
            the ZHS effective refractive index along the core-Xmax path.
        t_axis_ns:
            Optional explicit time axis.  Use this for one-to-one comparison
            with a measured/simulated trace or RFchain sampling.
        """

        library = self._ensure_library()
        theta_index = int(round(theta_deg)) - 45
        if theta_index < 0 or theta_index >= library.eg_total.shape[1]:
            raise ValueError("theta_deg must be within the FARSim library range 45-89 deg.")

        energy_eev = float(energy_eev)
        if energy_eev <= 0.0:
            raise ValueError("energy_eev must be positive.")
        log_energy_ev = float(np.log10(energy_eev * 1.0e18))
        amplitude_ratio = energy_eev / 0.316

        geomagnetic_ratio = self._geomagnetic_ratio(theta_deg=float(theta_deg), phi_deg=float(phi_deg), library=library)
        # Keep Xmax fixed at the reference geometry stored in the FARSim cache.
        # Energy changes the field amplitude, not the shower-plane projection.
        xmax_xyz = rotate_xmax(library.xmax_total[theta_index], float(phi_deg))

        rq_now = library.rq[:, theta_index]
        ea_now = library.ea_total[:, theta_index]
        eg_now = library.eg_total[:, theta_index] * geomagnetic_ratio
        x_grid, y_grid, _z_grid, field_grid, _scalar_grid = shower_to_ground_vector(
            xmax_xyz=xmax_xyz,
            b_field=library.b_field,
            ea_now=ea_now,
            eg_now=eg_now,
            rq_now=rq_now,
            signed_superposition=self.signed_superposition,
        )
        peak_vector = (
            self.polarity_sign
            * amplitude_ratio
            * _interpolate_vector_field(x_grid, y_grid, field_grid, float(x_m), float(y_m))
        )

        station_xyz = np.array([float(x_m), float(y_m), float(z_m)], dtype=float)
        resolved_psi_ref_deg = psi_ref_deg
        if resolved_psi_ref_deg is None and self.ring_reference == "empirical":
            resolved_psi_ref_deg = self._empirical_ring_angle(particle=particle, theta_deg=float(theta_deg))
        geometry = _station_geometry(
            station_xyz=station_xyz,
            xmax_xyz=xmax_xyz,
            theta_deg=float(theta_deg),
            phi_deg=float(phi_deg),
            particle=particle,
            b_field=library.b_field,
            psi_ref_deg=resolved_psi_ref_deg,
        )

        resolved_station_id = station_id if station_id is not None else f"x{x_m:.1f}_y{y_m:.1f}"
        item = WaveformInput(
            station_id=resolved_station_id,
            peak_ex_uv_m=float(peak_vector[0]),
            peak_ey_uv_m=float(peak_vector[1]),
            peak_ez_uv_m=float(peak_vector[2]),
            rho_shower_m=float(geometry["rho_shower_m"]),
            delta_psi_deg=float(geometry["delta_psi_deg"]),
            distance_xmax_km=float(geometry["distance_xmax_km"]),
            theta_deg=float(theta_deg),
            particle=particle,
            off_axis_deg=float(geometry["off_axis_deg"]),
            phi_deg=float(phi_deg),
            shower_phi_deg=float(geometry["shower_phi_deg"]),
            angle_to_b_deg=float(geometry["angle_to_b_deg"]),
            t_peak_ns=float(t_peak_ns),
        )
        synthesizer = (
            self.synthesizer
            if t_axis_ns is None
            else WaveformSynthesizer(
                t_axis_ns=t_axis_ns,
                template=self.template,
                pca_model_path=self.pca_model_path,
                frequency_model_path=self.frequency_model_path,
                complex_model_path=self.complex_model_path,
                kernel_model_path=self.kernel_model_path,
                mlp_model_path=self.mlp_model_path,
            )
        )
        result = synthesizer.synthesize_one(item)
        result.metadata.update(
            {
                "energy_eev": energy_eev,
                "log_energy_ev": log_energy_ev,
                "phi_deg": float(phi_deg),
                "x_m": float(x_m),
                "y_m": float(y_m),
                "z_m": float(z_m),
                "psi_ref_deg": float(geometry["psi_ref_deg"]),
                "ring_reference": self.ring_reference,
                "geomagnetic_ratio": float(geomagnetic_ratio),
                "polarity_sign": float(self.polarity_sign),
                "position_efield_cache": str(self.position_efield_cache),
            }
        )
        return StationPrediction(result=result, waveform_input=item, geometry=geometry)

    def predict_many(self, stations: Iterable[dict[str, float | str | int]]) -> list[StationPrediction]:
        """Predict traces from a sequence of mappings accepted by ``predict_one``."""

        return [self.predict_one(**station) for station in stations]

    def predict_fixed_shower_points(
        self,
        *,
        energy_eev: float,
        theta_deg: float,
        phi_deg: float,
        xy_m: np.ndarray,
        z_m: float = 0.0,
        particle: str = "Iron",
        t_peak_ns: float = 0.0,
    ) -> list[StationPrediction]:
        """Predict many stations for one fixed shower using shared interpolators."""

        library = self._ensure_library()
        theta_index = int(round(theta_deg)) - 45
        if theta_index < 0 or theta_index >= library.eg_total.shape[1]:
            raise ValueError("theta_deg must be within the FARSim library range 45-89 deg.")

        xy_m = np.asarray(xy_m, dtype=float)
        if xy_m.ndim != 2 or xy_m.shape[1] != 2:
            raise ValueError("xy_m must have shape (n_points, 2).")
        energy_eev = float(energy_eev)
        if energy_eev <= 0.0:
            raise ValueError("energy_eev must be positive.")
        amplitude_ratio = energy_eev / 0.316
        log_energy_ev = float(np.log10(energy_eev * 1.0e18))

        geomagnetic_ratio = self._geomagnetic_ratio(theta_deg=float(theta_deg), phi_deg=float(phi_deg), library=library)
        xmax_xyz = rotate_xmax(library.xmax_total[theta_index], float(phi_deg))
        x_grid, y_grid, _z_grid, field_grid, _scalar_grid = shower_to_ground_vector(
            xmax_xyz=xmax_xyz,
            b_field=library.b_field,
            ea_now=library.ea_total[:, theta_index],
            eg_now=library.eg_total[:, theta_index] * geomagnetic_ratio,
            rq_now=library.rq[:, theta_index],
            signed_superposition=self.signed_superposition,
        )
        peak_vectors = self.polarity_sign * amplitude_ratio * _interpolate_vector_field_many(x_grid, y_grid, field_grid, xy_m)
        resolved_psi_ref_deg = None
        if self.ring_reference == "empirical":
            resolved_psi_ref_deg = self._empirical_ring_angle(particle=particle, theta_deg=float(theta_deg))

        predictions: list[StationPrediction] = []
        for index, ((x_m, y_m), peak_vector) in enumerate(zip(xy_m, peak_vectors, strict=True)):
            station_xyz = np.array([float(x_m), float(y_m), float(z_m)], dtype=float)
            geometry = _station_geometry(
                station_xyz=station_xyz,
                xmax_xyz=xmax_xyz,
                theta_deg=float(theta_deg),
                phi_deg=float(phi_deg),
                particle=particle,
                b_field=library.b_field,
                psi_ref_deg=resolved_psi_ref_deg,
            )
            item = WaveformInput(
                station_id=f"p{index}",
                peak_ex_uv_m=float(peak_vector[0]),
                peak_ey_uv_m=float(peak_vector[1]),
                peak_ez_uv_m=float(peak_vector[2]),
                rho_shower_m=float(geometry["rho_shower_m"]),
                delta_psi_deg=float(geometry["delta_psi_deg"]),
                distance_xmax_km=float(geometry["distance_xmax_km"]),
                theta_deg=float(theta_deg),
                particle=particle,
                off_axis_deg=float(geometry["off_axis_deg"]),
                phi_deg=float(phi_deg),
                shower_phi_deg=float(geometry["shower_phi_deg"]),
                angle_to_b_deg=float(geometry["angle_to_b_deg"]),
                t_peak_ns=float(t_peak_ns),
            )
            result = self.synthesizer.synthesize_one(item)
            result.metadata.update(
                {
                    "energy_eev": energy_eev,
                    "log_energy_ev": log_energy_ev,
                    "phi_deg": float(phi_deg),
                    "x_m": float(x_m),
                    "y_m": float(y_m),
                    "z_m": float(z_m),
                    "psi_ref_deg": float(geometry["psi_ref_deg"]),
                    "ring_reference": self.ring_reference,
                    "geomagnetic_ratio": float(geomagnetic_ratio),
                    "polarity_sign": float(self.polarity_sign),
                    "position_efield_cache": str(self.position_efield_cache),
                }
            )
            predictions.append(StationPrediction(result=result, waveform_input=item, geometry=geometry))
        return predictions

    def _ensure_library(self) -> FieldLibrary:
        if self._field_library is None:
            cache = load_mat_file(self.position_efield_cache)
            self._field_library = field_library_from_cache(cache)
        return self._field_library

    def _empirical_ring_angle(self, *, particle: str, theta_deg: float) -> float | None:
        table = self._ensure_empirical_ring_table()
        key = "Proton" if str(particle).lower().startswith("proton") else "Iron"
        if key not in table:
            return None
        theta_axis, psi_axis = table[key]
        theta = float(np.clip(float(theta_deg), float(theta_axis[0]), float(theta_axis[-1])))
        return float(np.interp(theta, theta_axis, psi_axis))

    def _ensure_empirical_ring_table(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self._empirical_ring_table is not None:
            return self._empirical_ring_table
        if not self.ring_metrics_path.exists():
            self._empirical_ring_table = {}
            return self._empirical_ring_table

        values: dict[tuple[str, float], list[float]] = {}
        with self.ring_metrics_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    particle = "Proton" if row["particle"].lower().startswith("proton") else "Iron"
                    theta = float(row["theta_deg"])
                    psi = float(row["psi_ref_empirical_deg"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(theta) and np.isfinite(psi) and psi > 0.0:
                    values.setdefault((particle, theta), []).append(psi)

        table: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for particle in ("Iron", "Proton"):
            items = sorted((theta, float(np.median(psi_values))) for (p, theta), psi_values in values.items() if p == particle)
            if items:
                table[particle] = (
                    np.asarray([item[0] for item in items], dtype=float),
                    np.asarray([item[1] for item in items], dtype=float),
                )
        self._empirical_ring_table = table
        return table

    @staticmethod
    def _geomagnetic_ratio(*, theta_deg: float, phi_deg: float, library: FieldLibrary) -> float:
        return _geomagnetic_ratio(theta_deg=theta_deg, phi_deg=phi_deg, b_field=library.b_field)


def _interpolate_vector_field(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    field_grid: np.ndarray,
    x_m: float,
    y_m: float,
) -> np.ndarray:
    points = np.column_stack((np.ravel(x_grid), np.ravel(y_grid)))
    target = np.array([[float(x_m), float(y_m)]], dtype=float)
    vector = np.empty(3, dtype=float)
    for comp in range(3):
        values = np.ravel(field_grid[..., comp])
        linear = griddata(points, values, target, method="linear")[0]
        if np.isfinite(linear):
            vector[comp] = float(linear)
        else:
            vector[comp] = float(griddata(points, values, target, method="nearest")[0])
    return vector


def _interpolate_vector_field_many(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    field_grid: np.ndarray,
    xy_m: np.ndarray,
) -> np.ndarray:
    points = np.column_stack((np.ravel(x_grid), np.ravel(y_grid)))
    target = np.asarray(xy_m, dtype=float)
    vectors = np.empty((len(target), 3), dtype=float)
    for comp in range(3):
        values = np.ravel(field_grid[..., comp])
        linear = LinearNDInterpolator(points, values, fill_value=np.nan)
        nearest = NearestNDInterpolator(points, values)
        component = np.asarray(linear(target), dtype=float)
        missing = ~np.isfinite(component)
        if np.any(missing):
            component[missing] = nearest(target[missing])
        vectors[:, comp] = component
    return vectors


def _station_geometry(
    *,
    station_xyz: np.ndarray,
    xmax_xyz: np.ndarray,
    theta_deg: float,
    phi_deg: float,
    particle: str,
    b_field: np.ndarray,
    psi_ref_deg: float | None,
) -> dict[str, float | str | int]:
    axis_down = _unit(-np.asarray(xmax_xyz, dtype=float))
    if not np.isfinite(axis_down).all():
        axis_down = -_direction_vector(theta_deg, phi_deg)
    e1 = _unit(np.array([axis_down[1], -axis_down[0], 0.0], dtype=float))
    if not np.isfinite(e1).all() or np.linalg.norm(e1) == 0.0:
        e1 = np.array([1.0, 0.0, 0.0], dtype=float)
    e2 = _unit(np.cross(axis_down, e1))

    sx = float(np.dot(station_xyz, e1))
    sy = float(np.dot(station_xyz, e2))
    rho_shower = float(np.hypot(sx, sy))
    shower_phi = float(np.rad2deg(np.arctan2(sy, sx)))

    obs = station_xyz - xmax_xyz
    distance_xmax_km = float(np.linalg.norm(obs) / 1000.0)
    off_axis = _angle_deg(obs, axis_down)
    if psi_ref_deg is None:
        n_eff = get_zhs_effective_refractive_index(0.0, 0.0, 0.0, xmax_xyz[0], xmax_xyz[1], xmax_xyz[2])
        psi_ref = float(np.rad2deg(np.arccos(np.clip(1.0 / n_eff, -1.0, 1.0))))
    else:
        n_eff = np.nan
        psi_ref = float(psi_ref_deg)

    return {
        "theta_deg": float(theta_deg),
        "phi_deg": float(phi_deg),
        "particle": particle,
        "rho_shower_m": rho_shower,
        "shower_phi_deg": shower_phi,
        "off_axis_deg": float(off_axis),
        "psi_ref_deg": psi_ref,
        "delta_psi_deg": float(off_axis - psi_ref),
        "distance_xmax_km": distance_xmax_km,
        "n_eff": float(n_eff),
        "angle_to_b_deg": _angle_deg(axis_down, b_field),
    }


def _geomagnetic_ratio(*, theta_deg: float, phi_deg: float, b_field: np.ndarray) -> float:
    library_direction = _direction_vector(theta_deg, 45.0)
    target_direction = _direction_vector(theta_deg, phi_deg)
    reference = np.sin(np.deg2rad(_angle_deg(library_direction, b_field)))
    if abs(reference) < 1e-12:
        return 1.0
    return float(np.sin(np.deg2rad(_angle_deg(target_direction, b_field))) / reference)


def _direction_vector(theta_deg: float, phi_deg: float) -> np.ndarray:
    return np.array(
        [
            np.sin(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(phi_deg)),
            np.sin(np.deg2rad(theta_deg)) * np.sin(np.deg2rad(phi_deg)),
            np.cos(np.deg2rad(theta_deg)),
        ],
        dtype=float,
    )


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.full_like(vec, np.nan, dtype=float)
    return np.asarray(vec, dtype=float) / norm


def synthesize_trace_at_position(**kwargs) -> StationPrediction:
    """Convenience wrapper for one-off predictions with default FARSim paths."""

    return FARSimWaveformPredictor().predict_one(**kwargs)
