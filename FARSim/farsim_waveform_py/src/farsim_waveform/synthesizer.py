from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .complex_template import ComplexTemplateModel
from .frequency_template import FrequencyTemplateModel
from .kernel_template import DEFAULT_RING_FOCUSED_KERNEL_TEMPLATE_MODEL, ComponentKernelTemplateModel, KernelTemplateModel
from .mlp_pca_template import MLPPcaTemplateModel
from .pca_template import PCATemplateModel
from .templates import bipolar_gaussian_derivative, asymmetric_gaussian_tail, normalize_template, params_from_geometry
from .width_model import estimate_fwhm_ns


def estimate_trace_fwhm(t_ns: np.ndarray, y: np.ndarray) -> float:
    """Estimate the FWHM of the dominant absolute pulse lobe."""

    trace = np.abs(np.asarray(y, dtype=float))
    if trace.size == 0 or not np.isfinite(trace).any():
        return float("nan")
    peak = float(np.nanmax(trace))
    if peak <= 0.0:
        return 0.0
    above = np.flatnonzero(trace >= 0.5 * peak)
    if len(above) < 2:
        return 0.0
    axis = np.asarray(t_ns, dtype=float)
    return float(axis[above[-1]] - axis[above[0]])


@dataclass(slots=True)
class WaveformInput:
    """Inputs needed to synthesize one station trace from FARSim output."""

    station_id: str | int
    peak_ex_uv_m: float
    peak_ey_uv_m: float
    peak_ez_uv_m: float
    rho_shower_m: float
    delta_psi_deg: float
    distance_xmax_km: float
    theta_deg: float
    particle: str = "Iron"
    off_axis_deg: float | None = None
    phi_deg: float | None = None
    shower_phi_deg: float | None = None
    angle_to_b_deg: float | None = None
    t_peak_ns: float = 0.0
    fwhm_ns: float | None = None

    @property
    def peak_vector(self) -> np.ndarray:
        return np.array([self.peak_ex_uv_m, self.peak_ey_uv_m, self.peak_ez_uv_m], dtype=float)

    @property
    def peak_uv_m(self) -> float:
        return float(np.linalg.norm(self.peak_vector))


@dataclass(slots=True)
class WaveformResult:
    station_id: str | int
    t_ns: np.ndarray
    trace_uv_m: np.ndarray
    fwhm_ns: float
    metadata: dict[str, float | str | int]


class WaveformSynthesizer:
    """Generate Ex/Ey/Ez traces from FARSim peak vectors and geometry."""

    def __init__(
        self,
        *,
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
    ) -> None:
        if template not in {
            "asymmetric_tail",
            "bipolar_derivative",
            "pca_trace",
            "mlp_pca_trace",
            "frequency_trace",
            "complex_time",
            "kernel_trace",
            "ring_blend_kernel_trace",
            "component_kernel_trace",
        }:
            raise ValueError(
                "template must be 'asymmetric_tail', 'bipolar_derivative', 'pca_trace', 'mlp_pca_trace', "
                "'frequency_trace', 'complex_time', 'kernel_trace', 'ring_blend_kernel_trace', "
                "or 'component_kernel_trace'."
            )
        self.template = template
        self._pca_model = PCATemplateModel.load(pca_model_path) if template == "pca_trace" else None
        self._mlp_pca_model = MLPPcaTemplateModel.load(mlp_model_path) if template == "mlp_pca_trace" else None
        self._frequency_model = FrequencyTemplateModel.load(frequency_model_path) if template == "frequency_trace" else None
        self._complex_model = ComplexTemplateModel.load(complex_model_path) if template == "complex_time" else None
        self._kernel_model = KernelTemplateModel.load(kernel_model_path) if template == "kernel_trace" else None
        self._ring_blend_base_model = KernelTemplateModel.load() if template == "ring_blend_kernel_trace" else None
        self._ring_blend_ring_model = (
            KernelTemplateModel.load(kernel_model_path or DEFAULT_RING_FOCUSED_KERNEL_TEMPLATE_MODEL)
            if template == "ring_blend_kernel_trace"
            else None
        )
        self._component_kernel_model = (
            ComponentKernelTemplateModel.load(kernel_model_path) if template == "component_kernel_trace" else None
        )
        if t_axis_ns is not None:
            axis = np.asarray(t_axis_ns, dtype=float)
            if axis.ndim != 1 or len(axis) < 2:
                raise ValueError("t_axis_ns must be a one-dimensional array with at least two samples.")
            if np.any(np.diff(axis) <= 0.0):
                raise ValueError("t_axis_ns must be strictly increasing.")
            self.t_ns = axis.copy()
        else:
            if dt_ns <= 0.0:
                raise ValueError("dt_ns must be positive.")
            self.t_ns = np.arange(float(t_start_ns), float(t_stop_ns) + 0.5 * float(dt_ns), float(dt_ns))

    def synthesize_one(self, item: WaveformInput) -> WaveformResult:
        fwhm = (
            float(item.fwhm_ns)
            if item.fwhm_ns is not None
            else estimate_fwhm_ns(
                rho_shower_m=item.rho_shower_m,
                delta_psi_deg=item.delta_psi_deg,
                distance_xmax_km=item.distance_xmax_km,
                theta_deg=item.theta_deg,
                peak_uv_m=item.peak_uv_m,
                particle=item.particle,
                off_axis_deg=item.off_axis_deg,
            )
        )
        if self.template == "component_kernel_trace":
            if self._component_kernel_model is None:
                raise RuntimeError("Component kernel template model is not loaded.")
            trace = self._component_kernel_model.predict_trace(item, self.t_ns)
            shape_for_metadata = trace @ (item.peak_vector / max(item.peak_uv_m, 1e-12))
            fwhm = fwhm
            metadata: dict[str, float | str | int] = {
                "station_id": item.station_id,
                "particle": item.particle,
                "theta_deg": float(item.theta_deg),
                "rho_shower_m": float(item.rho_shower_m),
                "delta_psi_deg": float(item.delta_psi_deg),
                "distance_xmax_km": float(item.distance_xmax_km),
                "peak_uv_m": item.peak_uv_m,
                "fwhm_ns": fwhm,
                "template": self.template,
            }
            if item.off_axis_deg is not None:
                metadata["off_axis_deg"] = float(item.off_axis_deg)
            if item.phi_deg is not None:
                metadata["phi_deg"] = float(item.phi_deg)
            if item.shower_phi_deg is not None:
                metadata["shower_phi_deg"] = float(item.shower_phi_deg)
            if item.angle_to_b_deg is not None:
                metadata["angle_to_b_deg"] = float(item.angle_to_b_deg)
            return WaveformResult(item.station_id, self.t_ns.copy(), trace, estimate_trace_fwhm(self.t_ns, shape_for_metadata), metadata)

        if self.template == "kernel_trace":
            if self._kernel_model is None:
                raise RuntimeError("Kernel template model is not loaded.")
            shape = self._kernel_model.predict_shape(item, self.t_ns)
        elif self.template == "ring_blend_kernel_trace":
            if self._ring_blend_base_model is None or self._ring_blend_ring_model is None:
                raise RuntimeError("Ring-blended kernel template models are not loaded.")
            base_shape = self._ring_blend_base_model.predict_shape(item, self.t_ns)
            ring_shape = self._ring_blend_ring_model.predict_shape(item, self.t_ns)
            ring_weight = 0.5 * np.exp(-0.5 * (abs(float(item.delta_psi_deg)) / 0.25) ** 2)
            shape = normalize_template((1.0 - ring_weight) * base_shape + ring_weight * ring_shape)
        elif self.template == "complex_time":
            if self._complex_model is None:
                raise RuntimeError("Complex time-domain template model is not loaded.")
            shape = self._complex_model.predict_shape(item, self.t_ns)
        elif self.template == "frequency_trace":
            if self._frequency_model is None:
                raise RuntimeError("Frequency template model is not loaded.")
            shape = self._frequency_model.predict_shape(item, self.t_ns)
        elif self.template == "pca_trace":
            if self._pca_model is None:
                raise RuntimeError("PCA template model is not loaded.")
            shape = self._pca_model.predict_shape(item, self.t_ns)
        elif self.template == "mlp_pca_trace":
            if self._mlp_pca_model is None:
                raise RuntimeError("MLP-PCA template model is not loaded.")
            shape = self._mlp_pca_model.predict_shape(item, self.t_ns)
        elif self.template == "bipolar_derivative":
            shape = bipolar_gaussian_derivative(self.t_ns, t_peak_ns=item.t_peak_ns, fwhm_ns=fwhm)
        else:
            params = params_from_geometry(
                fwhm,
                rho_shower_m=item.rho_shower_m,
                delta_psi_deg=item.delta_psi_deg,
                distance_xmax_km=item.distance_xmax_km,
                theta_deg=item.theta_deg,
                peak_uv_m=item.peak_uv_m,
                particle=item.particle,
                off_axis_deg=item.off_axis_deg,
            )
            shape = asymmetric_gaussian_tail(self.t_ns, params, t_peak_ns=item.t_peak_ns)
        shape = normalize_template(shape)
        trace = shape[:, None] * item.peak_vector[None, :]
        metadata: dict[str, float | str | int] = {
            "station_id": item.station_id,
            "particle": item.particle,
            "theta_deg": float(item.theta_deg),
            "rho_shower_m": float(item.rho_shower_m),
            "delta_psi_deg": float(item.delta_psi_deg),
            "distance_xmax_km": float(item.distance_xmax_km),
            "peak_uv_m": item.peak_uv_m,
            "fwhm_ns": fwhm,
            "template": self.template,
        }
        if item.off_axis_deg is not None:
            metadata["off_axis_deg"] = float(item.off_axis_deg)
        if item.phi_deg is not None:
            metadata["phi_deg"] = float(item.phi_deg)
        if item.shower_phi_deg is not None:
            metadata["shower_phi_deg"] = float(item.shower_phi_deg)
        if item.angle_to_b_deg is not None:
            metadata["angle_to_b_deg"] = float(item.angle_to_b_deg)
        return WaveformResult(item.station_id, self.t_ns.copy(), trace, fwhm, metadata)

    def synthesize_many(self, items: Iterable[WaveformInput]) -> list[WaveformResult]:
        return [self.synthesize_one(item) for item in items]


def input_from_mapping(row: dict[str, object]) -> WaveformInput:
    """Build `WaveformInput` from a CSV/DataFrame row-like mapping."""

    return WaveformInput(
        station_id=row.get("station_id", row.get("antenna_idx", "0")),
        peak_ex_uv_m=float(row["peak_ex"]),
        peak_ey_uv_m=float(row["peak_ey"]),
        peak_ez_uv_m=float(row["peak_ez"]),
        rho_shower_m=float(row["rho_shower_m"]),
        delta_psi_deg=float(row["delta_psi_empirical_deg"]),
        distance_xmax_km=float(row["distance_xmax_km"]),
        theta_deg=float(row["theta_deg"]),
        particle=str(row.get("particle", "Iron")),
        off_axis_deg=float(row["off_axis_deg"]) if "off_axis_deg" in row and row["off_axis_deg"] is not None else None,
        phi_deg=float(row["phi_deg"]) if "phi_deg" in row and row["phi_deg"] is not None else None,
        shower_phi_deg=float(row["shower_phi_deg"])
        if "shower_phi_deg" in row and row["shower_phi_deg"] is not None
        else None,
        angle_to_b_deg=float(row["angle_to_b_deg"]) if "angle_to_b_deg" in row and row["angle_to_b_deg"] is not None else None,
        t_peak_ns=float(row.get("t_peak_ns", 0.0)),
        fwhm_ns=float(row["fwhm_ns"]) if "fwhm_ns" in row and row["fwhm_ns"] is not None else None,
    )
