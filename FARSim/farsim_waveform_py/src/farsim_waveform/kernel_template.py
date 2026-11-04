from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .templates import normalize_template

if TYPE_CHECKING:
    from .synthesizer import WaveformInput


DEFAULT_KERNEL_TEMPLATE_MODEL = Path(
    os.environ.get(
        "FARSIM_KERNEL_TEMPLATE_MODEL",
        Path(__file__).resolve().parents[2]
        / "models"
        / "kernel_trace_template_model_raw_full2500_1999x05ns.npz",
    )
)
DEFAULT_RING_FOCUSED_KERNEL_TEMPLATE_MODEL = (
    Path(
        os.environ.get(
            "FARSIM_RING_FOCUSED_KERNEL_TEMPLATE_MODEL",
            Path(__file__).resolve().parents[2]
            / "models"
            / "kernel_trace_template_model_ring_focused_full2500_1999x05ns.npz",
        )
    )
)

KERNEL_FEATURE_NAMES: tuple[str, ...] = (
    "log10_peak",
    "log1p_rho",
    "off_axis",
    "off_axis2",
    "ln_distance_xmax",
    "theta90",
    "is_proton",
)

RF_AWARE_KERNEL_FEATURE_NAMES: tuple[str, ...] = (
    "log10_peak",
    "log1p_rho",
    "off_axis",
    "off_axis2",
    "delta_psi",
    "abs_delta_psi",
    "delta_psi2",
    "ln_distance_xmax",
    "theta90",
    "phi_sin",
    "phi_cos",
    "shower_phi_sin",
    "shower_phi_cos",
    "sin_alpha",
    "is_proton",
)

COMPONENT_KERNEL_FEATURE_NAMES: tuple[str, ...] = RF_AWARE_KERNEL_FEATURE_NAMES + (
    "component_x",
    "component_y",
    "component_z",
)


@dataclass(slots=True)
class KernelTemplateModel:
    """Non-parametric waveform interpolation model.

    It stores a trained library of normalized traces and interpolates among
    nearest neighbors in FARSim-accessible feature space.  Inference does not
    read ZHAireS trace files; the trace library is the trained artifact.
    """

    path: Path
    t_axis_ns: np.ndarray
    feature_names: tuple[str, ...]
    feature_mu: np.ndarray
    feature_sd: np.ndarray
    train_features_std: np.ndarray
    train_shapes: np.ndarray
    default_neighbors: int
    bandwidth_scale: float

    @classmethod
    def load(cls, path: str | Path | None = None) -> "KernelTemplateModel":
        model_path = Path(path) if path is not None else DEFAULT_KERNEL_TEMPLATE_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"Kernel template model not found: {model_path}. "
                "Run examples/train_kernel_trace_template_model.py first."
            )
        data = np.load(model_path, allow_pickle=True)
        return cls(
            path=model_path,
            t_axis_ns=np.asarray(data["t_axis_ns"], dtype=float),
            feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
            feature_mu=np.asarray(data["feature_mu"], dtype=float),
            feature_sd=np.asarray(data["feature_sd"], dtype=float),
            train_features_std=np.asarray(data["train_features_std"], dtype=float),
            train_shapes=np.asarray(data["train_shapes"], dtype=float),
            default_neighbors=int(np.asarray(data["default_neighbors"]).ravel()[0]),
            bandwidth_scale=float(np.asarray(data["bandwidth_scale"]).ravel()[0]),
        )

    def predict_shape(self, item: WaveformInput, t_axis_ns: np.ndarray, neighbors: int | None = None) -> np.ndarray:
        feature = kernel_feature_vector_from_input(item, self.feature_names)
        feature_std = (feature - self.feature_mu) / self.feature_sd
        d2 = np.sum((self.train_features_std - feature_std[None, :]) ** 2, axis=1)
        k = max(1, min(int(neighbors or self.default_neighbors), len(d2)))
        idx = np.argpartition(d2, k - 1)[:k]
        local_d2 = d2[idx]
        bandwidth = max(float(np.median(local_d2)) * self.bandwidth_scale, 1e-8)
        weights = np.exp(-0.5 * local_d2 / bandwidth)
        weights /= max(float(np.sum(weights)), 1e-12)
        shape = normalize_template(weights @ self.train_shapes[idx])

        target_axis = np.asarray(t_axis_ns, dtype=float)
        source_axis = self.t_axis_ns + float(item.t_peak_ns)
        if len(target_axis) != len(source_axis) or not np.allclose(target_axis, source_axis):
            shape = np.interp(target_axis, source_axis, shape, left=0.0, right=0.0)
            shape = normalize_template(shape)
        return shape


@dataclass(slots=True)
class ComponentKernelTemplateModel:
    """Component-wise non-parametric waveform interpolation model.

    Unlike :class:`KernelTemplateModel`, this model predicts an independent
    normalized pulse shape for each electric-field component.  FARSim still
    supplies the component peak amplitudes, while the trained library supplies
    the component-dependent timing and spectral morphology.
    """

    path: Path
    t_axis_ns: np.ndarray
    feature_names: tuple[str, ...]
    feature_mu: np.ndarray
    feature_sd: np.ndarray
    train_features_std: np.ndarray
    train_shapes: np.ndarray
    default_neighbors: int
    bandwidth_scale: float

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ComponentKernelTemplateModel":
        model_path = Path(path) if path is not None else DEFAULT_KERNEL_TEMPLATE_MODEL
        if not model_path.exists():
            raise FileNotFoundError(f"Component kernel template model not found: {model_path}.")
        data = np.load(model_path, allow_pickle=True)
        return cls(
            path=model_path,
            t_axis_ns=np.asarray(data["t_axis_ns"], dtype=float),
            feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
            feature_mu=np.asarray(data["feature_mu"], dtype=float),
            feature_sd=np.asarray(data["feature_sd"], dtype=float),
            train_features_std=np.asarray(data["train_features_std"], dtype=float),
            train_shapes=np.asarray(data["train_shapes"], dtype=float),
            default_neighbors=int(np.asarray(data["default_neighbors"]).ravel()[0]),
            bandwidth_scale=float(np.asarray(data["bandwidth_scale"]).ravel()[0]),
        )

    def predict_trace(self, item: WaveformInput, t_axis_ns: np.ndarray, neighbors: int | None = None) -> np.ndarray:
        target_axis = np.asarray(t_axis_ns, dtype=float)
        trace = np.empty((len(target_axis), 3), dtype=float)
        peak_vector = item.peak_vector
        for comp in range(3):
            shape = self._predict_component_shape(item, target_axis, comp, neighbors)
            trace[:, comp] = shape * float(peak_vector[comp])
        return trace

    def _predict_component_shape(
        self,
        item: WaveformInput,
        target_axis: np.ndarray,
        component_index: int,
        neighbors: int | None,
    ) -> np.ndarray:
        feature = component_kernel_feature_vector_from_input(item, self.feature_names, component_index)
        feature_std = (feature - self.feature_mu) / self.feature_sd
        d2 = np.sum((self.train_features_std - feature_std[None, :]) ** 2, axis=1)
        k = max(1, min(int(neighbors or self.default_neighbors), len(d2)))
        idx = np.argpartition(d2, k - 1)[:k]
        local_d2 = d2[idx]
        bandwidth = max(float(np.median(local_d2)) * self.bandwidth_scale, 1e-8)
        weights = np.exp(-0.5 * local_d2 / bandwidth)
        weights /= max(float(np.sum(weights)), 1e-12)
        # Component templates are stored as E_component(t) / E_component,peak.
        # Do not renormalize them here; their relative lobe amplitude and
        # component-specific RF-band content are part of the trained model.
        shape = weights @ self.train_shapes[idx]

        source_axis = self.t_axis_ns + float(item.t_peak_ns)
        if len(target_axis) != len(source_axis) or not np.allclose(target_axis, source_axis):
            shape = np.interp(target_axis, source_axis, shape, left=0.0, right=0.0)
        return shape


def kernel_feature_vector_from_input(item: WaveformInput, feature_names: tuple[str, ...]) -> np.ndarray:
    off_axis = abs(float(item.off_axis_deg if item.off_axis_deg is not None else item.delta_psi_deg))
    delta_psi = float(item.delta_psi_deg)
    phi_rad = np.deg2rad(float(item.phi_deg if item.phi_deg is not None else 45.0))
    shower_phi_rad = np.deg2rad(float(item.shower_phi_deg if item.shower_phi_deg is not None else 0.0))
    angle_to_b = float(item.angle_to_b_deg if item.angle_to_b_deg is not None else np.nan)
    if not np.isfinite(angle_to_b):
        angle_to_b = 90.0
    values = {
        "log10_peak": np.log10(max(float(item.peak_uv_m), 1e-12)),
        "log1p_rho": np.log1p(max(float(item.rho_shower_m), 0.0)),
        "off_axis": off_axis,
        "off_axis2": off_axis * off_axis,
        "delta_psi": delta_psi,
        "abs_delta_psi": abs(delta_psi),
        "delta_psi2": delta_psi * delta_psi,
        "ln_distance_xmax": np.log(max(float(item.distance_xmax_km), 1e-6)),
        "theta90": float(item.theta_deg) / 90.0,
        "phi_sin": np.sin(phi_rad),
        "phi_cos": np.cos(phi_rad),
        "shower_phi_sin": np.sin(shower_phi_rad),
        "shower_phi_cos": np.cos(shower_phi_rad),
        "sin_alpha": np.sin(np.deg2rad(angle_to_b)),
        "is_proton": 1.0 if str(item.particle).lower().startswith("proton") else 0.0,
    }
    return np.asarray([values[name] for name in feature_names], dtype=float)


def component_kernel_feature_vector_from_input(
    item: WaveformInput,
    feature_names: tuple[str, ...],
    component_index: int,
) -> np.ndarray:
    values = dict(zip(RF_AWARE_KERNEL_FEATURE_NAMES, kernel_feature_vector_from_input(item, RF_AWARE_KERNEL_FEATURE_NAMES), strict=False))
    values.update(
        {
            "component_x": 1.0 if int(component_index) == 0 else 0.0,
            "component_y": 1.0 if int(component_index) == 1 else 0.0,
            "component_z": 1.0 if int(component_index) == 2 else 0.0,
        }
    )
    return np.asarray([values[name] for name in feature_names], dtype=float)
