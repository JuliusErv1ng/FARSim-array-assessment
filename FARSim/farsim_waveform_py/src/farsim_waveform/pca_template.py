from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .templates import normalize_template

if TYPE_CHECKING:
    from .synthesizer import WaveformInput


DEFAULT_PCA_TEMPLATE_MODEL = Path(__file__).resolve().parents[2] / "models" / "pca_trace_template_model_raw_1999x05ns.npz"


@dataclass(slots=True)
class PCATemplateModel:
    """PCA waveform template model trained from ZHAireS traces.

    The model file is the only training-derived object needed at inference
    time.  It contains the template time axis, mean shape, PCA components, and
    ridge-regression coefficients that map FARSim-accessible geometry variables
    to PCA scores.
    """

    path: Path
    t_axis_ns: np.ndarray
    mean_shape: np.ndarray
    components: np.ndarray
    feature_names: tuple[str, ...]
    feature_mu: np.ndarray
    feature_sd: np.ndarray
    beta: np.ndarray
    spectral_freq_mhz: np.ndarray | None = None
    spectral_gain: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PCATemplateModel":
        model_path = Path(path) if path is not None else DEFAULT_PCA_TEMPLATE_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"PCA template model not found: {model_path}. "
                "Run examples/train_pca_trace_template_model.py first."
            )
        data = np.load(model_path, allow_pickle=True)
        return cls(
            path=model_path,
            t_axis_ns=np.asarray(data["t_axis_ns"], dtype=float),
            mean_shape=np.asarray(data["mean_shape"], dtype=float),
            components=np.asarray(data["components"], dtype=float),
            feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
            feature_mu=np.asarray(data["feature_mu"], dtype=float),
            feature_sd=np.asarray(data["feature_sd"], dtype=float),
            beta=np.asarray(data["beta"], dtype=float),
            spectral_freq_mhz=np.asarray(data["spectral_freq_mhz"], dtype=float) if "spectral_freq_mhz" in data else None,
            spectral_gain=np.asarray(data["spectral_gain"], dtype=float) if "spectral_gain" in data else None,
        )

    def predict_shape(self, item: WaveformInput, t_axis_ns: np.ndarray) -> np.ndarray:
        features = feature_vector_from_input(item, self.feature_names)
        x = features.copy()
        x[1:] = (x[1:] - self.feature_mu) / self.feature_sd
        scores = x @ self.beta
        shape = self.mean_shape + scores @ self.components
        shape = self.apply_spectral_gain(shape)
        shape = normalize_template(shape)

        target_axis = np.asarray(t_axis_ns, dtype=float)
        source_axis = self.t_axis_ns + float(item.t_peak_ns)
        if len(target_axis) != len(source_axis) or not np.allclose(target_axis, source_axis):
            shape = np.interp(target_axis, source_axis, shape, left=0.0, right=0.0)
            shape = normalize_template(shape)
        return shape

    def apply_spectral_gain(self, shape: np.ndarray) -> np.ndarray:
        if self.spectral_gain is None:
            return shape
        y = np.asarray(shape, dtype=float) - float(np.mean(shape))
        gain = np.asarray(self.spectral_gain, dtype=float)
        if len(gain) != len(np.fft.rfft(y)):
            return shape
        filtered = np.fft.irfft(np.fft.rfft(y) * gain, n=len(y))
        return filtered


def feature_vector_from_input(item: WaveformInput, feature_names: tuple[str, ...]) -> np.ndarray:
    values = {
        "1": 1.0,
        "log10_peak": np.log10(max(float(item.peak_uv_m), 1e-12)),
        "log1p_rho": np.log1p(max(float(item.rho_shower_m), 0.0)),
        "inside_delta": max(-float(item.delta_psi_deg), 0.0),
        "outside_delta": max(float(item.delta_psi_deg), 0.0),
        "abs_delta": abs(float(item.delta_psi_deg)),
        "delta2": float(item.delta_psi_deg) ** 2,
        "off_axis": abs(float(item.off_axis_deg if item.off_axis_deg is not None else item.delta_psi_deg)),
        "off_axis2": abs(float(item.off_axis_deg if item.off_axis_deg is not None else item.delta_psi_deg)) ** 2,
        "ln_distance_xmax": np.log(max(float(item.distance_xmax_km), 1e-6)),
        "theta90": float(item.theta_deg) / 90.0,
        "is_proton": 1.0 if str(item.particle).lower().startswith("proton") else 0.0,
    }
    return np.array([values[name] for name in feature_names], dtype=float)
