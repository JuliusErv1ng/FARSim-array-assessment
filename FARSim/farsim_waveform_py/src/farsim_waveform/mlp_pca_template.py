from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from .templates import normalize_template

if TYPE_CHECKING:
    from .synthesizer import WaveformInput


DEFAULT_MLP_PCA_TEMPLATE_MODEL = Path(__file__).resolve().parents[2] / "models" / "mlp_pca_trace_template_model_1999x05ns.joblib"


@dataclass(slots=True)
class MLPPcaTemplateModel:
    """MLP-regressed PCA waveform template model.

    The model predicts only a peak-normalized scalar pulse shape.  FARSim still
    supplies the three-polarization peak vector used to set the final amplitude.
    """

    path: Path
    payload: dict[str, object]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MLPPcaTemplateModel":
        model_path = Path(path) if path is not None else DEFAULT_MLP_PCA_TEMPLATE_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"MLP-PCA template model not found: {model_path}. "
                "Run examples/train_mlp_pca_trace_template_model.py first."
            )
        return cls(path=model_path, payload=joblib.load(model_path))

    @property
    def t_axis_ns(self) -> np.ndarray:
        return np.asarray(self.payload["t_axis_ns"], dtype=float)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self.payload["feature_names"])

    def predict_shape(self, item: WaveformInput, t_axis_ns: np.ndarray) -> np.ndarray:
        feature = feature_vector_from_input(item, self.feature_names).reshape(1, -1)
        scaler = self.payload["feature_scaler"]
        mlp = self.payload["mlp"]
        pca = self.payload["pca"]
        scores = mlp.predict(scaler.transform(feature))
        raw = pca.inverse_transform(scores).reshape(-1)
        if self.payload.get("target_kind") == "time_spectrum_augmented":
            n_time = int(len(self.t_axis_ns))
            time_weight = float(self.payload.get("time_weight", 1.0))
            shape = raw[:n_time] / max(time_weight, 1e-12)
        else:
            shape = raw
        shape = normalize_template(shape)

        target_axis = np.asarray(t_axis_ns, dtype=float)
        source_axis = self.t_axis_ns + float(item.t_peak_ns)
        if len(target_axis) != len(source_axis) or not np.allclose(target_axis, source_axis):
            shape = np.interp(target_axis, source_axis, shape, left=0.0, right=0.0)
            shape = normalize_template(shape)
        return shape


def feature_vector_from_input(item: WaveformInput, feature_names: tuple[str, ...]) -> np.ndarray:
    delta = float(item.delta_psi_deg)
    off_axis = float(item.off_axis_deg if item.off_axis_deg is not None else item.delta_psi_deg)
    phi = float(item.phi_deg if item.phi_deg is not None else 0.0)
    shower_phi = float(item.shower_phi_deg if item.shower_phi_deg is not None else 0.0)
    values = {
        "log10_peak": np.log10(max(float(item.peak_uv_m), 1e-12)),
        "log1p_rho": np.log1p(max(float(item.rho_shower_m), 0.0)),
        "delta_psi": delta,
        "abs_delta_psi": abs(delta),
        "delta_psi2": delta * delta,
        "off_axis": abs(off_axis),
        "off_axis2": abs(off_axis) * abs(off_axis),
        "ln_distance_xmax": np.log(max(float(item.distance_xmax_km), 1e-6)),
        "theta90": float(item.theta_deg) / 90.0,
        "phi_sin": np.sin(np.deg2rad(phi)),
        "phi_cos": np.cos(np.deg2rad(phi)),
        "shower_phi_sin": np.sin(np.deg2rad(shower_phi)),
        "shower_phi_cos": np.cos(np.deg2rad(shower_phi)),
        "is_proton": 1.0 if str(item.particle).lower().startswith("proton") else 0.0,
    }
    return np.asarray([values[name] for name in feature_names], dtype=float)
