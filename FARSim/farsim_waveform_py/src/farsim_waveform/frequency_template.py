from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .pca_template import feature_vector_from_input
from .templates import normalize_template

if TYPE_CHECKING:
    from .synthesizer import WaveformInput


DEFAULT_FREQUENCY_TEMPLATE_MODEL = Path(__file__).resolve().parents[2] / "models" / "frequency_trace_template_model_raw_1999x05ns.npz"


@dataclass(slots=True)
class FrequencyTemplateModel:
    """Frequency-domain waveform model trained from aligned ZHAireS traces."""

    path: Path
    t_axis_ns: np.ndarray
    mean_vector: np.ndarray
    components: np.ndarray
    feature_names: tuple[str, ...]
    feature_mu: np.ndarray
    feature_sd: np.ndarray
    beta: np.ndarray
    n_time: int

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FrequencyTemplateModel":
        model_path = Path(path) if path is not None else DEFAULT_FREQUENCY_TEMPLATE_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"Frequency template model not found: {model_path}. "
                "Run examples/train_frequency_trace_template_model.py first."
            )
        data = np.load(model_path, allow_pickle=True)
        return cls(
            path=model_path,
            t_axis_ns=np.asarray(data["t_axis_ns"], dtype=float),
            mean_vector=np.asarray(data["mean_vector"], dtype=float),
            components=np.asarray(data["components"], dtype=float),
            feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
            feature_mu=np.asarray(data["feature_mu"], dtype=float),
            feature_sd=np.asarray(data["feature_sd"], dtype=float),
            beta=np.asarray(data["beta"], dtype=float),
            n_time=int(np.asarray(data["n_time"]).ravel()[0]),
        )

    def predict_shape(self, item: WaveformInput, t_axis_ns: np.ndarray) -> np.ndarray:
        features = feature_vector_from_input(item, self.feature_names)
        x = features.copy()
        x[1:] = (x[1:] - self.feature_mu) / self.feature_sd
        scores = x @ self.beta
        vector = self.mean_vector + scores @ self.components
        spectrum = vector_to_spectrum(vector, self.n_time)
        shape = np.fft.irfft(spectrum, n=self.n_time)
        shape = normalize_template(shape)

        target_axis = np.asarray(t_axis_ns, dtype=float)
        source_axis = self.t_axis_ns + float(item.t_peak_ns)
        if len(target_axis) != len(source_axis) or not np.allclose(target_axis, source_axis):
            shape = np.interp(target_axis, source_axis, shape, left=0.0, right=0.0)
            shape = normalize_template(shape)
        return shape


def spectrum_to_vector(spectrum: np.ndarray) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=complex)
    return np.concatenate([spectrum.real, spectrum.imag])


def vector_to_spectrum(vector: np.ndarray, n_time: int) -> np.ndarray:
    n_freq = n_time // 2 + 1
    vector = np.asarray(vector, dtype=float)
    return vector[:n_freq] + 1j * vector[n_freq : 2 * n_freq]
