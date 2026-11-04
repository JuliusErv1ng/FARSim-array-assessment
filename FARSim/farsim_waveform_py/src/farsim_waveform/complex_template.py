from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .templates import normalize_template
from .width_model import estimate_fwhm_ns

if TYPE_CHECKING:
    from .synthesizer import WaveformInput


DEFAULT_COMPLEX_TEMPLATE_MODEL = Path(__file__).resolve().parents[2] / "models" / "complex_time_domain_template_model.npz"

BASE_FEATURE_NAMES: tuple[str, ...] = (
    "log10_peak",
    "log1p_rho",
    "inside_delta",
    "outside_delta",
    "abs_delta",
    "delta2",
    "off_axis",
    "off_axis2",
    "ln_distance_xmax",
    "theta90",
    "is_proton",
    "ring_proximity",
)


@dataclass(slots=True)
class ComplexTemplateModel:
    """Analytic multi-scale time-domain model for FARSim waveform synthesis."""

    path: Path
    feature_names: tuple[str, ...]
    feature_mu: np.ndarray
    feature_sd: np.ndarray
    coefficient_beta: np.ndarray
    log_width10_beta: np.ndarray
    n_basis: int
    rbf_centers: np.ndarray | None = None
    rbf_scale: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ComplexTemplateModel":
        model_path = Path(path) if path is not None else DEFAULT_COMPLEX_TEMPLATE_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"Complex template model not found: {model_path}. "
                "Run examples/train_complex_time_domain_template_model.py first."
            )
        data = np.load(model_path, allow_pickle=True)
        return cls(
            path=model_path,
            feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
            feature_mu=np.asarray(data["feature_mu"], dtype=float),
            feature_sd=np.asarray(data["feature_sd"], dtype=float),
            coefficient_beta=np.asarray(data["coefficient_beta"], dtype=float),
            log_width10_beta=np.asarray(data["log_width10_beta"], dtype=float),
            n_basis=int(np.asarray(data["n_basis"]).ravel()[0]),
            rbf_centers=np.asarray(data["rbf_centers"], dtype=float) if "rbf_centers" in data else None,
            rbf_scale=np.asarray(data["rbf_scale"], dtype=float) if "rbf_scale" in data else None,
        )

    def predict_shape(self, item: WaveformInput, t_axis_ns: np.ndarray) -> np.ndarray:
        x = complex_feature_vector_from_input(item, self.feature_names, self.rbf_centers, self.rbf_scale)
        xs = x.copy()
        xs[1:] = (xs[1:] - self.feature_mu) / self.feature_sd
        width10_ns = float(np.exp(xs @ self.log_width10_beta))
        fwhm_ns = estimate_fwhm_ns(
            rho_shower_m=item.rho_shower_m,
            delta_psi_deg=item.delta_psi_deg,
            distance_xmax_km=item.distance_xmax_km,
            theta_deg=item.theta_deg,
            peak_uv_m=item.peak_uv_m,
            particle=item.particle,
            off_axis_deg=item.off_axis_deg,
        )
        coef = xs @ self.coefficient_beta
        basis = build_complex_dictionary(np.asarray(t_axis_ns, dtype=float) - float(item.t_peak_ns), fwhm_ns, width10_ns)
        if basis.shape[1] != self.n_basis:
            raise ValueError(f"Dictionary size mismatch: got {basis.shape[1]}, expected {self.n_basis}.")
        return normalize_template(basis @ coef)


def build_complex_dictionary(t_ns: np.ndarray, fwhm_ns: float, width10_ns: float) -> np.ndarray:
    """Return analytic basis functions for the complex time-domain model."""

    t = np.asarray(t_ns, dtype=float)
    tau0 = max(float(fwhm_ns) / 1.6, 0.35)
    width = max(float(width10_ns), float(fwhm_ns), 1.0)
    columns = [np.ones_like(t)]

    for tau in tau0 * np.array([0.45, 0.75, 1.0, 1.45, 2.2, 3.4]):
        u = t / max(float(tau), 1e-6)
        g = np.exp(-0.5 * u * u)
        columns.extend(
            [
                g,
                u * g,
                (u * u - 1.0) * g,
                (u**3 - 3.0 * u) * g,
                (u**4 - 6.0 * u * u + 3.0) * g,
            ]
        )

    for shift in [-0.8 * tau0, -0.35 * tau0, 0.35 * tau0, 0.8 * tau0, 1.5 * tau0]:
        for sigma in [0.55 * tau0, 1.2 * tau0, 0.28 * width]:
            u = (t - shift) / max(float(sigma), 1e-6)
            columns.append(np.exp(-0.5 * u * u))

    for delay in [0.0, 0.35 * tau0, 0.9 * tau0, 2.0 * tau0, 0.18 * width]:
        x = t - delay
        ramp = smooth_ramp(x, rise=max(0.25 * tau0, 0.15))
        for tau_tail in [1.8 * tau0, 4.0 * tau0, 0.45 * width, 1.2 * width]:
            columns.append(ramp * np.exp(-np.maximum(x, 0.0) / max(float(tau_tail), 1e-6)))

    return np.column_stack(columns)


def smooth_ramp(x: np.ndarray, rise: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ramp = np.zeros_like(x)
    ramp[x >= rise] = 1.0
    mask = (x > 0.0) & (x < rise)
    ramp[mask] = 0.5 * (1.0 - np.cos(np.pi * x[mask] / max(float(rise), 1e-9)))
    return ramp


def complex_feature_vector_from_input(
    item: WaveformInput,
    feature_names: tuple[str, ...],
    rbf_centers: np.ndarray | None = None,
    rbf_scale: np.ndarray | None = None,
) -> np.ndarray:
    values = base_feature_values(
        peak_uv_m=item.peak_uv_m,
        rho_shower_m=item.rho_shower_m,
        delta_psi_deg=item.delta_psi_deg,
        distance_xmax_km=item.distance_xmax_km,
        theta_deg=item.theta_deg,
        particle=item.particle,
        off_axis_deg=item.off_axis_deg,
    )
    return feature_vector_from_values(values, feature_names, rbf_centers=rbf_centers, rbf_scale=rbf_scale)


def base_feature_values(
    *,
    peak_uv_m: float,
    rho_shower_m: float,
    delta_psi_deg: float,
    distance_xmax_km: float,
    theta_deg: float,
    particle: str,
    off_axis_deg: float | None,
) -> dict[str, float]:
    delta = float(delta_psi_deg)
    off = abs(float(off_axis_deg if off_axis_deg is not None else delta))
    return {
        "log10_peak": np.log10(max(float(peak_uv_m), 1e-12)),
        "log1p_rho": np.log1p(max(float(rho_shower_m), 0.0)),
        "inside_delta": max(-delta, 0.0),
        "outside_delta": max(delta, 0.0),
        "abs_delta": abs(delta),
        "delta2": delta * delta,
        "off_axis": off,
        "off_axis2": off * off,
        "ln_distance_xmax": np.log(max(float(distance_xmax_km), 1e-6)),
        "theta90": float(theta_deg) / 90.0,
        "is_proton": 1.0 if str(particle).lower().startswith("proton") else 0.0,
        "ring_proximity": float(np.exp(-((delta / 0.35) ** 2))),
    }


def feature_vector_from_values(
    values: dict[str, float],
    feature_names: tuple[str, ...],
    *,
    rbf_centers: np.ndarray | None = None,
    rbf_scale: np.ndarray | None = None,
) -> np.ndarray:
    rbf_values = None
    if rbf_centers is not None and rbf_scale is not None:
        base = np.array([values[name] for name in BASE_FEATURE_NAMES], dtype=float)
        delta = (base[None, :] - np.asarray(rbf_centers, dtype=float)) / np.asarray(rbf_scale, dtype=float)[None, :]
        rbf_values = np.exp(-0.5 * np.sum(delta * delta, axis=1))
    out = []
    for name in feature_names:
        if name == "1":
            out.append(1.0)
        elif name.startswith("rbf_"):
            if rbf_values is None:
                raise ValueError("RBF feature requested but RBF centers/scales are missing.")
            out.append(float(rbf_values[int(name.split("_", 1)[1])]))
        elif "*" in name:
            left, right = name.split("*", 1)
            out.append(values[left] * values[right])
        else:
            out.append(values[name])
    return np.asarray(out, dtype=float)


def complex_feature_names(n_rbf: int = 0) -> tuple[str, ...]:
    names: list[str] = ["1", *BASE_FEATURE_NAMES]
    interactions = [
        ("log10_peak", "log10_peak"),
        ("log1p_rho", "log1p_rho"),
        ("inside_delta", "inside_delta"),
        ("outside_delta", "outside_delta"),
        ("abs_delta", "log10_peak"),
        ("abs_delta", "log1p_rho"),
        ("off_axis", "theta90"),
        ("theta90", "theta90"),
        ("ln_distance_xmax", "theta90"),
        ("ring_proximity", "log10_peak"),
        ("ring_proximity", "theta90"),
        ("is_proton", "theta90"),
        ("is_proton", "log10_peak"),
    ]
    names.extend(f"{left}*{right}" for left, right in interactions)
    names.extend(f"rbf_{idx}" for idx in range(int(n_rbf)))
    return tuple(names)
