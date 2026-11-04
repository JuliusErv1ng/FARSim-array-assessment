from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TEMPLATE_PARAMETER_FEATURES: tuple[str, ...] = (
    "1",
    "log1p_rho",
    "log1p_rho2",
    "log10_peak",
    "log10_peak2",
    "abs_delta",
    "abs_delta2",
    "off_axis",
    "off_axis2",
    "theta90",
    "distance100",
    "is_proton",
    "ring_proximity",
    "outside_delta",
    "outside_delta2",
    "outside_logpeak",
)

TEMPLATE_PARAMETER_TARGETS: tuple[str, ...] = (
    "log_sigma_left_over_fwhm",
    "log_sigma_right_over_fwhm",
    "tail_amp",
    "log_tail_delay_over_fwhm",
    "log_tail_tau_over_fwhm",
    "log_tail_rise_over_fwhm",
)

GAUSSIAN_DERIVATIVE_POSITIVE_FWHM_FACTOR = 1.60251721548708

TEMPLATE_PARAMETER_MU = np.array(
    [
        5.490172387851071,
        32.84747957875833,
        2.5567485418432723,
        7.082546652431576,
        0.5927306762568427,
        0.6074564647608831,
        0.9015211828426511,
        1.4417603873195097,
        0.7497318007662843,
        0.5553178549529868,
        0.5005444646098004,
        0.3734451713490916,
        0.36299886637710516,
        0.27329681499421316,
        0.7756261962891358,
    ],
    dtype=float,
)

TEMPLATE_PARAMETER_SD = np.array(
    [
        1.6448363840929872,
        16.705311742866268,
        0.7386362746398791,
        3.6252740759085893,
        0.5060897254291881,
        0.6785552812011758,
        0.7931077759078536,
        1.885237929866256,
        0.1475066149926347,
        0.7672224048812011,
        0.4999997035582156,
        0.4607392080976364,
        0.37620292131128885,
        0.38649955089144983,
        0.8159723242628106,
    ],
    dtype=float,
)

TEMPLATE_PARAMETER_BETA = np.array(
    [
        [
            -0.8802897691374735,
            0.9108630144533986,
            -0.30986643378331485,
            0.23605209451500864,
            0.07077579903460123,
            0.18131435803614385,
            0.07673460033080436,
            -0.5032711187476231,
            -0.304980130220664,
            -0.013670910115192593,
            -0.03910596412279526,
            -0.03099217380840162,
            -0.07123006555205125,
            0.13744488714391348,
            -0.02312308876958286,
            0.1413766972113129,
        ],
        [
            -0.9067887368023131,
            -0.8500298232645429,
            0.1941681815680615,
            -0.602450880240871,
            -0.05696486408264675,
            -0.154692168637656,
            -0.18790199255534312,
            0.7283614410550063,
            0.19038658434567893,
            -0.023769911047825037,
            -0.07796589038824889,
            0.034342462820113925,
            0.11336304746615984,
            0.07524787957381136,
            -0.23489351688791762,
            0.06779370092470983,
        ],
        [
            -0.113677998590444,
            0.027384194132440677,
            0.04725209949078586,
            -0.045547134326676106,
            0.09792295420432262,
            -0.07789103883068046,
            0.028234893017725395,
            -0.12654094246731654,
            0.060844557278979604,
            -0.020355050104004135,
            0.00229350627721222,
            -0.0031459614525661053,
            -0.08612443941210263,
            -0.1405391545907799,
            0.13583716525002867,
            -0.04490964996843512,
        ],
        [
            -1.1433733061742786,
            -0.7289843474840205,
            0.39223174259766447,
            0.03632187364919622,
            -0.6767813435242448,
            -0.9550371850975359,
            0.03713110140366931,
            -1.221714939633829,
            0.7625735018442192,
            -0.4336224514431264,
            0.02870576103842814,
            -0.07110645271565975,
            0.5207565250529032,
            0.14498117962179558,
            0.5650736010263349,
            -0.1092689847262019,
        ],
        [
            2.725166606204382,
            0.39244095852631267,
            0.36954285769708134,
            0.001124628169199512,
            0.4314228494631175,
            -0.29824478625035095,
            0.14920824846354713,
            -0.847528668985286,
            0.46714685164335856,
            -0.16628306465601753,
            -0.025299440621736426,
            -0.05662484275488061,
            0.3377004052225997,
            -0.629671051307962,
            0.6414146500597585,
            -0.15048701335572406,
        ],
        [
            -0.8103030787361668,
            -0.699831974550829,
            1.1196757953926302,
            0.8594395682469308,
            -1.349709083746095,
            -0.5328009007779543,
            -0.31926743215010095,
            -0.060608048571873054,
            0.24380593186791008,
            -0.34701804233374756,
            -0.34876343650339126,
            -0.011948988720250346,
            0.704195216256904,
            1.2765772154868085,
            -0.5624018543809909,
            -0.08358685046308809,
        ],
    ],
    dtype=float,
)

TEMPLATE_PARAMETER_COMPACT_FEATURES: tuple[str, ...] = (
    "1",
    "log10_peak",
    "inside_delta",
    "outside_delta",
    "delta_psi2",
    "ring_proximity",
    "ln_distance_xmax",
    "is_proton",
)

TEMPLATE_PARAMETER_COMPACT_MU = np.array(
    [
        2.5567485418432723,
        0.2070934892852592,
        0.38563718697158184,
        0.6135607214851244,
        0.3734451713490916,
        3.347353121127721,
        0.5005444646098004,
    ],
    dtype=float,
)

TEMPLATE_PARAMETER_COMPACT_SD = np.array(
    [
        0.7386362746398791,
        0.30843302364460906,
        0.5663230362669648,
        0.6785552812011758,
        0.4607392080976364,
        1.1203952024210762,
        0.4999997035582156,
    ],
    dtype=float,
)

TEMPLATE_PARAMETER_COMPACT_BETA = np.array(
    [
        [-0.8802897691374769, 0.8078891382884917, 0.7397988275418493, 1.3337181912564227, -0.5762742613803824, 0.11407529628421764, 0.7160224422462467, -0.05026906155422418],
        [-0.9067887368023096, -0.7996075698359155, -0.5167483668150934, -0.6210131582543077, 0.21416639829206272, -0.20780732759316375, -0.714705631909425, 0.03927135538444083],
        [-0.11367799859044338, 0.006442687558411341, -0.08551026997842925, -0.20260632552873967, 0.08248560065780798, -0.023947995813826985, 0.025965013197754398, -0.0018881453428198143],
        [-1.143373306174266, -1.0131427856062705, -0.33274004907063, -2.597415172808853, 1.7391794662780746, 0.8094892525743466, -0.5715145583483586, -0.11477697806945421],
        [2.7251666062043847, 0.5380937006705658, -0.1261758984459303, -0.05580824512212258, 0.2409095884424653, 0.7803096758500822, 0.5116232184203985, -0.06169645989768174],
        [-0.8103030787361561, 0.2886207876748303, 0.6458949525295966, 2.205271450404977, -1.0599693851405378, 0.8629842753879751, 0.13233595922266134, -0.063896263877094],
    ],
    dtype=float,
)


@dataclass(slots=True)
class AsymmetricGaussianTailParams:
    """Shape parameters for a normalized single-station radio pulse."""

    sigma_left_ns: float
    sigma_right_ns: float
    tail_amp: float = -0.08
    tail_delay_ns: float = 1.0
    tail_tau_ns: float = 20.0
    tail_rise_ns: float = 1.0
    baseline: float = 0.0


def raised_cosine_ramp(x: np.ndarray, rise_ns: float) -> np.ndarray:
    """Causal smooth turn-on with no leakage before the tail starts."""

    x = np.asarray(x, dtype=float)
    rise = max(float(rise_ns), 1e-9)
    ramp = np.zeros_like(x)
    ramp[x >= rise] = 1.0
    mask = (x > 0.0) & (x < rise)
    ramp[mask] = 0.5 * (1.0 - np.cos(np.pi * x[mask] / rise))
    return ramp


def asymmetric_gaussian_tail(
    t_ns: np.ndarray,
    params: AsymmetricGaussianTailParams,
    *,
    t_peak_ns: float = 0.0,
) -> np.ndarray:
    """Return an asymmetric Gaussian with a causal post-peak tail.

    The independent variable is relative time in ns.  The main lobe is allowed
    to have different left and right widths, while the tail is strictly zero
    before `t_peak_ns + tail_delay_ns`; this avoids artificial pre-peak spikes.
    """

    t = np.asarray(t_ns, dtype=float) - float(t_peak_ns)
    sigma_left = max(float(params.sigma_left_ns), 1e-9)
    sigma_right = max(float(params.sigma_right_ns), 1e-9)
    sigma = np.where(t < 0.0, sigma_left, sigma_right)
    main = np.exp(-0.5 * (t / sigma) ** 2)

    tail_x = t - float(params.tail_delay_ns)
    tail = (
        float(params.tail_amp)
        * raised_cosine_ramp(tail_x, params.tail_rise_ns)
        * np.exp(-np.maximum(tail_x, 0.0) / max(float(params.tail_tau_ns), 1e-9))
    )
    return main + tail + float(params.baseline)


def normalize_template(shape: np.ndarray) -> np.ndarray:
    """Normalize a pulse shape to unit maximum absolute amplitude."""

    shape = np.asarray(shape, dtype=float)
    scale = float(np.nanmax(np.abs(shape)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Cannot normalize an empty or zero pulse shape.")
    return shape / scale


def bipolar_gaussian_derivative(
    t_ns: np.ndarray,
    *,
    t_peak_ns: float = 0.0,
    fwhm_ns: float = 5.0,
) -> np.ndarray:
    """Return a zero-crossing bipolar pulse with a calibrated positive FWHM.

    This derivative-like shape is a better first-order RF-chain input than a
    purely unipolar Gaussian when the frequency response matters, because it
    suppresses the artificial low-frequency content produced by single-lobe
    templates.
    """

    tau = max(float(fwhm_ns), 1e-9) / GAUSSIAN_DERIVATIVE_POSITIVE_FWHM_FACTOR
    u = 1.0 + (np.asarray(t_ns, dtype=float) - float(t_peak_ns)) / tau
    return u * np.exp((1.0 - u * u) / 2.0)


def params_from_fwhm(
    fwhm_ns: float,
    *,
    delta_psi_deg: float = 0.0,
    off_axis_deg: float | None = None,
    peak_uv_m: float | None = None,
) -> AsymmetricGaussianTailParams:
    """Map a predicted FWHM to first-pass template parameters.

    This is intentionally conservative: the FWHM is calibrated from data, while
    the detailed tail parameters are a smooth heuristic until a full template
    parameter regression is trained.
    """

    fwhm = float(np.clip(fwhm_ns, 0.4, 80.0))
    sigma_base = fwhm / 2.355
    delta = abs(float(delta_psi_deg))
    off_axis = abs(float(off_axis_deg if off_axis_deg is not None else delta))
    peak = max(float(peak_uv_m if peak_uv_m is not None else 1.0), 1e-9)

    asymmetry = np.clip(1.0 + 0.04 * off_axis, 1.0, 2.4)
    tail_strength = np.clip(0.035 + 0.010 * delta + 0.020 / np.sqrt(peak), 0.02, 0.18)
    return AsymmetricGaussianTailParams(
        sigma_left_ns=max(0.35, 0.70 * sigma_base),
        sigma_right_ns=max(0.45, sigma_base * asymmetry),
        tail_amp=-float(tail_strength),
        tail_delay_ns=max(0.20, 0.35 * fwhm),
        tail_tau_ns=max(4.0, 4.5 * fwhm),
        tail_rise_ns=max(0.20, 0.25 * fwhm),
    )


def params_from_geometry(
    fwhm_ns: float,
    *,
    rho_shower_m: float,
    delta_psi_deg: float,
    distance_xmax_km: float,
    theta_deg: float,
    peak_uv_m: float,
    particle: str = "Iron",
    off_axis_deg: float | None = None,
) -> AsymmetricGaussianTailParams:
    """Predict calibrated template parameters from FARSim geometry.

    The parameter model was trained from 2755 center/ring/outside ZHAireS traces.
    It predicts asymmetric widths and tail parameters as ratios to the compact
    FWHM model.  This keeps the half-height core anchored while allowing the
    low-amplitude width and tail to vary independently.
    """

    fwhm = float(np.clip(fwhm_ns, 0.35, 120.0))
    peak = max(float(peak_uv_m), 1e-12)
    logp = float(np.log10(peak))
    delta = float(delta_psi_deg)
    inside_delta = max(-delta, 0.0)
    outside_delta = max(delta, 0.0)
    ring_proximity = float(np.exp(-((delta / 0.35) ** 2)))
    distance = max(float(distance_xmax_km), 1e-6)
    is_proton = 1.0 if str(particle).lower().startswith("proton") else 0.0
    features = np.array(
        [
            1.0,
            logp,
            inside_delta,
            outside_delta,
            delta * delta,
            ring_proximity,
            np.log(distance),
            is_proton,
        ],
        dtype=float,
    )
    features[1:] = (features[1:] - TEMPLATE_PARAMETER_COMPACT_MU) / TEMPLATE_PARAMETER_COMPACT_SD
    pred = TEMPLATE_PARAMETER_COMPACT_BETA @ features
    sigma_left = fwhm * float(np.exp(pred[0]))
    sigma_right = fwhm * float(np.exp(pred[1]))
    tail_amp = float(pred[2])
    tail_delay = fwhm * float(np.exp(pred[3]))
    tail_tau = fwhm * float(np.exp(pred[4]))
    tail_rise = fwhm * float(np.exp(pred[5]))
    return AsymmetricGaussianTailParams(
        sigma_left_ns=float(np.clip(sigma_left, 0.08, 80.0)),
        sigma_right_ns=float(np.clip(sigma_right, 0.08, 120.0)),
        tail_amp=float(np.clip(tail_amp, -0.50, 0.02)),
        tail_delay_ns=float(np.clip(tail_delay, 0.0, 180.0)),
        tail_tau_ns=float(np.clip(tail_tau, 0.5, 450.0)),
        tail_rise_ns=float(np.clip(tail_rise, 0.05, 50.0)),
    )
