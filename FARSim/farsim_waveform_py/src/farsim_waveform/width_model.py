from __future__ import annotations

import numpy as np


FWHM_MODEL_COEFFICIENTS: dict[str, float] = {
    "1": 7.1226697776796035,
    "log1p_rho_shower_m": -0.20072495999478532,
    "abs_delta_psi_deg": 0.6701162014735041,
    "abs_delta_psi_deg2": -0.004052379426169368,
    "distance_xmax_km": -0.00016952387813239748,
    "theta_deg": -0.016007594445484583,
    "log10_peak": -1.5304536557292705,
    "is_proton": 0.060316969792113234,
}

FWHM_MODEL_V2_FEATURES: tuple[str, ...] = (
    "1",
    "log1p_rho",
    "log1p_rho2",
    "log10_peak",
    "log10_peak2",
    "abs_delta",
    "abs_delta2",
    "abs_delta3",
    "off_axis",
    "off_axis2",
    "theta90",
    "distance100",
    "is_proton",
    "logrho_absdelta",
    "logpeak_absdelta",
    "logrho_logpeak",
    "ring_proximity",
    "outside_delta",
    "outside_delta2",
    "outside_logrho",
    "outside_logpeak",
)

FWHM_MODEL_V2_BETA = np.array(
    [
        0.9901655771955554,
        1.7000476292872388,
        -1.067248578607864,
        1.777441105876866,
        -1.493648216744801,
        -22.529514478915104,
        188.8472073400613,
        -93.3121115298564,
        2.752481150651907,
        -306.6847265682299,
        -0.15745496611902848,
        -0.0123624081677714,
        0.03202742761445272,
        36.36743201745429,
        1.670443569388735,
        -1.161085924542324,
        -0.2921489256705209,
        23.728920081853612,
        206.2476089625303,
        -35.97628662152173,
        -2.304220747803506,
    ],
    dtype=float,
)

FWHM_MODEL_V2_MU = np.array(
    [
        5.776803474353184,
        35.384894105855246,
        2.7224837566331956,
        7.877350786009872,
        0.4371377039846038,
        10.039199849399854,
        1732.0087016809498,
        0.9124098318487931,
        10.863188377604743,
        0.7496074879219911,
        0.640139496786158,
        0.49184782608695654,
        2.88747603367537,
        0.687742849229983,
        15.045792184005544,
        0.5422173362481161,
        0.21768648137032587,
        9.822174530629525,
        1.605964690137295,
        0.1274928908730332,
    ],
    dtype=float,
)

FWHM_MODEL_V2_SD = np.array(
    [
        1.4189558571556629,
        14.378792976631889,
        0.6822264879624058,
        3.3692423841587065,
        3.138169924837397,
        555.4905175139445,
        99021.25738498727,
        3.167127511855171,
        558.2312171712858,
        0.1484030449653528,
        3.0756743283375,
        0.4999335376438182,
        34.1029408461899,
        12.297245162630203,
        3.9322613415018375,
        0.3817881817083365,
        3.1264655965584827,
        553.3141326228883,
        34.02024775124104,
        12.25257363984339,
    ],
    dtype=float,
)

FWHM_MODEL_COMPACT_FEATURES: tuple[str, ...] = (
    "1",
    "inside_delta",
    "outside_delta",
    "delta_psi2",
    "ln_distance_xmax",
    "is_proton",
)

FWHM_MODEL_COMPACT_BETA = np.array(
    [
        1.3622581119014567,
        1.2839276176393095,
        3.2752679412917223,
        -1.8710820351659851,
        0.17587245735502252,
        -0.00035275670619106073,
    ],
    dtype=float,
)

FWHM_MODEL_COMPACT_MU = np.array(
    [
        0.20934018122521858,
        0.3810287885293014,
        0.598006187094954,
        3.368814337085955,
        0.49079189686924496,
    ],
    dtype=float,
)

FWHM_MODEL_COMPACT_SD = np.array(
    [
        0.3090373140006947,
        0.5599070248591983,
        0.6733023166427125,
        1.0889314173493247,
        0.4999152036463124,
    ],
    dtype=float,
)


def estimate_fwhm_ns(
    *,
    rho_shower_m: float,
    delta_psi_deg: float,
    distance_xmax_km: float,
    theta_deg: float,
    peak_uv_m: float,
    particle: str = "Iron",
    off_axis_deg: float | None = None,
    model: str = "compact",
) -> float:
    """Estimate pulse FWHM from FARSim geometry and vector peak amplitude.

    The coefficients are fitted to raw ZHAireS traces from
    `Layout_2024June_920events`.  The model predicts the positive-lobe FWHM in
    ns.  `model="compact"` is the default because it keeps only the physically
    essential variables used in the paper-level parameterization.  `model="v2"`
    preserves the higher-dimensional engineering correction, and `model="v1"`
    preserves the original compact expression for comparison.
    """

    if model == "compact":
        return estimate_fwhm_ns_compact(
            delta_psi_deg=delta_psi_deg,
            distance_xmax_km=distance_xmax_km,
            peak_uv_m=peak_uv_m,
            particle=particle,
        )
    if model == "v2":
        return estimate_fwhm_ns_v2(
            rho_shower_m=rho_shower_m,
            delta_psi_deg=delta_psi_deg,
            distance_xmax_km=distance_xmax_km,
            theta_deg=theta_deg,
            peak_uv_m=peak_uv_m,
            particle=particle,
            off_axis_deg=off_axis_deg,
        )
    if model != "v1":
        raise ValueError(f"Unknown FWHM model: {model}")

    peak = max(float(peak_uv_m), 1e-12)
    abs_delta = abs(float(delta_psi_deg))
    c = FWHM_MODEL_COEFFICIENTS
    log_fwhm = (
        c["1"]
        + c["log1p_rho_shower_m"] * np.log1p(float(rho_shower_m))
        + c["abs_delta_psi_deg"] * abs_delta
        + c["abs_delta_psi_deg2"] * abs_delta * abs_delta
        + c["distance_xmax_km"] * float(distance_xmax_km)
        + c["theta_deg"] * float(theta_deg)
        + c["log10_peak"] * np.log10(peak)
        + c["is_proton"] * (1.0 if str(particle).lower().startswith("proton") else 0.0)
    )
    return float(np.exp(log_fwhm))


def estimate_fwhm_ns_v2(
    *,
    rho_shower_m: float,
    delta_psi_deg: float,
    distance_xmax_km: float,
    theta_deg: float,
    peak_uv_m: float,
    particle: str = "Iron",
    off_axis_deg: float | None = None,
) -> float:
    """Estimate FWHM with the role-aware v2 width model.

    This model adds smooth ring-proximity and outside-ring terms instead of
    applying a hard center/ring/outside category.  It is still a scalar width
    model, so detailed tail parameters should be calibrated separately.
    """

    peak = max(float(peak_uv_m), 1e-12)
    logr = float(np.log1p(max(float(rho_shower_m), 0.0)))
    logp = float(np.log10(peak))
    abs_delta = abs(float(delta_psi_deg))
    off_axis = abs(float(off_axis_deg if off_axis_deg is not None else delta_psi_deg))
    theta90 = float(theta_deg) / 90.0
    distance100 = float(distance_xmax_km) / 100.0
    is_proton = 1.0 if str(particle).lower().startswith("proton") else 0.0
    ring_proximity = float(np.exp(-((abs_delta / 0.35) ** 2)))
    outside_delta = max(abs_delta - 0.35, 0.0)

    features = np.array(
        [
            1.0,
            logr,
            logr * logr,
            logp,
            logp * logp,
            abs_delta,
            abs_delta * abs_delta,
            abs_delta * abs_delta * abs_delta,
            off_axis,
            off_axis * off_axis,
            theta90,
            distance100,
            is_proton,
            logr * abs_delta,
            logp * abs_delta,
            logr * logp,
            ring_proximity,
            outside_delta,
            outside_delta * outside_delta,
            outside_delta * logr,
            outside_delta * logp,
        ],
        dtype=float,
    )
    features[1:] = (features[1:] - FWHM_MODEL_V2_MU) / FWHM_MODEL_V2_SD
    fwhm = float(np.exp(features @ FWHM_MODEL_V2_BETA))
    return float(np.clip(fwhm, 0.35, 120.0))


def estimate_fwhm_ns_compact(
    *,
    delta_psi_deg: float,
    distance_xmax_km: float,
    peak_uv_m: float,
    particle: str = "Iron",
) -> float:
    """Estimate FWHM with the minimal physically interpretable model.

    The model is

    ln(FWHM) = beta0 + beta1 max(-DeltaPsi, 0)
              + beta2 max(DeltaPsi, 0) + beta3 DeltaPsi^2
              + beta4 ln(D_Xmax) + beta5 I_proton,

    after feature standardization.  `DeltaPsi` is still the only angular
    variable; splitting it into inside/outside parts captures the asymmetric
    pulse broadening on the two sides of the Cherenkov ring.  The absolute
    peak field is intentionally not used by this default model, so energy
    scaling in FARSim does not artificially change the pulse width.
    """

    delta = float(delta_psi_deg)
    inside_delta = max(-delta, 0.0)
    outside_delta = max(delta, 0.0)
    distance = max(float(distance_xmax_km), 1e-6)
    is_proton = 1.0 if str(particle).lower().startswith("proton") else 0.0
    features = np.array(
        [
            1.0,
            inside_delta,
            outside_delta,
            delta * delta,
            np.log(distance),
            is_proton,
        ],
        dtype=float,
    )
    features[1:] = (features[1:] - FWHM_MODEL_COMPACT_MU) / FWHM_MODEL_COMPACT_SD
    return float(np.clip(np.exp(features @ FWHM_MODEL_COMPACT_BETA), 0.35, 120.0))
