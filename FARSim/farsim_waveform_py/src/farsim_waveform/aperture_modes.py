from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator

from farsim_py.efficiency import random_core_positions, trigger_efficiency
from farsim_py.footprint import DEFAULT_DATA_ROOT, FieldFootprint, compute_field_footprint
from farsim_py.io_utils import read_layout_positions
from farsim_py.physics import largest_contour_polygon

from .predictor import FARSimWaveformPredictor
from .rfchain_adapter import RFChainLengthNAdapter
from .digital_filters import adc_filter_metadata, apply_adc_iir_fir, filter_noise_library
from .voltage_trigger import AERAVoltageTriggerConfig, aera_voltage_trigger


ProgressCallback = Callable[[str], None]


@dataclass(slots=True)
class ApertureModeResult:
    mode: str
    layout_name: str
    efficiency: float
    mean_fired_antennas: float
    sampling_radius_m: float
    effective_area_m2: float
    metadata: dict[str, object]
    map_path: str | None = None


@dataclass(slots=True)
class VoltageMap:
    x_axis_m: np.ndarray
    y_axis_m: np.ndarray
    values: np.ndarray
    metadata: dict[str, object]


@dataclass(slots=True)
class AdaptiveVoltageMap(VoltageMap):
    sample_xy_m: np.ndarray | None = None
    sample_values: np.ndarray | None = None


@dataclass(slots=True)
class NoiseVoltageLibrary:
    """Reusable RF-chain ADC-noise traces.

    In the current ``rfchain_lengthN.py`` implementation, galactic noise is an
    all-sky integrated voltage-noise contribution controlled by LST and hardware
    settings, not by shower direction.  The L1 scan therefore samples this
    global library and adds it to direction-dependent signal-only RFChain traces.
    """

    t_ns: np.ndarray
    vadc_uv: np.ndarray
    metadata: dict[str, object]


def field_threshold_aperture(
    *,
    layout_path: str | Path,
    energy_eev: float,
    theta_deg: float,
    phi_deg: float,
    threshold_uv_m: float = 75.0,
    required_antennas: int = 5,
    num_core_samples: int = 5000,
    seed: int = 12345,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    position_efield_cache: str | Path | None = None,
    timing_jitter_ns: float = 5.0,
    progress: ProgressCallback | None = None,
) -> tuple[ApertureModeResult, FieldFootprint, np.ndarray]:
    """Fast field-threshold aperture from the electric-field contour."""

    layout_path = Path(layout_path)
    positions_xy = read_layout_positions(layout_path)
    if progress is not None:
        progress(f"computing field footprint for {layout_path.stem}")
    footprint = compute_field_footprint(
        energy_eev=energy_eev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        data_root=data_root,
        position_efield_cache=position_efield_cache,
    )
    polygon = largest_contour_polygon(footprint.x_m, footprint.y_m, footprint.field_peak_uv_m, threshold_uv_m)
    rng = np.random.default_rng(seed)
    sampling_radius = _contour_sampling_radius(positions_xy, polygon)
    result = trigger_efficiency(
        positions_xy=positions_xy,
        trigger_region_xy=polygon,
        required_antennas=required_antennas,
        num_core_samples=num_core_samples,
        trigger_radius_m=750.0,
        use_polygon_radius=True,
        xmax_xyz=footprint.xmax_xyz,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        timing_jitter_ns=timing_jitter_ns,
        rng=rng,
    )
    return (
        ApertureModeResult(
            mode="field-threshold-aperture",
            layout_name=_layout_name(layout_path),
            efficiency=result.efficiency,
            mean_fired_antennas=result.mean_fired_antennas,
            sampling_radius_m=sampling_radius,
            effective_area_m2=float(np.pi * sampling_radius**2 * result.efficiency),
            metadata={
                "threshold_uv_m": float(threshold_uv_m),
                "required_antennas": int(required_antennas),
                "num_core_samples": int(num_core_samples),
                "energy_eev": float(energy_eev),
                "theta_deg": float(theta_deg),
                "phi_deg": float(phi_deg),
                "position_efield_cache": str(footprint.metadata["position_efield_cache"]),
                "contour_vertices": int(len(polygon)),
            },
        ),
        footprint,
        polygon,
    )


def voltage_threshold_aperture(
    *,
    layout_path: str | Path,
    energy_eev: float,
    theta_deg: float,
    phi_deg: float,
    threshold_uv: float = 5000.0,
    required_antennas: int = 5,
    num_core_samples: int = 1000,
    grid_size: int = 17,
    grid_extent_m: float | None = None,
    adaptive: bool = False,
    adaptive_refine_factor: int = 4,
    adaptive_band_fraction: float = 0.35,
    adaptive_max_refine_points: int = 800,
    seed: int = 12345,
    predictor: FARSimWaveformPredictor | None = None,
    rfchain: RFChainLengthNAdapter | None = None,
    output_npz: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[ApertureModeResult, VoltageMap, np.ndarray]:
    """Fast voltage-threshold aperture using a deterministic Vpeak contour."""

    layout_path = Path(layout_path)
    positions_xy = read_layout_positions(layout_path)
    if adaptive:
        voltage_map = compute_adaptive_voltage_peak_map(
            energy_eev=energy_eev,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            threshold_uv=threshold_uv,
            coarse_grid_size=grid_size,
            grid_extent_m=grid_extent_m,
            refine_factor=adaptive_refine_factor,
            band_fraction=adaptive_band_fraction,
            max_refine_points=adaptive_max_refine_points,
            predictor=predictor,
            rfchain=rfchain,
            progress=progress,
        )
    else:
        voltage_map = compute_voltage_peak_map(
            energy_eev=energy_eev,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            grid_size=grid_size,
            grid_extent_m=grid_extent_m,
            predictor=predictor,
            rfchain=rfchain,
            progress=progress,
        )
    x_grid, y_grid = np.meshgrid(voltage_map.x_axis_m, voltage_map.y_axis_m)
    polygon = largest_contour_polygon(x_grid, y_grid, voltage_map.values, threshold_uv)
    rng = np.random.default_rng(seed)
    sampling_radius = _contour_sampling_radius(positions_xy, polygon)
    field_like_result = trigger_efficiency(
        positions_xy=positions_xy,
        trigger_region_xy=polygon,
        required_antennas=required_antennas,
        num_core_samples=num_core_samples,
        trigger_radius_m=750.0,
        use_polygon_radius=True,
        xmax_xyz=np.array([0.0, 0.0, 1.0], dtype=float),
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        timing_jitter_ns=0.0,
        rng=rng,
    )
    map_path = _save_voltage_map(output_npz, voltage_map)
    return (
        ApertureModeResult(
            mode="voltage-threshold-fast",
            layout_name=_layout_name(layout_path),
            efficiency=field_like_result.efficiency,
            mean_fired_antennas=field_like_result.mean_fired_antennas,
            sampling_radius_m=sampling_radius,
            effective_area_m2=float(np.pi * sampling_radius**2 * field_like_result.efficiency),
            metadata={
                "threshold_uv": float(threshold_uv),
                "required_antennas": int(required_antennas),
                "num_core_samples": int(num_core_samples),
                "grid_size": int(grid_size),
                "grid_extent_m": float(voltage_map.metadata["grid_extent_m"]),
                "adaptive": bool(adaptive),
                "sample_count": int(voltage_map.metadata.get("sample_count", voltage_map.values.size)),
                "contour_vertices": int(len(polygon)),
            },
            map_path=map_path,
        ),
        voltage_map,
        polygon,
    )


def voltage_l1_aperture(
    *,
    layout_path: str | Path,
    energy_eev: float,
    theta_deg: float,
    phi_deg: float,
    trigger_config: AERAVoltageTriggerConfig,
    required_antennas: int = 5,
    num_core_samples: int = 1000,
    noise_trials: int = 8,
    grid_size: int = 13,
    grid_extent_m: float | None = None,
    adaptive: bool = False,
    adaptive_refine_factor: int = 4,
    adaptive_boundary_level: float = 0.5,
    adaptive_boundary_band: float = 0.0,
    adaptive_max_refine_points: int = 800,
    seed: int = 12345,
    predictor: FARSimWaveformPredictor | None = None,
    rfchain: RFChainLengthNAdapter | None = None,
    signal_rfchain: RFChainLengthNAdapter | None = None,
    noise_rfchain: RFChainLengthNAdapter | None = None,
    noise_library: NoiseVoltageLibrary | None = None,
    noise_library_size: int = 16,
    noise_seed: int | None = None,
    signal_gate_rfchain: RFChainLengthNAdapter | None = None,
    require_signal_peak_uv: float | None = None,
    output_npz: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[ApertureModeResult, VoltageMap]:
    """Balanced voltage-L1 aperture from a station-trigger probability map."""

    layout_path = Path(layout_path)
    positions_xy = read_layout_positions(layout_path)
    if adaptive:
        probability_map = compute_adaptive_l1_probability_map(
            energy_eev=energy_eev,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            trigger_config=trigger_config,
            noise_trials=noise_trials,
            coarse_grid_size=grid_size,
            grid_extent_m=grid_extent_m,
            refine_factor=adaptive_refine_factor,
            boundary_level=adaptive_boundary_level,
            boundary_band=adaptive_boundary_band,
            max_refine_points=adaptive_max_refine_points,
            predictor=predictor,
            rfchain=rfchain,
            signal_rfchain=signal_rfchain,
            noise_rfchain=noise_rfchain,
            noise_library=noise_library,
            noise_library_size=noise_library_size,
            noise_seed=noise_seed,
            signal_gate_rfchain=signal_gate_rfchain,
            require_signal_peak_uv=require_signal_peak_uv,
            progress=progress,
        )
    else:
        probability_map = compute_l1_probability_map(
            energy_eev=energy_eev,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            trigger_config=trigger_config,
            noise_trials=noise_trials,
            grid_size=grid_size,
            grid_extent_m=grid_extent_m,
            predictor=predictor,
            rfchain=rfchain,
            signal_rfchain=signal_rfchain,
            noise_rfchain=noise_rfchain,
            noise_library=noise_library,
            noise_library_size=noise_library_size,
            noise_seed=noise_seed,
            signal_gate_rfchain=signal_gate_rfchain,
            require_signal_peak_uv=require_signal_peak_uv,
            progress=progress,
        )
    rng = np.random.default_rng(seed)
    efficiency, mean_fired, sampling_radius = probability_map_array_efficiency(
        positions_xy=positions_xy,
        x_axis_m=probability_map.x_axis_m,
        y_axis_m=probability_map.y_axis_m,
        probability_grid=probability_map.values,
        required_antennas=required_antennas,
        num_core_samples=num_core_samples,
        rng=rng,
    )
    map_path = _save_voltage_map(output_npz, probability_map)
    return (
        ApertureModeResult(
            mode="voltage-l1-balanced",
            layout_name=_layout_name(layout_path),
            efficiency=efficiency,
            mean_fired_antennas=mean_fired,
            sampling_radius_m=sampling_radius,
            effective_area_m2=float(np.pi * sampling_radius**2 * efficiency),
            metadata={
                "required_antennas": int(required_antennas),
                "num_core_samples": int(num_core_samples),
                "noise_trials": int(noise_trials),
                "grid_size": int(grid_size),
                "grid_extent_m": float(probability_map.metadata["grid_extent_m"]),
                "adaptive": bool(adaptive),
                "sample_count": int(probability_map.metadata.get("sample_count", probability_map.values.size)),
                "trigger_config": asdict(trigger_config),
                "signal_noise_split": True,
                "noise_library_size": int(probability_map.metadata.get("noise_library_size", 0)),
                "require_signal_peak_uv": None if require_signal_peak_uv is None else float(require_signal_peak_uv),
            },
            map_path=map_path,
        ),
        probability_map,
    )


def compute_voltage_peak_map(
    *,
    energy_eev: float,
    theta_deg: float,
    phi_deg: float,
    grid_size: int = 17,
    grid_extent_m: float | None = None,
    predictor: FARSimWaveformPredictor | None = None,
    rfchain: RFChainLengthNAdapter | None = None,
    progress: ProgressCallback | None = None,
) -> VoltageMap:
    """Map RF-chain peak voltage on a regular relative-core grid."""

    predictor = predictor or _default_predictor()
    rfchain = rfchain or RFChainLengthNAdapter(noise_galactic=False, noise_inner=False)
    x_axis, y_axis, extent = _analysis_grid(grid_size=grid_size, grid_extent_m=grid_extent_m)
    sample_points = _grid_points(x_axis, y_axis)
    peaks = _evaluate_voltage_points(
        sample_points,
        energy_eev=energy_eev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        predictor=predictor,
        rfchain=rfchain,
        progress=progress,
        label="voltage grid",
    )
    values = peaks.reshape(len(y_axis), len(x_axis))
    return VoltageMap(
        x_axis_m=x_axis,
        y_axis_m=y_axis,
        values=values,
        metadata={
            "quantity": "peak_vadc_uv",
            "energy_eev": float(energy_eev),
            "theta_deg": float(theta_deg),
            "phi_deg": float(phi_deg),
            "grid_size": int(grid_size),
            "grid_extent_m": float(extent),
            "rfchain_noise": False,
        },
    )


def compute_adaptive_voltage_peak_map(
    *,
    energy_eev: float,
    theta_deg: float,
    phi_deg: float,
    threshold_uv: float,
    coarse_grid_size: int = 9,
    grid_extent_m: float | None = None,
    refine_factor: int = 4,
    band_fraction: float = 0.35,
    max_refine_points: int = 800,
    predictor: FARSimWaveformPredictor | None = None,
    rfchain: RFChainLengthNAdapter | None = None,
    progress: ProgressCallback | None = None,
) -> AdaptiveVoltageMap:
    """Compute a Vpeak map by refining only cells near the threshold contour.

    The first pass samples a coarse square grid.  Cells whose corner voltages
    bracket the threshold, or come close to it, are subdivided.  The sampled
    points are then interpolated onto a fine regular grid for contouring.
    """

    if coarse_grid_size < 3:
        raise ValueError("coarse_grid_size must be at least 3.")
    refine_factor = max(2, int(refine_factor))
    band_fraction = max(0.0, float(band_fraction))
    max_refine_points = max(0, int(max_refine_points))

    predictor = predictor or _default_predictor()
    rfchain = rfchain or RFChainLengthNAdapter(noise_galactic=False, noise_inner=False)
    coarse = compute_voltage_peak_map(
        energy_eev=energy_eev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        grid_size=coarse_grid_size,
        grid_extent_m=grid_extent_m,
        predictor=predictor,
        rfchain=rfchain,
        progress=progress,
    )

    refine_points = _adaptive_refine_points(
        coarse.x_axis_m,
        coarse.y_axis_m,
        coarse.values,
        threshold_uv=float(threshold_uv),
        refine_factor=refine_factor,
        band_fraction=band_fraction,
        max_refine_points=max_refine_points,
    )
    if progress is not None:
        progress(f"adaptive voltage refine points {len(refine_points)}")

    sample_points = _grid_points(coarse.x_axis_m, coarse.y_axis_m)
    sample_values = coarse.values.ravel()
    if len(refine_points):
        refined_values = _evaluate_voltage_points(
            refine_points,
            energy_eev=energy_eev,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            predictor=predictor,
            rfchain=rfchain,
            progress=progress,
            label="adaptive refine",
        )
        sample_points = np.vstack((sample_points, refine_points))
        sample_values = np.concatenate((sample_values, refined_values))

    fine_size = (coarse_grid_size - 1) * refine_factor + 1
    x_fine = np.linspace(float(coarse.x_axis_m[0]), float(coarse.x_axis_m[-1]), fine_size)
    y_fine = np.linspace(float(coarse.y_axis_m[0]), float(coarse.y_axis_m[-1]), fine_size)
    x_grid, y_grid = np.meshgrid(x_fine, y_fine)
    linear = LinearNDInterpolator(sample_points, sample_values, fill_value=np.nan)
    nearest = NearestNDInterpolator(sample_points, sample_values)
    values = np.asarray(linear(x_grid, y_grid), dtype=float)
    missing = ~np.isfinite(values)
    if np.any(missing):
        values[missing] = nearest(x_grid[missing], y_grid[missing])

    metadata = {
        **coarse.metadata,
        "quantity": "peak_vadc_uv_adaptive",
        "adaptive": True,
        "coarse_grid_size": int(coarse_grid_size),
        "refine_factor": int(refine_factor),
        "fine_grid_size": int(fine_size),
        "band_fraction": float(band_fraction),
        "threshold_uv": float(threshold_uv),
        "refine_points": int(len(refine_points)),
        "sample_count": int(len(sample_values)),
    }
    return AdaptiveVoltageMap(
        x_axis_m=x_fine,
        y_axis_m=y_fine,
        values=values,
        metadata=metadata,
        sample_xy_m=sample_points,
        sample_values=sample_values,
    )


def compute_l1_probability_map(
    *,
    energy_eev: float,
    theta_deg: float,
    phi_deg: float,
    trigger_config: AERAVoltageTriggerConfig,
    noise_trials: int = 8,
    grid_size: int = 13,
    grid_extent_m: float | None = None,
    predictor: FARSimWaveformPredictor | None = None,
    rfchain: RFChainLengthNAdapter | None = None,
    signal_rfchain: RFChainLengthNAdapter | None = None,
    noise_rfchain: RFChainLengthNAdapter | None = None,
    noise_library: NoiseVoltageLibrary | None = None,
    noise_library_size: int = 16,
    noise_seed: int | None = None,
    signal_gate_rfchain: RFChainLengthNAdapter | None = None,
    require_signal_peak_uv: float | None = None,
    progress: ProgressCallback | None = None,
) -> VoltageMap:
    """Map station-level L1 probability using signal-only RFChain plus noise library."""

    predictor = predictor or _default_predictor()
    signal_rfchain = _signal_rfchain_from_args(rfchain=rfchain, signal_rfchain=signal_rfchain)
    if noise_library is None and noise_library_size > 0:
        noise_rfchain = noise_rfchain or _noise_rfchain_from_args(rfchain=rfchain)
    l1_rng = np.random.default_rng(noise_seed)
    x_axis, y_axis, extent = _analysis_grid(grid_size=grid_size, grid_extent_m=grid_extent_m)
    sample_points = _grid_points(x_axis, y_axis)
    values = _evaluate_l1_points(
        sample_points,
        energy_eev=energy_eev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        trigger_config=trigger_config,
        noise_trials=noise_trials,
        predictor=predictor,
        signal_rfchain=signal_rfchain,
        noise_rfchain=noise_rfchain,
        noise_library=noise_library,
        noise_library_size=noise_library_size,
        noise_rng=l1_rng,
        require_signal_peak_uv=require_signal_peak_uv,
        progress=progress,
        label="L1 probability grid",
    ).reshape(len(y_axis), len(x_axis))
    return VoltageMap(
        x_axis_m=x_axis,
        y_axis_m=y_axis,
        values=values,
        metadata={
            "quantity": "station_l1_probability",
            "energy_eev": float(energy_eev),
            "theta_deg": float(theta_deg),
            "phi_deg": float(phi_deg),
            "grid_size": int(grid_size),
            "grid_extent_m": float(extent),
            "noise_trials": int(noise_trials),
            "trigger_config": asdict(trigger_config),
            "adc_digital_filter": adc_filter_metadata(),
            "trigger_channel": None if trigger_config.trigger_channel is None else int(trigger_config.trigger_channel),
            "signal_noise_split": True,
            "rfchain_noise": False,
            "noise_library_size": int(0 if noise_library is None else len(noise_library.vadc_uv))
            if noise_library is not None
            else int(max(0, noise_library_size)),
            "require_signal_peak_uv": None if require_signal_peak_uv is None else float(require_signal_peak_uv),
        },
    )


def compute_adaptive_l1_probability_map(
    *,
    energy_eev: float,
    theta_deg: float,
    phi_deg: float,
    trigger_config: AERAVoltageTriggerConfig,
    noise_trials: int = 8,
    coarse_grid_size: int = 7,
    grid_extent_m: float | None = None,
    refine_factor: int = 4,
    boundary_level: float = 0.5,
    boundary_band: float = 0.0,
    max_refine_points: int = 800,
    predictor: FARSimWaveformPredictor | None = None,
    rfchain: RFChainLengthNAdapter | None = None,
    signal_rfchain: RFChainLengthNAdapter | None = None,
    noise_rfchain: RFChainLengthNAdapter | None = None,
    noise_library: NoiseVoltageLibrary | None = None,
    noise_library_size: int = 16,
    noise_seed: int | None = None,
    signal_gate_rfchain: RFChainLengthNAdapter | None = None,
    require_signal_peak_uv: float | None = None,
    progress: ProgressCallback | None = None,
) -> AdaptiveVoltageMap:
    """Adaptive station-level L1 probability map.

    Coarse cells are refined only when their corner trigger probabilities cross
    the chosen boundary level, or lie close to it.  For one noise trial the
    map is binary at sampled points; interpolation only estimates the boundary
    location used by the array-level Monte Carlo.
    """

    if coarse_grid_size < 3:
        raise ValueError("coarse_grid_size must be at least 3.")
    refine_factor = max(2, int(refine_factor))
    max_refine_points = max(0, int(max_refine_points))
    boundary_level = float(np.clip(boundary_level, 0.0, 1.0))
    boundary_band = max(0.0, float(boundary_band))

    predictor = predictor or _default_predictor()
    signal_rfchain = _signal_rfchain_from_args(rfchain=rfchain, signal_rfchain=signal_rfchain)
    if noise_library is None and noise_library_size > 0:
        noise_rfchain = noise_rfchain or _noise_rfchain_from_args(rfchain=rfchain)
    l1_rng = np.random.default_rng(noise_seed)
    coarse = compute_l1_probability_map(
        energy_eev=energy_eev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        trigger_config=trigger_config,
        noise_trials=noise_trials,
        grid_size=coarse_grid_size,
        grid_extent_m=grid_extent_m,
        predictor=predictor,
        signal_rfchain=signal_rfchain,
        noise_rfchain=noise_rfchain,
        noise_library=noise_library,
        noise_library_size=noise_library_size,
        noise_seed=None if noise_seed is None else int(l1_rng.integers(0, np.iinfo(np.uint32).max)),
        require_signal_peak_uv=require_signal_peak_uv,
        progress=progress,
    )

    refine_points = _adaptive_probability_refine_points(
        coarse.x_axis_m,
        coarse.y_axis_m,
        coarse.values,
        boundary_level=boundary_level,
        boundary_band=boundary_band,
        refine_factor=refine_factor,
        max_refine_points=max_refine_points,
    )
    if progress is not None:
        progress(f"adaptive L1 refine points {len(refine_points)}")

    sample_points = _grid_points(coarse.x_axis_m, coarse.y_axis_m)
    sample_values = coarse.values.ravel()
    if len(refine_points):
        refined_values = _evaluate_l1_points(
            refine_points,
            energy_eev=energy_eev,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            trigger_config=trigger_config,
            noise_trials=noise_trials,
            predictor=predictor,
            signal_rfchain=signal_rfchain,
            noise_rfchain=noise_rfchain,
            noise_library=noise_library,
            noise_library_size=noise_library_size,
            noise_rng=l1_rng,
            require_signal_peak_uv=require_signal_peak_uv,
            progress=progress,
            label="adaptive L1 refine",
        )
        sample_points = np.vstack((sample_points, refine_points))
        sample_values = np.concatenate((sample_values, refined_values))

    fine_size = (coarse_grid_size - 1) * refine_factor + 1
    x_fine = np.linspace(float(coarse.x_axis_m[0]), float(coarse.x_axis_m[-1]), fine_size)
    y_fine = np.linspace(float(coarse.y_axis_m[0]), float(coarse.y_axis_m[-1]), fine_size)
    x_grid, y_grid = np.meshgrid(x_fine, y_fine)
    linear = LinearNDInterpolator(sample_points, sample_values, fill_value=np.nan)
    nearest = NearestNDInterpolator(sample_points, sample_values)
    values = np.asarray(linear(x_grid, y_grid), dtype=float)
    missing = ~np.isfinite(values)
    if np.any(missing):
        values[missing] = nearest(x_grid[missing], y_grid[missing])
    values = np.clip(values, 0.0, 1.0)

    metadata = {
        **coarse.metadata,
        "quantity": "station_l1_probability_adaptive",
        "adaptive": True,
        "coarse_grid_size": int(coarse_grid_size),
        "refine_factor": int(refine_factor),
        "fine_grid_size": int(fine_size),
        "boundary_level": float(boundary_level),
        "boundary_band": float(boundary_band),
        "refine_points": int(len(refine_points)),
        "sample_count": int(len(sample_values)),
    }
    return AdaptiveVoltageMap(
        x_axis_m=x_fine,
        y_axis_m=y_fine,
        values=values,
        metadata=metadata,
        sample_xy_m=sample_points,
        sample_values=sample_values,
    )


def _evaluate_voltage_points(
    xy_m: np.ndarray,
    *,
    energy_eev: float,
    theta_deg: float,
    phi_deg: float,
    predictor: FARSimWaveformPredictor,
    rfchain: RFChainLengthNAdapter,
    progress: ProgressCallback | None,
    label: str,
) -> np.ndarray:
    xy_m = np.asarray(xy_m, dtype=float)
    total = len(xy_m)
    if total == 0:
        return np.empty(0, dtype=float)
    if progress is not None:
        progress(f"{label} waveform batch {total}")
    predictions = predictor.predict_fixed_shower_points(
        energy_eev=energy_eev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        xy_m=xy_m,
        t_peak_ns=0.0,
    )
    time_axis = predictions[0].t_ns
    traces = np.stack([prediction.trace_uv_m for prediction in predictions], axis=0)
    if progress is not None:
        progress(f"{label} RF-chain batch peak voltage for {len(predictions)} traces")
    try:
        return rfchain.forward_traces_peak(
            t_ns=time_axis,
            electric_traces_uv_m=traces,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
        )
    except ValueError:
        return np.array(
            [
                rfchain.forward_trace(
                    t_ns=time_axis,
                    electric_trace_uv_m=trace,
                    theta_deg=theta_deg,
                    phi_deg=phi_deg,
                ).peak_voltage_uv
                for trace in traces
            ],
            dtype=float,
        )


def _evaluate_l1_points(
    xy_m: np.ndarray,
    *,
    energy_eev: float,
    theta_deg: float,
    phi_deg: float,
    trigger_config: AERAVoltageTriggerConfig,
    noise_trials: int,
    predictor: FARSimWaveformPredictor,
    signal_rfchain: RFChainLengthNAdapter,
    noise_rfchain: RFChainLengthNAdapter | None,
    noise_library: NoiseVoltageLibrary | None,
    noise_library_size: int,
    noise_rng: np.random.Generator,
    require_signal_peak_uv: float | None,
    progress: ProgressCallback | None,
    label: str,
) -> np.ndarray:
    xy_m = np.asarray(xy_m, dtype=float)
    values = np.zeros(len(xy_m), dtype=float)
    if len(xy_m) == 0:
        return values

    if progress is not None:
        progress(f"{label} waveform batch {len(xy_m)}")
    predictions = predictor.predict_fixed_shower_points(
        energy_eev=energy_eev,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        xy_m=xy_m,
        t_peak_ns=0.0,
    )
    time_axis = predictions[0].t_ns
    traces = np.stack([prediction.trace_uv_m for prediction in predictions], axis=0)

    if noise_library is None and noise_library_size > 0:
        if noise_rfchain is None:
            raise ValueError("noise_rfchain is required when noise_library is not supplied and noise_library_size > 0.")
        if progress is not None:
            progress(f"{label} building RF-chain noise library ({noise_library_size})")
        noise_library = build_adc_noise_library(
            t_ns=time_axis,
            rfchain=noise_rfchain,
            size=noise_library_size,
        )
    if noise_library is not None and "adc_digital_filter" not in noise_library.metadata:
        noise_library = filter_noise_library(noise_library)

    if progress is not None:
        progress(f"{label} RF-chain signal batch {len(xy_m)}")
    signal_groups = signal_rfchain.forward_traces_vadc_groups(
        t_ns=time_axis,
        electric_traces_uv_m=traces,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
    )

    trials = max(1, int(noise_trials))
    for group_index, (trace_indices, t_group, signal_vadc) in enumerate(signal_groups, start=1):
        if progress is not None:
            progress(f"{label} trigger group {group_index}/{len(signal_groups)} ({len(trace_indices)} traces)")
        signal_vadc_l1 = apply_adc_iir_fir(t_group, signal_vadc)
        if trigger_config.trigger_channel is None:
            signal_peaks = np.max(np.abs(signal_vadc_l1), axis=(1, 2))
        else:
            channel = int(trigger_config.trigger_channel)
            signal_peaks = np.max(np.abs(signal_vadc_l1[:, :, channel]), axis=1)
        active = np.ones(len(trace_indices), dtype=bool)
        if require_signal_peak_uv is not None:
            active &= signal_peaks >= float(require_signal_peak_uv)
        if not np.any(active):
            continue

        triggered = np.zeros(len(trace_indices), dtype=int)
        for trial_index in range(trials):
            for local_index in np.flatnonzero(active):
                noise = _draw_noise_segment(
                    noise_library,
                    n_samples=signal_vadc_l1.shape[1],
                    rng=noise_rng,
                )
                combined = signal_vadc_l1[local_index] if noise is None else signal_vadc_l1[local_index] + noise
                decision = aera_voltage_trigger(t_group, combined, trigger_config)
                triggered[local_index] += int(decision.triggered)
        values[trace_indices] = triggered / trials
    return values


def build_adc_noise_library(
    *,
    t_ns: np.ndarray,
    rfchain: RFChainLengthNAdapter,
    size: int = 16,
    theta_deg: float = 75.0,
    phi_deg: float = 45.0,
) -> NoiseVoltageLibrary:
    """Generate reusable noise-only ADC traces with the existing RFChain code."""

    size = int(size)
    if size <= 0:
        return NoiseVoltageLibrary(
            t_ns=np.asarray(t_ns, dtype=float).copy(),
            vadc_uv=np.empty((0, len(np.asarray(t_ns).reshape(-1)), 3), dtype=float),
            metadata={"size": 0, "noise_galactic": bool(rfchain.noise_galactic), "noise_inner": bool(rfchain.noise_inner)},
        )
    time = np.asarray(t_ns, dtype=float).reshape(-1)
    zero_trace = np.zeros((time.size, 3), dtype=float)
    traces = []
    t_out: np.ndarray | None = None
    for _ in range(size):
        result = rfchain.forward_trace(
            t_ns=time,
            electric_trace_uv_m=zero_trace,
            theta_deg=float(theta_deg),
            phi_deg=float(phi_deg),
        )
        t_out = result.t_ns if t_out is None else t_out
        traces.append(np.asarray(result.vadc_uv, dtype=float))
    return NoiseVoltageLibrary(
        t_ns=np.asarray(t_out, dtype=float),
        vadc_uv=np.stack(traces, axis=0),
        metadata={
            "size": int(size),
            "theta_deg_dummy": float(theta_deg),
            "phi_deg_dummy": float(phi_deg),
            "noise_galactic": bool(rfchain.noise_galactic),
            "noise_inner": bool(rfchain.noise_inner),
            "lst_hour": int(rfchain.lst_hour),
            "lst_minute": int(rfchain.lst_minute),
        },
    )


def _draw_noise_segment(
    noise_library: NoiseVoltageLibrary | None,
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if noise_library is None or noise_library.vadc_uv.shape[0] == 0:
        return None
    traces = np.asarray(noise_library.vadc_uv, dtype=float)
    trace = traces[int(rng.integers(0, traces.shape[0]))]
    if trace.shape[0] == n_samples:
        return trace
    if trace.shape[0] > n_samples:
        start = int(rng.integers(0, trace.shape[0] - n_samples + 1))
        return trace[start : start + n_samples]
    repeats = int(np.ceil(n_samples / trace.shape[0]))
    return np.tile(trace, (repeats, 1))[:n_samples]


def _signal_rfchain_from_args(
    *,
    rfchain: RFChainLengthNAdapter | None,
    signal_rfchain: RFChainLengthNAdapter | None,
) -> RFChainLengthNAdapter:
    if signal_rfchain is not None:
        if signal_rfchain.noise_galactic or signal_rfchain.noise_inner:
            raise ValueError("signal_rfchain must be configured with noise_galactic=False and noise_inner=False.")
        return signal_rfchain
    if rfchain is not None and not (rfchain.noise_galactic or rfchain.noise_inner):
        return rfchain
    if rfchain is not None:
        return RFChainLengthNAdapter(
            module_path=rfchain.module_path,
            backend=rfchain.backend,
            params_dir=rfchain.params_dir,
            noise_galactic=False,
            noise_inner=False,
            lst_hour=rfchain.lst_hour,
            lst_minute=rfchain.lst_minute,
            auto_drop_nyquist=rfchain.auto_drop_nyquist,
        )
    return RFChainLengthNAdapter(noise_galactic=False, noise_inner=False)


def _noise_rfchain_from_args(*, rfchain: RFChainLengthNAdapter | None) -> RFChainLengthNAdapter:
    if rfchain is not None and (rfchain.noise_galactic or rfchain.noise_inner):
        return rfchain
    if rfchain is not None:
        return RFChainLengthNAdapter(
            module_path=rfchain.module_path,
            backend=rfchain.backend,
            params_dir=rfchain.params_dir,
            noise_galactic=True,
            noise_inner=True,
            lst_hour=rfchain.lst_hour,
            lst_minute=rfchain.lst_minute,
            auto_drop_nyquist=rfchain.auto_drop_nyquist,
            seed=rfchain.seed,
        )
    return RFChainLengthNAdapter(noise_galactic=True, noise_inner=True)


def _grid_points(x_axis_m: np.ndarray, y_axis_m: np.ndarray) -> np.ndarray:
    x_grid, y_grid = np.meshgrid(x_axis_m, y_axis_m)
    return np.column_stack((x_grid.ravel(), y_grid.ravel()))


def _adaptive_refine_points(
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    values: np.ndarray,
    *,
    threshold_uv: float,
    refine_factor: int,
    band_fraction: float,
    max_refine_points: int,
) -> np.ndarray:
    x_axis_m = np.asarray(x_axis_m, dtype=float)
    y_axis_m = np.asarray(y_axis_m, dtype=float)
    values = np.asarray(values, dtype=float)
    threshold = float(threshold_uv)
    band_width = max(abs(threshold) * band_fraction, 1.0)
    existing = {(_round_key(x), _round_key(y)) for x, y in _grid_points(x_axis_m, y_axis_m)}
    points: list[tuple[float, float, float]] = []

    for iy in range(len(y_axis_m) - 1):
        for ix in range(len(x_axis_m) - 1):
            corners = values[iy : iy + 2, ix : ix + 2].ravel()
            if not np.isfinite(corners).any():
                continue
            v_min = float(np.nanmin(corners))
            v_max = float(np.nanmax(corners))
            brackets = v_min <= threshold <= v_max
            close = min(abs(v_min - threshold), abs(v_max - threshold)) <= band_width
            if not brackets and not close:
                continue
            x_local = np.linspace(x_axis_m[ix], x_axis_m[ix + 1], refine_factor + 1)
            y_local = np.linspace(y_axis_m[iy], y_axis_m[iy + 1], refine_factor + 1)
            cell_center_x = 0.5 * (x_axis_m[ix] + x_axis_m[ix + 1])
            cell_center_y = 0.5 * (y_axis_m[iy] + y_axis_m[iy + 1])
            for y in y_local:
                for x in x_local:
                    key = (_round_key(x), _round_key(y))
                    if key in existing:
                        continue
                    existing.add(key)
                    distance = float(np.hypot(x - cell_center_x, y - cell_center_y))
                    points.append((distance, float(x), float(y)))

    if not points:
        return np.empty((0, 2), dtype=float)
    points.sort(key=lambda item: item[0])
    if max_refine_points:
        points = points[:max_refine_points]
    return np.asarray([(x, y) for _distance, x, y in points], dtype=float)


def _adaptive_probability_refine_points(
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    values: np.ndarray,
    *,
    boundary_level: float,
    boundary_band: float,
    refine_factor: int,
    max_refine_points: int,
) -> np.ndarray:
    x_axis_m = np.asarray(x_axis_m, dtype=float)
    y_axis_m = np.asarray(y_axis_m, dtype=float)
    values = np.asarray(values, dtype=float)
    level = float(np.clip(boundary_level, 0.0, 1.0))
    band = max(0.0, float(boundary_band))
    existing = {(_round_key(x), _round_key(y)) for x, y in _grid_points(x_axis_m, y_axis_m)}
    points: list[tuple[float, float, float, float]] = []

    for iy in range(len(y_axis_m) - 1):
        for ix in range(len(x_axis_m) - 1):
            corners = values[iy : iy + 2, ix : ix + 2].ravel()
            if not np.isfinite(corners).any():
                continue
            v_min = float(np.nanmin(corners))
            v_max = float(np.nanmax(corners))
            crosses = v_min <= level <= v_max and v_max > v_min
            close = min(abs(v_min - level), abs(v_max - level)) <= band
            if not crosses and not close:
                continue
            x_local = np.linspace(x_axis_m[ix], x_axis_m[ix + 1], refine_factor + 1)
            y_local = np.linspace(y_axis_m[iy], y_axis_m[iy + 1], refine_factor + 1)
            cell_center_x = 0.5 * (x_axis_m[ix] + x_axis_m[ix + 1])
            cell_center_y = 0.5 * (y_axis_m[iy] + y_axis_m[iy + 1])
            score = abs(0.5 * (v_min + v_max) - level)
            for y in y_local:
                for x in x_local:
                    key = (_round_key(x), _round_key(y))
                    if key in existing:
                        continue
                    existing.add(key)
                    distance = float(np.hypot(x - cell_center_x, y - cell_center_y))
                    points.append((score, distance, float(x), float(y)))

    if not points:
        return np.empty((0, 2), dtype=float)
    points.sort(key=lambda item: (item[0], item[1]))
    if max_refine_points:
        points = points[:max_refine_points]
    return np.asarray([(x, y) for _score, _distance, x, y in points], dtype=float)


def _round_key(value: float) -> int:
    return int(round(float(value) * 1.0e6))


def probability_map_array_efficiency(
    *,
    positions_xy: np.ndarray,
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    probability_grid: np.ndarray,
    required_antennas: int,
    num_core_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Array trigger probability from a relative-core station probability map."""

    active = np.argwhere(probability_grid > 1e-3)
    if active.size:
        active_x = x_axis_m[active[:, 1]]
        active_y = y_axis_m[active[:, 0]]
        probability_radius = float(np.max(np.hypot(active_x, active_y)))
    else:
        probability_radius = 0.5 * float(max(np.ptp(x_axis_m), np.ptp(y_axis_m)))
    layout_radius = float(np.max(np.linalg.norm(positions_xy, axis=1)))
    sampling_radius = probability_radius + layout_radius
    centers = random_core_positions(sampling_radius, num_core_samples, rng)

    interpolator = RegularGridInterpolator(
        (np.asarray(y_axis_m, dtype=float), np.asarray(x_axis_m, dtype=float)),
        np.asarray(probability_grid, dtype=float),
        bounds_error=False,
        fill_value=0.0,
    )
    array_probabilities = np.zeros(num_core_samples, dtype=float)
    mean_fired = np.zeros(num_core_samples, dtype=float)
    for idx, center in enumerate(centers):
        relative = positions_xy - center
        station_probabilities = interpolator(np.column_stack((relative[:, 1], relative[:, 0])))
        station_probabilities = np.clip(station_probabilities, 0.0, 1.0)
        mean_fired[idx] = float(np.sum(station_probabilities))
        array_probabilities[idx] = poisson_binomial_at_least(station_probabilities, required_antennas)
    return float(np.mean(array_probabilities)), float(np.mean(mean_fired)), float(sampling_radius)


def poisson_binomial_at_least(probabilities: np.ndarray, required: int) -> float:
    """Probability that at least ``required`` Bernoulli trials fire."""

    probs = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if required <= 0:
        return 1.0
    if required > probs.size:
        return 0.0
    miss = np.zeros(required, dtype=float)
    miss[0] = 1.0
    for prob in probs:
        updated = miss * (1.0 - prob)
        updated[1:] += miss[:-1] * prob
        miss = updated
    return float(np.clip(1.0 - np.sum(miss), 0.0, 1.0))


def _default_predictor() -> FARSimWaveformPredictor:
    t_axis_ns = (np.arange(1999, dtype=float) - 999.0) * 0.5
    return FARSimWaveformPredictor(template="ring_blend_kernel_trace", t_axis_ns=t_axis_ns)


def _analysis_grid(*, grid_size: int, grid_extent_m: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    extent = float(grid_extent_m) if grid_extent_m is not None else 2200.0
    axis = np.linspace(-extent, extent, int(grid_size))
    return axis, axis.copy(), extent


def _contour_sampling_radius(positions_xy: np.ndarray, polygon: np.ndarray) -> float:
    if polygon.size == 0:
        return float(np.max(np.linalg.norm(positions_xy, axis=1)))
    return float(np.max(np.linalg.norm(polygon, axis=1)) + np.max(np.linalg.norm(positions_xy, axis=1)))


def _layout_name(layout_path: Path) -> str:
    stem = layout_path.stem
    if stem == "gp80_hybrid_position_meters":
        return "hybrid"
    return stem


def _save_voltage_map(output_npz: str | Path | None, voltage_map: VoltageMap) -> str | None:
    if output_npz is None:
        return None
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x_axis_m": voltage_map.x_axis_m,
        "y_axis_m": voltage_map.y_axis_m,
        "values": voltage_map.values,
        "metadata": np.array([str(voltage_map.metadata)], dtype=object),
    }
    if isinstance(voltage_map, AdaptiveVoltageMap):
        if voltage_map.sample_xy_m is not None:
            payload["sample_xy_m"] = voltage_map.sample_xy_m
        if voltage_map.sample_values is not None:
            payload["sample_values"] = voltage_map.sample_values
    np.savez_compressed(output_npz, **payload)
    return str(output_npz)
