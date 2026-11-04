from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from farsim_py.config import GridConfig, PipelineConfig, RuntimeConfig, TriggerConfig
from farsim_py.fast_event_rate import FastRateConfig, run_fast_event_rate
from farsim_py.footprint import FieldFootprint, compute_field_footprint
from farsim_py.io_utils import read_layout_positions
from farsim_py.pipeline import run_pipeline

from .aperture_modes import (
    AdaptiveVoltageMap,
    ApertureModeResult,
    NoiseVoltageLibrary,
    VoltageMap,
    build_adc_noise_library,
    compute_adaptive_voltage_peak_map,
    field_threshold_aperture,
    voltage_l1_aperture,
    voltage_threshold_aperture,
)
from .predictor import FARSimWaveformPredictor, StationPrediction
from .rfchain_adapter import RFChainLengthNAdapter, RFChainVoltageResult
from .voltage_trigger import AERAVoltageTriggerConfig


FARSIM_ROOT = Path(os.environ.get("FARSIM_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_DATA_ROOT = Path(os.environ.get("FARSIM_DATA_ROOT", FARSIM_ROOT / "data" / "StshpOutbox_XDS"))
DEFAULT_LAYOUT_ROOT = DEFAULT_DATA_ROOT / "layout" / "gp80"
DEFAULT_OUTPUT_ROOT = FARSIM_ROOT / "farsim_waveform_py" / "outputs" / "unified_api"
DEFAULT_NATIVE_RFCHAIN_FORWARD = Path(
    os.environ.get(
        "FARSIM_NATIVE_RFCHAIN_FORWARD",
        FARSIM_ROOT / "native_rfchain" / "batch_forward_rfchain_save_mat.py",
    )
)
DEFAULT_NATIVE_RFCHAIN_PARAMS = Path(
    os.environ.get(
        "FARSIM_NATIVE_RFCHAIN_PARAMS",
        FARSIM_ROOT / "native_rfchain" / "params",
    )
)


LAYOUT_FILES = {
    "infill": DEFAULT_LAYOUT_ROOT / "infill.txt",
    "coarse": DEFAULT_LAYOUT_ROOT / "coarse.txt",
    "elliptic1": DEFAULT_LAYOUT_ROOT / "elliptic1.txt",
    "elliptic2": DEFAULT_LAYOUT_ROOT / "elliptic2.txt",
    "elliptic3": DEFAULT_LAYOUT_ROOT / "elliptic3.txt",
    "hybrid": DEFAULT_LAYOUT_ROOT / "gp80_hybrid_position_meters.txt",
}


@dataclass(slots=True)
class PrimaryParticle:
    energy_eev: float
    theta_deg: float
    phi_deg: float
    particle: str = "Iron"


@dataclass(slots=True)
class ArrayTriggerConfig:
    required_antennas: int = 5
    field_threshold_uv_m: float = 75.0
    voltage_threshold_uv: float = 5000.0
    num_core_samples: int = 5000
    seed: int = 12345


@dataclass(slots=True)
class FARSimAPIConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    waveform_template: str = "ring_blend_kernel_trace"
    t_axis_ns: np.ndarray = field(default_factory=lambda: (np.arange(1999, dtype=float) - 999.0) * 0.5)
    rfchain_backend: str = "native"
    rfchain_module: Path = DEFAULT_NATIVE_RFCHAIN_FORWARD
    rfchain_params_dir: Path = DEFAULT_NATIVE_RFCHAIN_PARAMS
    noise_seed: int = 20260520
    noise_library_size: int = 16


@dataclass(slots=True)
class ApertureComparison:
    field_threshold: ApertureModeResult | None = None
    voltage_threshold: ApertureModeResult | None = None
    voltage_l1: ApertureModeResult | None = None
    voltage_map: VoltageMap | None = None
    l1_probability_map: VoltageMap | None = None

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for result in (self.field_threshold, self.voltage_threshold, self.voltage_l1):
            if result is None:
                continue
            row = asdict(result)
            row["effective_area_km2"] = row["effective_area_m2"] / 1.0e6
            rows.append(row)
        return rows


class FARSimAPI:
    """Unified FARSim API for field, waveform, RF-chain, and aperture studies."""

    def __init__(self, config: FARSimAPIConfig | None = None) -> None:
        self.config = config or FARSimAPIConfig()
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        self._predictor: FARSimWaveformPredictor | None = None
        self._rfchain_signal: RFChainLengthNAdapter | None = None
        self._rfchain_noise: RFChainLengthNAdapter | None = None
        self._adc_noise_library: NoiseVoltageLibrary | None = None

    def layout_path(self, layout: str | Path) -> Path:
        if isinstance(layout, Path):
            return layout
        if layout in LAYOUT_FILES:
            return LAYOUT_FILES[layout]
        candidate = Path(layout)
        if candidate.exists():
            return candidate
        raise ValueError(f"Unknown layout {layout!r}. Known layouts: {', '.join(LAYOUT_FILES)}")

    def electric_field_footprint(self, primary: PrimaryParticle) -> FieldFootprint:
        return compute_field_footprint(
            energy_eev=primary.energy_eev,
            theta_deg=primary.theta_deg,
            phi_deg=primary.phi_deg,
            data_root=self.config.data_root,
        )

    def waveform_at(
        self,
        primary: PrimaryParticle,
        *,
        x_m: float,
        y_m: float,
        z_m: float = 0.0,
        station_id: str | int | None = None,
        t_peak_ns: float = 0.0,
    ) -> StationPrediction:
        return self._predictor_cached().predict_one(
            energy_eev=primary.energy_eev,
            theta_deg=primary.theta_deg,
            phi_deg=primary.phi_deg,
            particle=primary.particle,
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            station_id=station_id,
            t_peak_ns=t_peak_ns,
        )

    def voltage_at(
        self,
        primary: PrimaryParticle,
        *,
        x_m: float,
        y_m: float,
        z_m: float = 0.0,
        station_id: str | int | None = None,
        noise: bool = False,
    ) -> RFChainVoltageResult:
        prediction = self.waveform_at(primary, x_m=x_m, y_m=y_m, z_m=z_m, station_id=station_id)
        rfchain = self._rfchain_noise_cached() if noise else self._rfchain_signal_cached()
        return rfchain.forward_trace(
            t_ns=prediction.t_ns,
            electric_trace_uv_m=prediction.trace_uv_m,
            theta_deg=primary.theta_deg,
            phi_deg=primary.phi_deg,
        )

    def field_threshold_event_rate(
        self,
        *,
        layout: str | Path = "elliptic3",
        threshold_uv_m: float = 75.0,
        required_antennas: int = 5,
        num_core_samples: int = 5000,
        trigger_library: str | Path | None = None,
        output_dir: str | Path | None = None,
        seed: int = 12345,
    ) -> dict[str, object]:
        resolved_output_dir = Path(output_dir) if output_dir is not None else self.config.output_root / "field_event_rate"
        if trigger_library is None:
            pipeline_config = PipelineConfig(
                data_root=self.config.data_root,
                layout_path=self.layout_path(layout),
                trigger=TriggerConfig(threshold_uv_m=float(threshold_uv_m), num_antennas=int(required_antennas)),
                grid=GridConfig(phi_step=10),
                runtime=RuntimeConfig(
                    num_core_samples=int(num_core_samples),
                    trigger_radius_m=750.0,
                    use_polygon_radius=True,
                    timing_jitter_ns=5.0,
                    output_dir=resolved_output_dir,
                ),
                position_efield_cache=self.config.data_root / "position&efield.mat",
            )
            result = run_pipeline(pipeline_config, seed=int(seed))
            summary = {
                "layout": self.layout_path(layout).stem,
                "threshold_uv_m": float(threshold_uv_m),
                "required_antennas": int(required_antennas),
                "num_core_samples": int(num_core_samples),
                "total_daily_rate": float(np.nansum(result["trg_rate"])),
                "mean_efficiency": float(np.mean(result["yita"])),
                "mean_fired_antennas": float(np.mean(result["fire_num"])),
                "seed": int(seed),
                "trigger_library": None,
                "position_efield_cache": str(pipeline_config.position_efield_cache),
                "calculation": "current_position_efield_contours",
            }
            return {"mat_path": str(result["output_path"]), "summary": summary, **result}

        config = FastRateConfig(
            data_root=self.config.data_root,
            layout=self.layout_path(layout),
            trigger_library=Path(trigger_library),
            threshold_uv_m=float(threshold_uv_m),
            required_antennas=int(required_antennas),
            num_core_samples=int(num_core_samples),
            seed=int(seed),
            output_dir=resolved_output_dir,
        )
        return run_fast_event_rate(config)

    def aperture_comparison(
        self,
        primary: PrimaryParticle,
        *,
        layout: str | Path = "elliptic3",
        trigger: ArrayTriggerConfig | None = None,
        modes: tuple[str, ...] = ("field", "voltage_threshold", "l1"),
        voltage_grid_size: int = 11,
        adaptive_voltage: bool = True,
        adaptive_refine_factor: int = 4,
        adaptive_band_fraction: float = 0.35,
        adaptive_max_refine_points: int = 800,
        l1_grid_size: int = 7,
        adaptive_l1: bool = True,
        adaptive_l1_refine_factor: int = 4,
        adaptive_l1_boundary_level: float = 0.5,
        adaptive_l1_boundary_band: float = 0.0,
        adaptive_l1_max_refine_points: int = 800,
        grid_extent_m: float = 3000.0,
        l1_config: AERAVoltageTriggerConfig | None = None,
        noise_trials: int = 1,
        noise_library_size: int | None = None,
    ) -> ApertureComparison:
        trigger = trigger or ArrayTriggerConfig()
        layout_path = self.layout_path(layout)
        out_dir = self.config.output_root / "aperture_modes" / _primary_label(primary) / layout_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        comparison = ApertureComparison()
        if "field" in modes:
            comparison.field_threshold, _footprint, _polygon = field_threshold_aperture(
                layout_path=layout_path,
                energy_eev=primary.energy_eev,
                theta_deg=primary.theta_deg,
                phi_deg=primary.phi_deg,
                threshold_uv_m=trigger.field_threshold_uv_m,
                required_antennas=trigger.required_antennas,
                num_core_samples=trigger.num_core_samples,
                seed=trigger.seed,
                data_root=self.config.data_root,
            )
        if "voltage_threshold" in modes:
            comparison.voltage_threshold, comparison.voltage_map, _polygon = voltage_threshold_aperture(
                layout_path=layout_path,
                energy_eev=primary.energy_eev,
                theta_deg=primary.theta_deg,
                phi_deg=primary.phi_deg,
                threshold_uv=trigger.voltage_threshold_uv,
                required_antennas=trigger.required_antennas,
                num_core_samples=trigger.num_core_samples,
                grid_size=voltage_grid_size,
                grid_extent_m=grid_extent_m,
                adaptive=adaptive_voltage,
                adaptive_refine_factor=adaptive_refine_factor,
                adaptive_band_fraction=adaptive_band_fraction,
                adaptive_max_refine_points=adaptive_max_refine_points,
                seed=trigger.seed,
                predictor=self._predictor_cached(),
                rfchain=self._rfchain_signal_cached(),
                output_npz=out_dir / "voltage_threshold_map.npz",
            )
        if "l1" in modes:
            comparison.voltage_l1, comparison.l1_probability_map = voltage_l1_aperture(
                layout_path=layout_path,
                energy_eev=primary.energy_eev,
                theta_deg=primary.theta_deg,
                phi_deg=primary.phi_deg,
                trigger_config=l1_config or default_l1_config(),
                required_antennas=trigger.required_antennas,
                num_core_samples=trigger.num_core_samples,
                noise_trials=noise_trials,
                grid_size=l1_grid_size,
                grid_extent_m=grid_extent_m,
                adaptive=adaptive_l1,
                adaptive_refine_factor=adaptive_l1_refine_factor,
                adaptive_boundary_level=adaptive_l1_boundary_level,
                adaptive_boundary_band=adaptive_l1_boundary_band,
                adaptive_max_refine_points=adaptive_l1_max_refine_points,
                seed=trigger.seed,
                predictor=self._predictor_cached(),
                signal_rfchain=self._rfchain_signal_cached(),
                noise_rfchain=self._rfchain_noise_cached(),
                noise_library=self._adc_noise_library_cached(noise_library_size),
                noise_library_size=self.config.noise_library_size if noise_library_size is None else int(noise_library_size),
                noise_seed=self.config.noise_seed,
                output_npz=out_dir / "l1_probability_map.npz",
            )
        return comparison

    def acceleration_notes(self) -> list[str]:
        return [
            "Field-threshold aperture now defaults to current position&efield contours; a precomputed contour library is only used when trigger_library is passed explicitly.",
            "Voltage-threshold aperture should use RF-chain linearity: compute a reference Vpeak map once, then scale signal with energy instead of rerunning RFChain for every energy bin.",
            "Voltage-threshold contours should be adaptive: scan a coarse grid first, then refine only cells that bracket or approach the voltage threshold.",
            "L1 probability maps should be adaptive too: scan a coarse station-probability grid first, then refine only cells crossing the P_L1 boundary.",
            "RFChain now defaults to the native FFT-bin forward chain; signal-only Vpeak maps still use RFChainLengthNAdapter.forward_traces_peak for a shared FARSim-facing API.",
            "L1 studies should split signal and noise: precompute signal spectra/voltage once per direction and station grid, then add reusable noise-only voltage traces for threshold scans.",
            "For full event-rate scans, parallelize independent theta/phi bins and write intermediate maps to disk so interrupted runs can resume.",
        ]

    def _predictor_cached(self) -> FARSimWaveformPredictor:
        if self._predictor is None:
            self._predictor = FARSimWaveformPredictor(
                data_root=self.config.data_root,
                template=self.config.waveform_template,
                t_axis_ns=self.config.t_axis_ns,
            )
        return self._predictor

    def _rfchain_signal_cached(self) -> RFChainLengthNAdapter:
        if self._rfchain_signal is None:
            self._rfchain_signal = RFChainLengthNAdapter(
                module_path=self.config.rfchain_module,
                backend=self.config.rfchain_backend,
                params_dir=self.config.rfchain_params_dir,
                noise_galactic=False,
                noise_inner=False,
            )
        return self._rfchain_signal

    def _rfchain_noise_cached(self) -> RFChainLengthNAdapter:
        if self._rfchain_noise is None:
            self._rfchain_noise = RFChainLengthNAdapter(
                module_path=self.config.rfchain_module,
                backend=self.config.rfchain_backend,
                params_dir=self.config.rfchain_params_dir,
                noise_galactic=True,
                noise_inner=True,
                seed=self.config.noise_seed,
            )
        return self._rfchain_noise

    def _adc_noise_library_cached(self, size: int | None = None) -> NoiseVoltageLibrary:
        requested_size = self.config.noise_library_size if size is None else int(size)
        if requested_size <= 0:
            return build_adc_noise_library(t_ns=self.config.t_axis_ns, rfchain=self._rfchain_noise_cached(), size=0)
        if self._adc_noise_library is None or int(self._adc_noise_library.metadata.get("size", 0)) != requested_size:
            self._adc_noise_library = build_adc_noise_library(
                t_ns=self.config.t_axis_ns,
                rfchain=self._rfchain_noise_cached(),
                size=requested_size,
            )
        return self._adc_noise_library


def default_l1_config() -> AERAVoltageTriggerConfig:
    return AERAVoltageTriggerConfig(
        threshold_t1_uv=6100.0,
        threshold_t2_uv=5500.0,
        tprev_ns=100.0,
        tper_ns=100.0,
        tc_max_ns=25.0,
        nc_min=1,
        nc_max=10,
        charge_min=None,
        charge_max=None,
        charge_mode="count",
        baseline="median",
        polarity="absolute",
        channel_logic="or",
        trigger_channel=1,
        adapt_windows_to_trace=True,
        signal_window_ns=(-100.0, 180.0),
    )


def _primary_label(primary: PrimaryParticle) -> str:
    return f"{primary.particle}_E{primary.energy_eev:g}_theta{primary.theta_deg:g}_phi{primary.phi_deg:g}".replace(".", "p")
