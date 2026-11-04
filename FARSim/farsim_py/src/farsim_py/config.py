from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TriggerConfig:
    threshold_uv_m: float = 109.0
    num_antennas: int = 5


@dataclass(slots=True)
class GridConfig:
    theta_start: int = 45
    theta_stop: int = 89
    phi_start: int = 0
    phi_stop: int = 359
    phi_step: int = 10
    energy_start: float = 16.0
    energy_stop: float = 18.0
    energy_step: float = 0.1


@dataclass(slots=True)
class RuntimeConfig:
    num_core_samples: int = 5000
    trigger_radius_m: float = 750.0
    use_polygon_radius: bool = False
    timing_jitter_ns: float = 5.0
    day_seconds: int = 24 * 3600
    output_dir: Path = Path("outputs")


@dataclass(slots=True)
class PipelineConfig:
    data_root: Path
    layout_path: Path
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    position_efield_cache: Path | None = None
    xmax2energy_path: Path | None = None
    flux_path: Path | None = None

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.layout_path = Path(self.layout_path)
        if self.position_efield_cache is None:
            self.position_efield_cache = self.data_root / "position&efield_raw_unfiltered.mat"
        if self.xmax2energy_path is None:
            self.xmax2energy_path = self.data_root / "xmax2energy.mat"
        if self.flux_path is None:
            self.flux_path = self.data_root / "Flux.xlsx"

    @property
    def layout_stem(self) -> str:
        return self.layout_path.stem
