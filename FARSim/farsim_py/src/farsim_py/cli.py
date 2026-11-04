from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig, RuntimeConfig, TriggerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FARSim Python trigger-rate workflow.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root folder that contains the MATLAB data products.")
    parser.add_argument("--layout", type=Path, required=True, help="Position file for the array layout.")
    parser.add_argument("--threshold", type=float, default=109.0, help="Trigger threshold in uV/m.")
    parser.add_argument("--num-antennas", type=int, default=5, help="Minimum triggered antennas.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for final outputs.")
    parser.add_argument("--num-core-samples", type=int, default=5000, help="Number of Monte Carlo core positions.")
    parser.add_argument("--phi-step", type=int, default=10, help="Azimuth step used during efficiency evaluation.")
    parser.add_argument("--energy-step", type=float, default=0.1, help="Energy step in log10(E/eV).")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from .pipeline import run_pipeline

    config = PipelineConfig(
        data_root=args.data_root,
        layout_path=args.layout,
        trigger=TriggerConfig(
            threshold_uv_m=args.threshold,
            num_antennas=args.num_antennas,
        ),
        runtime=RuntimeConfig(
            output_dir=args.output_dir,
            num_core_samples=args.num_core_samples,
        ),
    )
    config.grid.phi_step = args.phi_step
    config.grid.energy_step = args.energy_step

    result = run_pipeline(config, seed=args.seed)
    print(result["output_path"])


if __name__ == "__main__":
    main()
