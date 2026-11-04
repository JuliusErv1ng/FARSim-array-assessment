from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .api import ArrayTriggerConfig, FARSimAPI, PrimaryParticle


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified FARSim API command line.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    field_rate = subparsers.add_parser("field-rate", help="Run field-threshold event-rate scan.")
    field_rate.add_argument("--layout", default="elliptic3")
    field_rate.add_argument("--threshold-uvm", type=float, default=75.0)
    field_rate.add_argument("--required-antennas", type=int, default=5)
    field_rate.add_argument("--core-samples", type=int, default=5000)
    field_rate.add_argument("--trigger-library", type=Path, default=None)
    field_rate.add_argument("--output-dir", type=Path, default=None)

    waveform = subparsers.add_parser("waveform", help="Generate one station electric-field waveform.")
    _add_primary_args(waveform)
    waveform.add_argument("--x-m", type=float, required=True)
    waveform.add_argument("--y-m", type=float, required=True)
    waveform.add_argument("--z-m", type=float, default=0.0)

    aperture = subparsers.add_parser("aperture", help="Run one-bin aperture comparison.")
    _add_primary_args(aperture)
    aperture.add_argument("--layout", default="elliptic3")
    aperture.add_argument("--mode", action="append", choices=["field", "voltage_threshold", "l1"], default=None)
    aperture.add_argument("--field-threshold-uvm", type=float, default=75.0)
    aperture.add_argument("--voltage-threshold-uv", type=float, default=5000.0)
    aperture.add_argument("--required-antennas", type=int, default=5)
    aperture.add_argument("--core-samples", type=int, default=1000)
    aperture.add_argument("--voltage-grid-size", type=int, default=7)
    aperture.add_argument("--no-adaptive-voltage", action="store_true")
    aperture.add_argument("--adaptive-refine-factor", type=int, default=4)
    aperture.add_argument("--adaptive-band-fraction", type=float, default=0.35)
    aperture.add_argument("--adaptive-max-refine-points", type=int, default=800)
    aperture.add_argument("--l1-grid-size", type=int, default=5)
    aperture.add_argument("--no-adaptive-l1", action="store_true")
    aperture.add_argument("--adaptive-l1-refine-factor", type=int, default=4)
    aperture.add_argument("--adaptive-l1-boundary-level", type=float, default=0.5)
    aperture.add_argument("--adaptive-l1-boundary-band", type=float, default=0.0)
    aperture.add_argument("--adaptive-l1-max-refine-points", type=int, default=800)
    aperture.add_argument("--grid-extent-m", type=float, default=3000.0)
    aperture.add_argument("--noise-trials", type=int, default=1)
    aperture.add_argument("--noise-library-size", type=int, default=None)

    subparsers.add_parser("notes", help="Print acceleration recommendations.")
    args = parser.parse_args()

    api = FARSimAPI()
    if args.command == "field-rate":
        result = api.field_threshold_event_rate(
            layout=args.layout,
            threshold_uv_m=args.threshold_uvm,
            required_antennas=args.required_antennas,
            num_core_samples=args.core_samples,
            trigger_library=args.trigger_library,
            output_dir=args.output_dir,
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(result["mat_path"])
        return

    if args.command == "waveform":
        primary = _primary_from_args(args)
        result = api.waveform_at(primary, x_m=args.x_m, y_m=args.y_m, z_m=args.z_m)
        peak = float(np.max(np.linalg.norm(result.trace_uv_m, axis=1)))
        print(json.dumps({"peak_uv_m": peak, "fwhm_ns": result.result.fwhm_ns, **result.geometry}, ensure_ascii=False, indent=2))
        return

    if args.command == "aperture":
        primary = _primary_from_args(args)
        trigger = ArrayTriggerConfig(
            required_antennas=args.required_antennas,
            field_threshold_uv_m=args.field_threshold_uvm,
            voltage_threshold_uv=args.voltage_threshold_uv,
            num_core_samples=args.core_samples,
        )
        comparison = api.aperture_comparison(
            primary,
            layout=args.layout,
            trigger=trigger,
            modes=tuple(args.mode or ["field", "voltage_threshold", "l1"]),
            voltage_grid_size=args.voltage_grid_size,
            adaptive_voltage=not args.no_adaptive_voltage,
            adaptive_refine_factor=args.adaptive_refine_factor,
            adaptive_band_fraction=args.adaptive_band_fraction,
            adaptive_max_refine_points=args.adaptive_max_refine_points,
            l1_grid_size=args.l1_grid_size,
            adaptive_l1=not args.no_adaptive_l1,
            adaptive_l1_refine_factor=args.adaptive_l1_refine_factor,
            adaptive_l1_boundary_level=args.adaptive_l1_boundary_level,
            adaptive_l1_boundary_band=args.adaptive_l1_boundary_band,
            adaptive_l1_max_refine_points=args.adaptive_l1_max_refine_points,
            grid_extent_m=args.grid_extent_m,
            noise_trials=args.noise_trials,
            noise_library_size=args.noise_library_size,
        )
        print(json.dumps(comparison.summary(), ensure_ascii=False, indent=2))
        return

    if args.command == "notes":
        for index, note in enumerate(api.acceleration_notes(), start=1):
            print(f"{index}. {note}")


def _add_primary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--energy-eev", type=float, default=0.316)
    parser.add_argument("--theta-deg", type=float, default=75.0)
    parser.add_argument("--phi-deg", type=float, default=45.0)
    parser.add_argument("--particle", default="Iron")


def _primary_from_args(args: argparse.Namespace) -> PrimaryParticle:
    return PrimaryParticle(
        energy_eev=args.energy_eev,
        theta_deg=args.theta_deg,
        phi_deg=args.phi_deg,
        particle=args.particle,
    )


if __name__ == "__main__":
    main()
