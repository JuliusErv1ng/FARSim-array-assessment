from __future__ import annotations

import argparse
from pathlib import Path

from .io import save_results_npz, save_rfchain_long_csv
from .synthesizer import WaveformSynthesizer, input_from_mapping


DEFAULT_METRICS = Path("/Volumes/T7/FARSim/waveform_width_analysis/outputs/pulse_width_metrics_raw.csv")
DEFAULT_OUTPUT = Path("/Volumes/T7/FARSim/farsim_waveform_py/outputs")


def _read_first_metric_row(path: Path) -> dict[str, object]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return row
    raise ValueError(f"No rows found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize a FARSim-compatible example waveform.")
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--station-row", type=int, default=0, help="Zero-based row index in metrics CSV.")
    parser.add_argument("--dt-ns", type=float, default=0.5)
    args = parser.parse_args()

    row = _read_row(args.metrics_csv, args.station_row)
    item = input_from_mapping(row)
    synthesizer = WaveformSynthesizer(t_start_ns=-100.0, t_stop_ns=250.0, dt_ns=args.dt_ns)
    result = synthesizer.synthesize_one(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = save_results_npz(args.output_dir / "demo_waveform_station_trace.npz", [result])
    csv_path = save_rfchain_long_csv(args.output_dir / "demo_waveform_station_trace.csv", [result])
    fig_path = args.output_dir / "demo_waveform_station_trace.png"
    plot_path = _plot_result(result, fig_path)

    print(f"station_id: {result.station_id}")
    print(f"fwhm_ns: {result.fwhm_ns:.3f}")
    print(f"saved: {npz_path}")
    print(f"saved: {csv_path}")
    if plot_path is not None:
        print(f"saved: {plot_path}")
    else:
        print("plot skipped: matplotlib is not installed")


def _read_row(path: Path, row_index: int) -> dict[str, object]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index == row_index:
                return row
    raise ValueError(f"Row {row_index} not found in {path}")


def _plot_result(result, path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        return None

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(result.t_ns, result.trace_uv_m[:, 0], label="Ex", lw=1.6)
    ax.plot(result.t_ns, result.trace_uv_m[:, 1], label="Ey", lw=1.6)
    ax.plot(result.t_ns, result.trace_uv_m[:, 2], label="Ez", lw=1.6)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Electric field (uV/m)")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
