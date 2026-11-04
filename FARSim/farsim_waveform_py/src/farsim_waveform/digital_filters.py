from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal


FIR_FINAL_DIR = Path("/Volumes/T7/FARSim/FIR_final")


@dataclass(frozen=True, slots=True)
class ADCDigitalFilterConfig:
    """Post-ADC digital filters applied before the station trigger."""

    enabled: bool = True
    include_iir_notches: bool = True
    iir_notch_count: int = 1
    include_fir_lowpass: bool = True
    sample_rate_hz: float = 500.0e6
    fir_numtaps: int = 71
    fir_passband_hz: float = 115.0e6
    fir_stopband_hz: float = 120.0e6
    fir_passband_ripple_db: float = 1.0
    fir_stopband_attenuation_db: float = 40.0


DEFAULT_ADC_DIGITAL_FILTER = ADCDigitalFilterConfig()


def adc_filter_metadata(config: ADCDigitalFilterConfig | None = None) -> dict[str, object]:
    cfg = config or DEFAULT_ADC_DIGITAL_FILTER
    return {
        **asdict(cfg),
        "source": str(FIR_FINAL_DIR / "filter_parameters_genration.m"),
        "iir_notch_freq_mhz_used": [float(v / 1.0e6) for v in _notch_freq_hz()[: cfg.iir_notch_count]],
        "iir_notch_r_used": [float(v) for v in _notch_r()[: cfg.iir_notch_count]],
        "iir_notch_freq_mhz_available": [float(v / 1.0e6) for v in _notch_freq_hz()],
        "iir_notch_r_available": [float(v) for v in _notch_r()],
        "implementation_note": (
            "Only the post-ADC digital IIR notch stage and FIR low-pass stage are applied. "
            "The current trigger configuration uses the first IIR notch only; the remaining "
            "notch frequencies are kept in the parameter list but disabled by default. "
            "The analog AM/FEB S-parameter filters in FIR_final are not multiplied again, "
            "because they are already represented by the RFChain response."
        ),
    }


def apply_adc_iir_fir(
    t_ns: np.ndarray,
    voltage_uv: np.ndarray,
    config: ADCDigitalFilterConfig | None = None,
) -> np.ndarray:
    """Apply the post-ADC IIR notch bank and FIR low-pass filter.

    The input can be shaped as ``(n_samples,)``, ``(n_samples, n_channels)``,
    or ``(n_traces, n_samples, n_channels)``.  The output preserves the input
    shape.  Filtering is causal and returns the same number of samples.
    """

    cfg = config or DEFAULT_ADC_DIGITAL_FILTER
    voltage = np.asarray(voltage_uv, dtype=float)
    if not cfg.enabled:
        return voltage.copy()

    arr, restore = _to_trace_array(voltage)
    if arr.shape[1] < 4:
        return voltage.copy()
    fs_hz = _sample_rate_from_time(t_ns)

    out = arr.copy()
    if cfg.include_iir_notches:
        out = _apply_iir_notches(out, fs_hz, cfg.iir_notch_count)
    if cfg.include_fir_lowpass:
        taps = _fir_taps(
            fs_hz,
            int(cfg.fir_numtaps),
            float(cfg.fir_passband_hz),
            float(cfg.fir_stopband_hz),
            float(cfg.fir_passband_ripple_db),
            float(cfg.fir_stopband_attenuation_db),
        )
        out = signal.lfilter(taps, np.array([1.0]), out, axis=1)
    return restore(out)


def filter_noise_library(noise_library):
    """Return a filtered copy of a ``NoiseVoltageLibrary`` without importing it here."""

    vadc = apply_adc_iir_fir(noise_library.t_ns, noise_library.vadc_uv)
    metadata = dict(noise_library.metadata)
    metadata["adc_digital_filter"] = adc_filter_metadata()
    return type(noise_library)(
        t_ns=np.asarray(noise_library.t_ns, dtype=float).copy(),
        vadc_uv=vadc,
        metadata=metadata,
    )


def _to_trace_array(voltage: np.ndarray):
    if voltage.ndim == 1:
        return voltage.reshape(1, -1, 1), lambda arr: arr[0, :, 0]
    if voltage.ndim == 2:
        return voltage.reshape(1, voltage.shape[0], voltage.shape[1]), lambda arr: arr[0]
    if voltage.ndim == 3:
        return voltage, lambda arr: arr
    raise ValueError("voltage_uv must have 1, 2, or 3 dimensions.")


def _sample_rate_from_time(t_ns: np.ndarray) -> float:
    time = np.asarray(t_ns, dtype=float).reshape(-1)
    if time.size < 2:
        return float(DEFAULT_ADC_DIGITAL_FILTER.sample_rate_hz)
    dt_ns = float(np.median(np.diff(time)))
    if dt_ns <= 0.0:
        raise ValueError("t_ns must be strictly increasing.")
    return 1.0e9 / dt_ns


def _apply_iir_notches(arr: np.ndarray, fs_hz: float, notch_count: int) -> np.ndarray:
    sections, gain = _iir_notch_sections(float(fs_hz), int(notch_count))
    out = arr
    for b, a in sections:
        out = signal.lfilter(b, a, out, axis=1)
    return out / gain


@lru_cache(maxsize=16)
def _iir_notch_sections(fs_hz: float, notch_count: int) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], float]:
    sections: list[tuple[np.ndarray, np.ndarray]] = []
    notch_count = max(0, min(int(notch_count), len(_notch_freq_hz())))
    for freq_hz, radius in zip(_notch_freq_hz()[:notch_count], _notch_r()[:notch_count], strict=True):
        wn = 2.0 * np.pi * float(freq_hz) / float(fs_hz)
        r = float(radius)
        h13 = np.array([1.0, -2.0 * np.cos(wn), 1.0], dtype=float)
        h23 = np.array([1.0, 2.0 * r * np.cos(wn), r * r], dtype=float)
        h3u = np.array(
            [0.0, 0.0, 1.0, 0.0, 2.0 * r * r * np.cos(2.0 * wn), 0.0, r**4],
            dtype=float,
        )
        h3d = np.array(
            [1.0, 0.0, 0.0, 0.0, -2.0 * r**4 * np.cos(4.0 * wn), 0.0, 0.0, 0.0, r**8],
            dtype=float,
        )
        b = np.convolve(np.convolve(h13, h23), h3u)
        a = h3d
        sections.append((b, a))

    freqs = np.linspace(0.0, 0.5 * float(fs_hz), 8192)
    response = np.ones(freqs.shape, dtype=complex)
    for b, a in sections:
        _w, h = signal.freqz(b, a, worN=freqs, fs=float(fs_hz))
        response *= h
    gain = max(float(np.nanmax(np.abs(response))), 1.0e-12)
    return tuple(sections), gain


@lru_cache(maxsize=16)
def _fir_taps(
    fs_hz: float,
    numtaps: int,
    fpass_hz: float,
    fstop_hz: float,
    passband_ripple_db: float,
    stopband_attenuation_db: float,
) -> np.ndarray:
    numtaps = int(numtaps)
    if numtaps % 2 == 0:
        numtaps += 1
    pass_ripple = max(10.0 ** (float(passband_ripple_db) / 20.0) - 1.0, 1.0e-6)
    stop_ripple = max(10.0 ** (-float(stopband_attenuation_db) / 20.0), 1.0e-8)
    weights = [1.0 / pass_ripple, 1.0 / stop_ripple]
    try:
        return signal.remez(
            numtaps,
            [0.0, float(fpass_hz), float(fstop_hz), 0.5 * float(fs_hz)],
            [1.0, 0.0],
            weight=weights,
            fs=float(fs_hz),
        )
    except Exception:
        return signal.firwin(numtaps, float(fpass_hz), fs=float(fs_hz), window="hamming")


def _notch_freq_hz() -> np.ndarray:
    return np.array([39.0, 119.4, 132.0, 137.8, 121.5, 134.2, 124.7, 119.2], dtype=float) * 1.0e6


def _notch_r() -> np.ndarray:
    return np.array([0.9, 0.98, 0.95, 0.98, 0.96, 0.96, 0.96, 0.98], dtype=float)
