from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AERAVoltageTriggerConfig:
    """AERA-like station trigger applied to RF-chain voltage traces.

    Thresholds are expressed in microvolts because the RF-chain code returns
    ``Vadc(t)`` in microvolts.  The default values are deliberately conservative
    placeholders; tune them from noise-only traces before production scans.
    """

    threshold_t1_uv: float = 200.0
    threshold_t2_uv: float | None = None
    tprev_ns: float = 1250.0
    tper_ns: float = 6250.0
    tc_max_ns: float = 130.0
    nc_min: int = 1
    nc_max: int = 8
    q_min_uv: float | None = None
    q_max_uv: float | None = None
    charge_min: float | None = None
    charge_max: float | None = None
    charge_mode: str = "sum_abs"
    baseline: str = "median"
    polarity: str = "absolute"
    channel_logic: str = "or"
    trigger_channel: int | Sequence[int] | None = None
    adapt_windows_to_trace: bool = True
    signal_window_ns: tuple[float, float] | None = None

    @classmethod
    def grand_reference(cls, *, signal_threshold_adc: float = 66.0) -> "AERAVoltageTriggerConfig":
        """Reference parameter set from the current GRAND station trigger.

        The RF-chain code currently returns microvolts rather than ADC counts.
        ``signal_threshold_adc`` is therefore used as a voltage-like threshold
        until the ADC-count calibration is wired in explicitly.
        """

        return cls(
            threshold_t1_uv=float(signal_threshold_adc),
            threshold_t2_uv=float(signal_threshold_adc),
            tprev_ns=500.0,
            tper_ns=500.0,
            tc_max_ns=25.0,
            nc_min=1,
            nc_max=10,
            charge_min=0.0,
            charge_max=255.0,
            charge_mode="count",
            polarity="absolute",
            channel_logic="or",
            trigger_channel=1,
            adapt_windows_to_trace=True,
            signal_window_ns=(-150.0, 250.0),
        )


@dataclass(slots=True)
class VoltageTriggerDecision:
    triggered: bool
    triggered_channels: np.ndarray
    first_trigger_time_ns: float | None
    peak_voltage_uv: float
    crossing_counts: np.ndarray


def aera_voltage_trigger(
    t_ns: np.ndarray,
    voltage_uv: np.ndarray,
    config: AERAVoltageTriggerConfig | None = None,
) -> VoltageTriggerDecision:
    """Apply an AERA-inspired L1 trigger to a station voltage trace."""

    cfg = config or AERAVoltageTriggerConfig()
    time = np.asarray(t_ns, dtype=float).reshape(-1)
    voltage = np.asarray(voltage_uv, dtype=float)
    if voltage.ndim == 1:
        voltage = voltage[:, None]
    if voltage.ndim != 2:
        raise ValueError("voltage_uv must be one- or two-dimensional.")
    if voltage.shape[0] != time.size:
        raise ValueError("t_ns and voltage_uv must have the same length.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("t_ns must be strictly increasing.")
    channel_indices = _trigger_channel_indices(cfg.trigger_channel, voltage.shape[1])
    if channel_indices is not None:
        voltage = voltage[:, channel_indices]
    if cfg.channel_logic not in {"or", "and"}:
        raise ValueError("channel_logic must be 'or' or 'and'.")

    triggered_channels = np.zeros(voltage.shape[1], dtype=bool)
    crossing_counts = np.zeros(voltage.shape[1], dtype=int)
    trigger_times: list[float] = []
    for channel in range(voltage.shape[1]):
        channel_triggered, first_time, nc = _channel_trigger(time, voltage[:, channel], cfg)
        triggered_channels[channel] = channel_triggered
        crossing_counts[channel] = nc
        if first_time is not None:
            trigger_times.append(first_time)

    station_triggered = bool(np.any(triggered_channels)) if cfg.channel_logic == "or" else bool(np.all(triggered_channels))
    return VoltageTriggerDecision(
        triggered=station_triggered,
        triggered_channels=triggered_channels,
        first_trigger_time_ns=float(min(trigger_times)) if trigger_times else None,
        peak_voltage_uv=float(np.max(np.abs(voltage))),
        crossing_counts=crossing_counts,
    )


def _trigger_channel_indices(trigger_channel: int | Sequence[int] | None, n_channels: int) -> list[int] | None:
    if trigger_channel is None:
        return None
    if isinstance(trigger_channel, Sequence) and not isinstance(trigger_channel, (str, bytes)):
        channels = [int(channel) for channel in trigger_channel]
    else:
        channels = [int(trigger_channel)]
    if not channels:
        raise ValueError("trigger_channel must contain at least one channel index.")
    unique_channels = list(dict.fromkeys(channels))
    bad = [channel for channel in unique_channels if channel < 0 or channel >= n_channels]
    if bad:
        raise ValueError(f"trigger_channel={bad} is outside voltage channel count {n_channels}.")
    return unique_channels


def _channel_trigger(
    t_ns: np.ndarray,
    voltage_uv: np.ndarray,
    cfg: AERAVoltageTriggerConfig,
) -> tuple[bool, float | None, int]:
    signal = _baseline_subtracted_signal(voltage_uv, cfg)
    t1 = float(cfg.threshold_t1_uv)
    t2 = float(cfg.threshold_t2_uv if cfg.threshold_t2_uv is not None else 0.75 * cfg.threshold_t1_uv)
    t1_crossings = _rising_crossing_indices(signal, t1)
    if cfg.signal_window_ns is not None:
        lo_sig, hi_sig = cfg.signal_window_ns
        t1_crossings = t1_crossings[(t_ns[t1_crossings] >= lo_sig) & (t_ns[t1_crossings] <= hi_sig)]
    if t1_crossings.size == 0:
        return False, None, 0

    for idx in t1_crossings:
        t_cross = float(t_ns[idx])
        tprev_ns = float(cfg.tprev_ns)
        tper_ns = float(cfg.tper_ns)
        if cfg.adapt_windows_to_trace:
            tprev_ns = min(tprev_ns, max(0.0, t_cross - float(t_ns[0])))
            tper_ns = min(tper_ns, max(0.0, float(t_ns[-1]) - t_cross))

        previous = t1_crossings[(t1_crossings < idx) & (t_ns[t1_crossings] >= t_cross - tprev_ns)]
        if previous.size:
            continue

        t2_crossings = _rising_crossing_indices(signal, t2)
        # Include the T2 crossing that belongs to the same pulse, which can
        # happen shortly before T1 when T2 < T1.
        lo = t_cross - cfg.tc_max_ns
        hi = t_cross + tper_ns
        local = t2_crossings[(t_ns[t2_crossings] >= lo) & (t_ns[t2_crossings] <= hi)]
        nc = int(local.size)
        if nc < cfg.nc_min or nc > cfg.nc_max:
            continue
        if local.size > 1:
            dt = np.diff(t_ns[local])
            if np.any(dt > cfg.tc_max_ns):
                continue

        pmax = float(np.max(signal[(t_ns >= t_cross) & (t_ns <= hi)]))
        charge = _charge(signal[(t_ns >= t_cross) & (t_ns <= hi)], t2, cfg)
        q = pmax / max(nc, 1)
        if cfg.q_min_uv is not None and q <= cfg.q_min_uv:
            continue
        if cfg.q_max_uv is not None and q >= cfg.q_max_uv:
            continue
        if cfg.charge_min is not None and charge < cfg.charge_min:
            continue
        if cfg.charge_max is not None and charge > cfg.charge_max:
            continue
        return True, t_cross, nc
    return False, None, 0


def _baseline_subtracted_signal(voltage_uv: np.ndarray, cfg: AERAVoltageTriggerConfig) -> np.ndarray:
    if cfg.baseline == "median":
        centered = voltage_uv - float(np.median(voltage_uv))
    elif cfg.baseline == "mean":
        centered = voltage_uv - float(np.mean(voltage_uv))
    elif cfg.baseline == "none":
        centered = voltage_uv.copy()
    else:
        raise ValueError("baseline must be 'median', 'mean', or 'none'.")

    if cfg.polarity == "absolute":
        return np.abs(centered)
    if cfg.polarity == "positive":
        return centered
    if cfg.polarity == "negative":
        return -centered
    raise ValueError("polarity must be 'absolute', 'positive', or 'negative'.")


def _rising_crossing_indices(signal: np.ndarray, threshold: float) -> np.ndarray:
    below = signal[:-1] < threshold
    above = signal[1:] >= threshold
    return np.flatnonzero(below & above) + 1


def _charge(signal: np.ndarray, threshold: float, cfg: AERAVoltageTriggerConfig) -> float:
    if signal.size == 0:
        return 0.0
    above = signal[signal >= threshold]
    if cfg.charge_mode == "sum_abs":
        return float(np.sum(above))
    if cfg.charge_mode == "sum_excess":
        return float(np.sum(above - threshold))
    if cfg.charge_mode == "count":
        return float(above.size)
    raise ValueError("charge_mode must be 'sum_abs', 'sum_excess', or 'count'.")
