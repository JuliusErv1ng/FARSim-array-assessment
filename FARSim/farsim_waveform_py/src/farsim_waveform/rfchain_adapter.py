from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np


DEFAULT_RFCHAIN_LENGTHN = Path("/Volumes/T7/RFChain/RFchain_20250302_updated/rfchain_lengthN.py")
DEFAULT_BUNDLE_ROOT = Path(os.environ.get("FARSIM_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_NATIVE_RFCHAIN_FORWARD = Path(
    os.environ.get(
        "FARSIM_NATIVE_RFCHAIN_FORWARD",
        DEFAULT_BUNDLE_ROOT / "native_rfchain" / "batch_forward_rfchain_save_mat.py",
    )
)
DEFAULT_NATIVE_RFCHAIN_PARAMS = Path(
    os.environ.get(
        "FARSIM_NATIVE_RFCHAIN_PARAMS",
        DEFAULT_BUNDLE_ROOT / "native_rfchain" / "params",
    )
)
DEFAULT_RFCHAIN_BACKEND = "native"


@dataclass(slots=True)
class RFChainVoltageResult:
    """RF-chain output for one station trace.

    All voltages are in microvolts, following the original RF-chain scripts.
    """

    t_ns: np.ndarray
    e_shower_downsample_uv_m: np.ndarray
    voc_uv: np.ndarray
    vadc_uv: np.ndarray
    fs_hz: float
    fs_ds_hz: float
    freq_ghz: np.ndarray

    @property
    def peak_voltage_uv(self) -> float:
        return float(np.max(np.abs(self.vadc_uv)))

    @property
    def peak_voltage_vector_uv(self) -> np.ndarray:
        idx = np.argmax(np.max(np.abs(self.vadc_uv), axis=1))
        return self.vadc_uv[int(idx), :].copy()


class RFChainLengthNAdapter:
    """Small Python wrapper around the existing RF-chain projects.

    FARSim now defaults to the native FFT-bin forward chain used by the
    reconstruction MAT workflow.  The historical 221-point ``rfchain_lengthN``
    implementation is still available with ``backend="lengthn"`` for
    comparison.
    """

    def __init__(
        self,
        *,
        module_path: str | Path | None = None,
        backend: str = DEFAULT_RFCHAIN_BACKEND,
        params_dir: str | Path = DEFAULT_NATIVE_RFCHAIN_PARAMS,
        noise_galactic: bool = True,
        noise_inner: bool = True,
        lst_hour: int = 11,
        lst_minute: int = 0,
        auto_drop_nyquist: bool = True,
        seed: int | None = None,
    ) -> None:
        if backend not in {"native", "lengthn"}:
            raise ValueError("backend must be 'native' or 'lengthn'.")
        self.backend = str(backend)
        default_module = DEFAULT_NATIVE_RFCHAIN_FORWARD if self.backend == "native" else DEFAULT_RFCHAIN_LENGTHN
        self.module_path = Path(module_path) if module_path is not None else default_module
        self.params_dir = Path(params_dir)
        self.noise_galactic = bool(noise_galactic)
        self.noise_inner = bool(noise_inner)
        self.lst_hour = int(lst_hour)
        self.lst_minute = int(lst_minute)
        self.auto_drop_nyquist = bool(auto_drop_nyquist)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        if self.seed is not None:
            # The original RF-chain functions use np.random directly; seed once
            # per adapter so repeated noise trials are not identical.
            np.random.seed(int(self.seed))
        self._module: ModuleType | None = None
        self._cache: dict[tuple[float, int], dict[str, object]] = {}
        self._efl_cache: dict[tuple[float, int, float, float], np.ndarray] = {}
        self._native_cache: object | None = None
        self._native_config: object | None = None

    def forward_trace(
        self,
        *,
        t_ns: np.ndarray,
        electric_trace_uv_m: np.ndarray,
        theta_deg: float,
        phi_deg: float,
    ) -> RFChainVoltageResult:
        """Convert a three-component electric-field trace into ADC voltage."""

        if self.backend == "native":
            return self._forward_trace_native(
                t_ns=t_ns,
                electric_trace_uv_m=electric_trace_uv_m,
                theta_deg=theta_deg,
                phi_deg=phi_deg,
            )

        module = self._load_module()
        trace = _trace_matrix(t_ns, electric_trace_uv_m)
        dt_s = float((trace[1, 0] - trace[0, 0]) * 1e-9)
        if dt_s <= 0.0:
            raise ValueError("t_ns must be strictly increasing.")
        fs_hz = 1.0 / dt_s
        cache = self._prepare_cache(fs_hz)
        cfg = self._make_rf_config(module)

        e_ds, e_complex, fs_ds = module.fft_trace_to_grid(trace, fs_hz, cache["freq_all"])
        efl = self._pick_efl_cached(cache, float(theta_deg), float(phi_deg), fs_hz)
        voc = module.compute_voc(efl, e_complex)
        voc = module.inject_galactic_noise(voc, cfg, fs_ds, e_ds.shape[0], cache["freq_all"])
        v1, i1 = module.solve_antenna_terminal(voc, cache["zin"], cache["zin_z"], cache["params"]["Z_ant"])
        v2, _i2 = module.solve_chain_response(v1, i1, cache["chain"])
        vadc = module.inject_inner_noise(v2, self.noise_inner, fs_ds, e_ds.shape[0], cache["freq_all"])
        time_outputs = module.spectra_to_time_outputs(voc, vadc, fs_ds, e_ds.shape[0], cache["freq_all"])

        return RFChainVoltageResult(
            t_ns=np.asarray(e_ds[:, 0], dtype=float),
            e_shower_downsample_uv_m=np.asarray(e_ds[:, 1:4], dtype=float),
            voc_uv=np.asarray(time_outputs["voc_t"], dtype=float),
            vadc_uv=np.asarray(time_outputs["vadc_t"], dtype=float),
            fs_hz=fs_hz,
            fs_ds_hz=float(fs_ds),
            freq_ghz=np.asarray(cache["freq_all"], dtype=float),
        )

    def forward_traces_peak(
        self,
        *,
        t_ns: np.ndarray,
        electric_traces_uv_m: np.ndarray,
        theta_deg: float,
        phi_deg: float,
    ) -> np.ndarray:
        """Return peak ADC voltages for many traces with one cached RF-chain setup.

        This vectorized path is intended for deterministic voltage-threshold
        aperture scans.  Noise is deliberately not supported here because
        production L1 studies should treat signal and noise as separate random
        fields rather than reusing one deterministic peak map.
        """

        if self.noise_galactic or self.noise_inner:
            raise ValueError("forward_traces_peak requires noise_galactic=False and noise_inner=False.")

        if self.backend == "native":
            return np.array(
                [
                    self.forward_trace(
                        t_ns=t_ns,
                        electric_trace_uv_m=trace,
                        theta_deg=theta_deg,
                        phi_deg=phi_deg,
                    ).peak_voltage_uv
                    for trace in np.asarray(electric_traces_uv_m, dtype=float)
                ],
                dtype=float,
            )

        time = np.asarray(t_ns, dtype=float).reshape(-1)
        traces = np.asarray(electric_traces_uv_m, dtype=float)
        if traces.ndim != 3 or traces.shape[2] != 3:
            raise ValueError("electric_traces_uv_m must have shape (n_traces, n_samples, 3).")
        if traces.shape[1] != time.size:
            raise ValueError("t_ns and electric_traces_uv_m must have the same sample length.")
        if time.size < 4:
            raise ValueError("At least four time samples are required.")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("t_ns must be strictly increasing.")

        module = self._load_module()
        dt_s = float((time[1] - time[0]) * 1e-9)
        fs_hz = 1.0 / dt_s
        cache = self._prepare_cache(fs_hz)
        freq_all = np.asarray(cache["freq_all"], dtype=float)
        efl = self._pick_efl_cached(cache, float(theta_deg), float(phi_deg), fs_hz)

        downsampled_groups, fs_ds = _bandpass_downsample_groups(time, traces, freq_all)
        peaks = np.zeros(traces.shape[0], dtype=float)
        chain_cache = _prepare_chain_cache(cache)
        for trace_indices, _offset, downsampled in downsampled_groups:
            if len(trace_indices) == 0:
                continue
            e_complex = _fft_traces_to_grid(downsampled, fs_ds, freq_all)
            v_adc = _rfchain_spectrum_batch(efl, e_complex, chain_cache)
            voltage_t = _spectra_to_time_batch(module, v_adc, fs_ds, downsampled.shape[1], freq_all)
            peaks[trace_indices] = np.max(np.abs(voltage_t), axis=(1, 2))
        return peaks

    def forward_traces_vadc_groups(
        self,
        *,
        t_ns: np.ndarray,
        electric_traces_uv_m: np.ndarray,
        theta_deg: float,
        phi_deg: float,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return batched ADC voltage traces grouped by RF-chain downsampling offset.

        The existing RF-chain preprocessing can choose different downsampling
        offsets for different input traces.  Returning groups keeps the exact
        time axis for each subset while still sharing EFL, impedance, and chain
        caches across all traces in one shower direction.
        """

        if self.noise_galactic or self.noise_inner:
            raise ValueError("forward_traces_vadc_groups requires noise_galactic=False and noise_inner=False.")

        if self.backend == "native":
            return self._forward_traces_vadc_groups_native(
                t_ns=t_ns,
                electric_traces_uv_m=electric_traces_uv_m,
                theta_deg=theta_deg,
                phi_deg=phi_deg,
            )

        time = np.asarray(t_ns, dtype=float).reshape(-1)
        traces = np.asarray(electric_traces_uv_m, dtype=float)
        if traces.ndim != 3 or traces.shape[2] != 3:
            raise ValueError("electric_traces_uv_m must have shape (n_traces, n_samples, 3).")
        if traces.shape[1] != time.size:
            raise ValueError("t_ns and electric_traces_uv_m must have the same sample length.")
        if time.size < 4:
            raise ValueError("At least four time samples are required.")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("t_ns must be strictly increasing.")

        module = self._load_module()
        dt_s = float((time[1] - time[0]) * 1e-9)
        fs_hz = 1.0 / dt_s
        cache = self._prepare_cache(fs_hz)
        freq_all = np.asarray(cache["freq_all"], dtype=float)
        efl = self._pick_efl_cached(cache, float(theta_deg), float(phi_deg), fs_hz)

        downsampled_groups, fs_ds = _bandpass_downsample_groups(time, traces, freq_all)
        decimation = max(1, int(round(fs_hz / fs_ds)))
        out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        chain_cache = _prepare_chain_cache(cache)
        for trace_indices, offset, downsampled in downsampled_groups:
            if len(trace_indices) == 0:
                continue
            e_complex = _fft_traces_to_grid(downsampled, fs_ds, freq_all)
            v_adc = _rfchain_spectrum_batch(efl, e_complex, chain_cache)
            voltage_t = _spectra_to_time_batch(module, v_adc, fs_ds, downsampled.shape[1], freq_all)
            time_ds = time[offset::decimation][: voltage_t.shape[1]]
            out.append((trace_indices, time_ds, voltage_t))
        return out

    def _pick_efl_cached(self, cache: dict[str, object], theta_deg: float, phi_deg: float, fs_hz: float) -> np.ndarray:
        """Return EFL for one direction using cached regular-grid interpolation.

        The original RF-chain script calls ``scipy.interpolate.griddata`` for
        every frequency/port/polarization entry.  That is fine for one trace,
        but prohibitively slow when scanning all antennas in one shower.  The
        EFL tables are stored on a regular theta-phi grid, so a vectorized
        bilinear interpolation gives the same interpolation logic at a tiny
        fraction of the cost and can be reused for every station in the event.
        """

        freq_all = np.asarray(cache["freq_all"], dtype=float)
        key = (float(fs_hz), int(len(freq_all)), round(float(theta_deg), 6), round(float(phi_deg) % 360.0, 6))
        if key not in self._efl_cache:
            params = cache["params"]
            self._efl_cache[key] = _bilinear_efl(
                np.asarray(params["EFL_all"]),
                np.ravel(np.asarray(params["Theta"], dtype=float)),
                np.ravel(np.asarray(params["Phi"], dtype=float)),
                float(theta_deg),
                float(phi_deg),
            )
        return self._efl_cache[key]

    def _load_module(self) -> ModuleType:
        if self._module is not None:
            return self._module
        if not self.module_path.exists():
            raise FileNotFoundError(f"RF-chain module not found: {self.module_path}")
        spec = importlib.util.spec_from_file_location("rfchain_lengthN_external", self.module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import RF-chain module: {self.module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        original_sys_path = sys.path.copy()
        try:
            spec.loader.exec_module(module)
        finally:
            # The external RF-chain script prepends its project-local .pydeps
            # directory.  Keep that import detail local so spawned workers do
            # not inherit a conflicting NumPy/SciPy stack.
            sys.path[:] = original_sys_path
        self._module = module
        return module

    def _forward_trace_native(
        self,
        *,
        t_ns: np.ndarray,
        electric_trace_uv_m: np.ndarray,
        theta_deg: float,
        phi_deg: float,
    ) -> RFChainVoltageResult:
        module = self._load_module()
        trace = _trace_matrix(t_ns, electric_trace_uv_m)
        dt_s = float((trace[1, 0] - trace[0, 0]) * 1e-9)
        if dt_s <= 0.0:
            raise ValueError("t_ns must be strictly increasing.")
        fs_hz = 1.0 / dt_s
        cfg = self._native_config_cached(module)
        cache = self._native_cache_cached(module)
        result = module.forward_trace(
            trace,
            float(theta_deg),
            float(phi_deg),
            cache,
            cfg,
            self._rng,
        )
        return RFChainVoltageResult(
            t_ns=np.asarray(result.e_shower_downsample[:, 0], dtype=float),
            e_shower_downsample_uv_m=np.asarray(result.e_shower_downsample[:, 1:4], dtype=float),
            voc_uv=np.asarray(result.voc_t, dtype=float),
            vadc_uv=np.asarray(result.vadc_t, dtype=float),
            fs_hz=fs_hz,
            fs_ds_hz=float(result.fs_ds),
            freq_ghz=np.asarray(cfg.freq_all_ghz, dtype=float),
        )

    def _forward_traces_vadc_groups_native(
        self,
        *,
        t_ns: np.ndarray,
        electric_traces_uv_m: np.ndarray,
        theta_deg: float,
        phi_deg: float,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        time = np.asarray(t_ns, dtype=float).reshape(-1)
        traces = np.asarray(electric_traces_uv_m, dtype=float)
        if traces.ndim != 3 or traces.shape[2] != 3:
            raise ValueError("electric_traces_uv_m must have shape (n_traces, n_samples, 3).")
        if traces.shape[1] != time.size:
            raise ValueError("t_ns and electric_traces_uv_m must have the same sample length.")
        grouped: dict[tuple[float, ...], list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for trace_idx, trace in enumerate(traces):
            result = self.forward_trace(
                t_ns=time,
                electric_trace_uv_m=trace,
                theta_deg=theta_deg,
                phi_deg=phi_deg,
            )
            # Native forward can choose different decimation offsets.  Group by
            # output time axis so downstream trigger windows remain exact.
            key = tuple(np.round(np.asarray(result.t_ns, dtype=float), decimals=9).tolist())
            grouped.setdefault(key, []).append((trace_idx, np.asarray(result.t_ns, dtype=float), np.asarray(result.vadc_uv, dtype=float)))

        out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for rows in grouped.values():
            indices = np.array([row[0] for row in rows], dtype=int)
            t_group = rows[0][1]
            vadc = np.stack([row[2] for row in rows], axis=0)
            out.append((indices, t_group, vadc))
        return out

    def _native_config_cached(self, module: ModuleType):
        if self._native_config is None:
            self._native_config = module.Config(
                sim_dir=self.params_dir.parent.parent / "finish",
                params_dir=self.params_dir,
                freq_all_ghz=np.arange(0.03, 0.251, 0.001, dtype=np.float64),
                noiseflag=self.noise_galactic,
                innernoise_flag=self.noise_inner,
                lst_hour=self.lst_hour,
                lst_minute=self.lst_minute,
                overwrite=True,
                band_lo_hz=30.0e6,
                band_hi_hz=250.0e6,
                edge_taper_hz=10.0e6,
                visualize=False,
                max_plots=0,
                compare_clean=False,
                max_folders=None,
                max_traces=None,
                folder_contains=None,
                folders_file=None,
                adc_only=True,
                legacy_adc_names=False,
            )
        return self._native_config

    def _native_cache_cached(self, module: ModuleType):
        if self._native_cache is None:
            cfg = self._native_config_cached(module)
            if not Path(cfg.params_dir).is_dir():
                raise FileNotFoundError(f"Native RF-chain params dir not found: {cfg.params_dir}")
            self._native_cache = module.load_chain_cache(cfg)
        return self._native_cache

    def _make_rf_config(self, module: ModuleType):
        return module.RFConfig(
            noiseflag=self.noise_galactic,
            innernoise_flag=self.noise_inner,
            lst_hour=self.lst_hour,
            lst_minute=self.lst_minute,
            auto_drop_nyquist=self.auto_drop_nyquist,
        )

    def _prepare_cache(self, fs_hz: float) -> dict[str, object]:
        module = self._load_module()
        freq_all = module.make_working_frequency_grid(float(fs_hz), self.auto_drop_nyquist)
        key = (float(fs_hz), len(freq_all))
        if key not in self._cache:
            params = module.load_rfchain_parameters(freq_all)
            chain = module.build_abcd_chain(params["s"])
            zin, zin_z = module.solve_antenna_input_impedance(chain, params["s_load"])
            self._cache[key] = {
                "freq_all": freq_all,
                "params": params,
                "chain": chain,
                "zin": zin,
                "zin_z": zin_z,
            }
        return self._cache[key]


def _trace_matrix(t_ns: np.ndarray, electric_trace_uv_m: np.ndarray) -> np.ndarray:
    time = np.asarray(t_ns, dtype=float).reshape(-1)
    trace = np.asarray(electric_trace_uv_m, dtype=float)
    if trace.ndim != 2 or trace.shape[1] != 3:
        raise ValueError("electric_trace_uv_m must have shape (N, 3).")
    if trace.shape[0] != time.size:
        raise ValueError("t_ns and electric_trace_uv_m must have the same length.")
    if time.size < 4:
        raise ValueError("At least four time samples are required.")
    return np.column_stack((time, trace))


def _bandpass_downsample_groups(
    t_ns: np.ndarray,
    traces: np.ndarray,
    freq_all: np.ndarray,
) -> tuple[list[tuple[np.ndarray, int, np.ndarray]], float]:
    """Apply the RF-chain bandpass/downsampling step to a batch of traces."""

    dt_s = float((t_ns[1] - t_ns[0]) * 1e-9)
    fs = 1.0 / dt_s
    n_samples = traces.shape[1]
    band_start_hz = float(np.min(freq_all) * 1e9)
    band_stop_hz = float(np.max(freq_all) * 1e9)
    freq = np.arange(n_samples) * fs / n_samples
    mask = ((freq >= band_start_hz) & (freq <= band_stop_hz)) | (
        (freq >= fs - band_stop_hz) & (freq <= fs - band_start_hz)
    )

    trace_fft = np.fft.fft(traces, axis=1)
    trace_fft[:, ~mask, :] = 0.0
    bandpassed = np.fft.ifft(trace_fft, axis=1).real

    target_fs = 500e6
    nyquist_guard = 1.02
    max_decim_by_band = int(np.floor(fs / (2.0 * band_stop_hz * nyquist_guard)))
    max_decim_by_target = int(np.floor(fs / target_fs))
    decimation = max(1, min(max_decim_by_band, max_decim_by_target))
    fs_ds = fs / decimation

    amplitudes = []
    for offset in range(decimation):
        amplitudes.append(np.max(np.abs(bandpassed[:, offset::decimation, :]), axis=(1, 2)))
    best_offsets = np.argmax(np.stack(amplitudes, axis=1), axis=1)

    groups: list[tuple[np.ndarray, int, np.ndarray]] = []
    for offset in range(decimation):
        trace_indices = np.flatnonzero(best_offsets == offset)
        if trace_indices.size == 0:
            continue
        groups.append((trace_indices, offset, bandpassed[trace_indices, offset::decimation, :]))
    return groups, float(fs_ds)


def _fft_traces_to_grid(traces_t: np.ndarray, fs: float, freq_all: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of rfchain_lengthN.fft_n_to_grid for traces."""

    n_traces, n_samples, n_components = traces_t.shape
    trace_fft = np.fft.fft(traces_t, axis=1)
    npos = n_samples // 2 + 1 if n_samples % 2 == 0 else (n_samples + 1) // 2
    xpos = trace_fft[:, :npos, :] / n_samples * 2.0
    xpos[:, 0, :] /= 2.0
    if n_samples % 2 == 0:
        xpos[:, -1, :] /= 2.0
    f_pos = np.arange(npos, dtype=float) * fs / n_samples
    target = np.asarray(freq_all, dtype=float) * 1e9

    out = np.zeros((n_traces, len(target), n_components), dtype=complex)
    for trace_idx in range(n_traces):
        for comp_idx in range(n_components):
            real = np.interp(target, f_pos, np.real(xpos[trace_idx, :, comp_idx]), left=0.0, right=0.0)
            imag = np.interp(target, f_pos, np.imag(xpos[trace_idx, :, comp_idx]), left=0.0, right=0.0)
            out[trace_idx, :, comp_idx] = real + 1j * imag
    return out


def _prepare_chain_cache(cache: dict[str, object]) -> dict[str, np.ndarray]:
    chain = cache["chain"]
    zin = np.asarray(cache["zin"], dtype=complex)
    zin_z = np.asarray(cache["zin_z"], dtype=complex)
    z_ant = np.asarray(cache["params"]["Z_ant"], dtype=complex)
    freq_count = z_ant.shape[0]
    inverse_xy = np.zeros((freq_count, 2, 2), dtype=complex)
    inverse_z = np.zeros((freq_count, 2, 2), dtype=complex)
    for freq_idx in range(freq_count):
        inverse_xy[freq_idx] = np.linalg.pinv(chain["a_total"][freq_idx] @ chain["a_balun_d"][freq_idx])
        inverse_z[freq_idx] = np.linalg.pinv(chain["a_total_z"][freq_idx] @ chain["a_balun_d"][freq_idx])
    return {
        "inverse_xy": inverse_xy,
        "inverse_z": inverse_z,
        "zin": zin,
        "zin_z": zin_z,
        "z_ant": z_ant,
    }


def _rfchain_spectrum_batch(efl: np.ndarray, e_complex: np.ndarray, chain_cache: dict[str, np.ndarray]) -> np.ndarray:
    voc = np.einsum("fpc,mfc->mfp", efl, e_complex)
    zin = chain_cache["zin"]
    zin_z = chain_cache["zin_z"]
    z_ant = chain_cache["z_ant"]

    denominator = np.empty_like(voc)
    terminal_impedance = np.empty_like(voc)
    denominator[:, :, :2] = zin[None, :, None] + z_ant[None, :, :2]
    denominator[:, :, 2] = zin_z[None, :] + z_ant[None, :, 2]
    terminal_impedance[:, :, :2] = zin[None, :, None]
    terminal_impedance[:, :, 2] = zin_z[None, :]

    current = voc / denominator
    terminal_voltage = current * terminal_impedance
    v_adc = np.empty_like(terminal_voltage)
    inverse_xy = chain_cache["inverse_xy"]
    inverse_z = chain_cache["inverse_z"]
    v_adc[:, :, :2] = (
        inverse_xy[None, :, 0, 0, None] * terminal_voltage[:, :, :2]
        + inverse_xy[None, :, 0, 1, None] * current[:, :, :2]
    )
    v_adc[:, :, 2] = inverse_z[None, :, 0, 0] * terminal_voltage[:, :, 2] + inverse_z[None, :, 0, 1] * current[:, :, 2]
    return v_adc


def _spectra_to_time_batch(
    module: ModuleType,
    spectra: np.ndarray,
    fs: float,
    n_samples: int,
    freq_all: np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of ifft_grid_to_n for batched spectra."""

    n_traces, _n_freq, n_ports = spectra.shape
    npos = n_samples // 2 + 1 if n_samples % 2 == 0 else (n_samples + 1) // 2
    f_pos = np.arange(npos, dtype=float) * fs / n_samples
    f_grid = module.make_1mhz_grid(fs)
    grid = np.full((n_traces, len(f_grid), n_ports), 1e-9 + 0.0j, dtype=complex)

    k0 = int(round(float(freq_all[0] * 1e9) / 1e6))
    k1 = int(round(float(freq_all[-1] * 1e9) / 1e6))
    k0 = max(0, min(k0, len(f_grid) - 1))
    k1 = max(0, min(k1, len(f_grid) - 1))
    if k1 - k0 + 1 != spectra.shape[1]:
        raise ValueError("Frequency grid length mismatch while embedding RF-chain spectra.")
    grid[:, k0 : k1 + 1, :] = spectra

    nearest = np.searchsorted(f_grid, f_pos, side="left")
    nearest = np.clip(nearest, 0, len(f_grid) - 1)
    left = np.clip(nearest - 1, 0, len(f_grid) - 1)
    choose_left = np.abs(f_grid[left] - f_pos) <= np.abs(f_grid[nearest] - f_pos)
    nearest = np.where(choose_left, left, nearest)
    xpos_scaled = grid[:, nearest, :]
    xpos_raw = xpos_scaled * (n_samples / 2.0)
    xpos_raw[:, 0, :] = np.real(xpos_raw[:, 0, :]) * 2.0
    if n_samples % 2 == 0:
        xpos_raw[:, -1, :] = np.real(xpos_raw[:, -1, :]) * 2.0

    full = np.zeros((n_traces, n_samples, n_ports), dtype=complex)
    full[:, :npos, :] = xpos_raw
    if n_samples % 2 == 0:
        full[:, npos:, :] = np.conj(xpos_raw[:, -2:0:-1, :])
    else:
        full[:, npos:, :] = np.conj(xpos_raw[:, -1:0:-1, :])
    return np.fft.ifft(full, axis=1).real


def _bilinear_efl(
    efl_all: np.ndarray,
    theta_axis: np.ndarray,
    phi_axis: np.ndarray,
    theta_deg: float,
    phi_deg: float,
) -> np.ndarray:
    """Vectorized bilinear interpolation of the regular EFL theta-phi table."""

    theta_axis = np.asarray(theta_axis, dtype=float)
    phi_axis = np.asarray(phi_axis, dtype=float)
    if theta_axis.ndim != 1 or phi_axis.ndim != 1:
        raise ValueError("Theta and Phi axes must be one-dimensional.")
    if np.any(np.diff(theta_axis) <= 0.0) or np.any(np.diff(phi_axis) <= 0.0):
        raise ValueError("Theta and Phi axes must be strictly increasing.")

    it0, it1, wt = _bracket_axis(theta_axis, float(theta_deg), periodic=False)
    ip0, ip1, wp = _bracket_axis(phi_axis, float(phi_deg), periodic=True)

    e00 = efl_all[:, :, :, it0, ip0]
    e10 = efl_all[:, :, :, it1, ip0]
    e01 = efl_all[:, :, :, it0, ip1]
    e11 = efl_all[:, :, :, it1, ip1]
    return (
        (1.0 - wt) * (1.0 - wp) * e00
        + wt * (1.0 - wp) * e10
        + (1.0 - wt) * wp * e01
        + wt * wp * e11
    )


def _bracket_axis(axis: np.ndarray, value: float, *, periodic: bool) -> tuple[int, int, float]:
    """Find lower/upper indices and interpolation weight on a sorted axis."""

    axis = np.asarray(axis, dtype=float)
    if periodic:
        period = 360.0
        value = value % period
        idx = int(np.searchsorted(axis, value, side="right") - 1)
        if idx < 0:
            idx = len(axis) - 1
            upper = axis[0] + period
            lower = axis[idx]
            weight = (value + period - lower) / (upper - lower)
            return idx, 0, float(weight)
        if idx >= len(axis) - 1:
            lower = axis[-1]
            upper = axis[0] + period
            weight = (value - lower) / (upper - lower)
            return len(axis) - 1, 0, float(weight)
    else:
        value = float(np.clip(value, axis[0], axis[-1]))
        idx = int(np.searchsorted(axis, value, side="right") - 1)
        idx = max(0, min(idx, len(axis) - 2))

    lower = axis[idx]
    upper = axis[idx + 1]
    weight = 0.0 if upper == lower else (value - lower) / (upper - lower)
    return idx, idx + 1, float(weight)
