#!/usr/bin/env python3
"""
Python port of batch_forward_rfchain_save_mat.m

Features
- Recursively scan simulated event folders under finish/
- Forward model E(t) -> E(f) -> Voc(f) -> Vad(f) -> Vad(t)
- Save *_adc.trace and *_fwd.mat
- Change output filename suffix based on noise switches
- Optionally generate comparison figures

Local dependencies
- Uses project-local .pydeps if present
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYVER = f"{sys.version_info.major}{sys.version_info.minor}"
PYDEPS_CANDIDATES = [
    PROJECT_ROOT / f".pydeps-py{PYVER}",
    PROJECT_ROOT / ".pydeps",
]
for _pydeps in PYDEPS_CANDIDATES:
    if _pydeps.exists():
        sys.path.insert(0, str(_pydeps))
        break

import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d

@dataclass
class Config:
    sim_dir: Path
    params_dir: Path
    freq_all_ghz: np.ndarray
    noiseflag: bool
    innernoise_flag: bool
    lst_hour: int
    lst_minute: int
    overwrite: bool
    band_lo_hz: float
    band_hi_hz: float
    edge_taper_hz: float
    visualize: bool
    max_plots: int
    compare_clean: bool
    max_folders: Optional[int]
    max_traces: Optional[int]
    folder_contains: Optional[str]
    folders_file: Optional[Path]
    adc_only: bool
    legacy_adc_names: bool


@dataclass
class ChainCache:
    freq_all_ghz: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    efl_all: np.ndarray
    z_ant: np.ndarray
    pg: np.ndarray
    data_total_seperate: np.ndarray
    zin_xy: np.ndarray
    zin_z: np.ndarray
    z_ad: np.ndarray
    a_total_xy: np.ndarray
    a_total_z: np.ndarray
    a_balun_d: np.ndarray
    chain_xy: np.ndarray
    chain_z: np.ndarray


@dataclass
class TraceResult:
    e_shower_downsample: np.ndarray
    e_complex: np.ndarray
    voc: np.ndarray
    voc_t: np.ndarray
    vadc: np.ndarray
    vadc_t: np.ndarray
    fs_ds: float
    suffix: str


def output_suffix(noiseflag: bool, innernoise_flag: bool, legacy_adc_names: bool = False) -> str:
    if legacy_adc_names and (noiseflag or innernoise_flag):
        return ""
    if noiseflag and innernoise_flag:
        return "_gn_in"
    if noiseflag:
        return "_gn"
    if innernoise_flag:
        return "_in"
    return "_clean"


def parse_folder_meta(folder_name: str) -> Optional[Tuple[float, float, float]]:
    parts = folder_name.split("_")
    if len(parts) < 7:
        return None
    try:
        energy = float(parts[4])
        zenith = float(parts[5])
        azimuth = float(parts[6])
    except ValueError:
        return None
    return energy, zenith, azimuth


def read_numeric_table(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith(("!", "#", "//")):
                continue
            parts = s.split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no numeric rows parsed from {path}")
    return np.asarray(rows, dtype=np.float64)


def read_s2p(path: Path) -> np.ndarray:
    arr = read_numeric_table(path)
    if arr.shape[1] < 9:
        raise ValueError(f"s2p file needs at least 9 columns: {path}")
    return arr[:, :9]


def read_s1p(path: Path) -> np.ndarray:
    arr = read_numeric_table(path)
    if arr.shape[1] < 3:
        raise ValueError(f"s1p file needs at least 3 columns: {path}")
    return arr[:, :3]


def complex_interp_linear(x: np.ndarray, y: np.ndarray, xq: np.ndarray, fill_value: complex = 0j) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    xq = np.asarray(xq, dtype=np.float64).reshape(-1)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]

    out = np.zeros((xq.size, y.shape[1]), dtype=np.complex128)
    for i in range(y.shape[1]):
        fr = interp1d(
            x,
            np.real(y[:, i]),
            kind="linear",
            bounds_error=False,
            fill_value=np.real(fill_value),
            assume_sorted=True,
        )
        fi = interp1d(
            x,
            np.imag(y[:, i]),
            kind="linear",
            bounds_error=False,
            fill_value=np.imag(fill_value),
            assume_sorted=True,
        )
        out[:, i] = fr(xq) + 1j * fi(xq)
    return out[:, 0] if out.shape[1] == 1 else out


def complex_interp_cubic(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    xq = np.asarray(xq, dtype=np.float64).reshape(-1)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    out = np.zeros((xq.size, y.shape[1]), dtype=np.complex128)
    kind = "cubic" if x.size >= 4 else "linear"
    for i in range(y.shape[1]):
        fr = interp1d(x, np.real(y[:, i]), kind=kind, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
        fi = interp1d(x, np.imag(y[:, i]), kind=kind, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
        out[:, i] = fr(xq) + 1j * fi(xq)
    return out[:, 0] if out.shape[1] == 1 else out


def complex_interp_nd(x: np.ndarray, y: np.ndarray, xq: np.ndarray, fill_value: complex = 0j) -> np.ndarray:
    y = np.asarray(y)
    trailing_shape = y.shape[1:]
    y_flat = y.reshape(y.shape[0], -1)
    out = complex_interp_linear(x, y_flat, xq, fill_value=fill_value)
    return out.reshape(len(np.asarray(xq).reshape(-1)), *trailing_shape)


def one_sided_band_taper(freq_hz: np.ndarray, flo_hz: float, fhi_hz: float, edge_hz: float) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=np.float64).reshape(-1)
    w = np.zeros_like(freq_hz)
    if edge_hz <= 0:
        sel = (freq_hz >= flo_hz) & (freq_hz <= fhi_hz)
        w[sel] = 1.0
        return w

    flat_lo = flo_hz + edge_hz
    flat_hi = fhi_hz - edge_hz
    w[(freq_hz >= flat_lo) & (freq_hz <= flat_hi)] = 1.0

    ilo = (freq_hz >= flo_hz) & (freq_hz < flat_lo)
    if np.any(ilo):
        x = (freq_hz[ilo] - flo_hz) / edge_hz
        w[ilo] = 0.5 - 0.5 * np.cos(np.pi * x)

    ihi = (freq_hz > flat_hi) & (freq_hz <= fhi_hz)
    if np.any(ihi):
        x = (fhi_hz - freq_hz[ihi]) / edge_hz
        w[ihi] = 0.5 - 0.5 * np.cos(np.pi * x)
    return w


def s2p_dbang_to_complex(raw: np.ndarray) -> np.ndarray:
    out = np.zeros((raw.shape[0], 4), dtype=np.complex128)
    for i, c in enumerate((1, 3, 5, 7)):
        out[:, i] = 10.0 ** (raw[:, c] / 20.0) * np.exp(1j * np.deg2rad(raw[:, c + 1]))
    return out


def s2p_magang_to_complex(raw: np.ndarray) -> np.ndarray:
    out = np.zeros((raw.shape[0], 4), dtype=np.complex128)
    for i, c in enumerate((1, 3, 5, 7)):
        out[:, i] = raw[:, c] * np.exp(1j * np.deg2rad(raw[:, c + 1]))
    return out


def a_from_s_scaled(srow: np.ndarray) -> np.ndarray:
    s11, s21, s12, s22 = srow
    a0 = np.array(
        [
            [((1 + s11) * (1 - s22) + s12 * s21) / (2 * s21), ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)],
            [((1 - s11) * (1 - s22) - s12 * s21) / (2 * s21), ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21)],
        ],
        dtype=np.complex128,
    )
    return a0 * np.array([[1.0, 50.0], [1.0 / 50.0, 1.0]], dtype=np.complex128)


def load_chain_cache(cfg: Config) -> ChainCache:
    freq_all_ghz = cfg.freq_all_ghz.reshape(-1)
    freq_all_hz = freq_all_ghz * 1e9

    efl_mat = loadmat(cfg.params_dir / "EFL_data.mat", variable_names=["EFL_all", "Theta", "Phi", "freq_ALL"])
    efl_all = np.asarray(efl_mat["EFL_all"], dtype=np.complex128)
    theta = np.asarray(efl_mat["Theta"]).reshape(-1).astype(np.float64)
    phi = np.asarray(efl_mat["Phi"]).reshape(-1).astype(np.float64)

    z_ant_raw = np.loadtxt(cfg.params_dir / "Z_ant_3.2m.csv", delimiter=",", skiprows=1)
    z_ant = np.column_stack(
        [
            z_ant_raw[:, 1] + 1j * z_ant_raw[:, 2],
            z_ant_raw[:, 3] + 1j * z_ant_raw[:, 4],
            z_ant_raw[:, 5] + 1j * z_ant_raw[:, 6],
        ]
    ).astype(np.complex128)

    pg_mat = loadmat(cfg.params_dir / "PG_ALL_jifen_perminute.mat")
    pg_key = next(k for k, v in pg_mat.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 4)
    pg = np.asarray(pg_mat[pg_key], dtype=np.float64)

    inner_mat = loadmat(cfg.params_dir / "Data_save_all_withAM.mat", variable_names=["Data_total_seperate"])
    data_total_seperate = np.asarray(inner_mat["Data_total_seperate"], dtype=np.float64)

    balun_a = read_s2p(cfg.params_dir / "balun13in20230612.s2p")
    balun_d = read_s2p(cfg.params_dir / "balun_before_ad.s2p")
    lna = read_s2p(cfg.params_dir / "LNA-X.s2p")
    cable = read_s2p(cfg.params_dir / "cable+Connector.s2p")
    vga = read_s2p(cfg.params_dir / "feb+amfitler+biast.s2p")
    adload = read_s1p(cfg.params_dir / "S_balun_AD.s1p")
    mn_xy = read_s2p(cfg.params_dir / "MatchingNetworkX.s2p")
    mn_z = read_s2p(cfg.params_dir / "MatchingNetworkZ.s2p")

    lna_complex = s2p_dbang_to_complex(lna)
    cable_complex = s2p_dbang_to_complex(cable)
    vga_complex = s2p_dbang_to_complex(vga)
    balun_a_complex = s2p_magang_to_complex(balun_a)
    balun_d_complex = s2p_magang_to_complex(balun_d)
    mn_xy_complex = s2p_magang_to_complex(mn_xy)
    mn_z_complex = s2p_magang_to_complex(mn_z)
    load_complex = 10.0 ** (adload[:, 1] / 20.0) * np.exp(1j * np.deg2rad(adload[:, 2]))

    lamda = 3e8 / freq_all_hz
    theta1 = 0.3 * math.sqrt(2.2) * 2.0 * math.pi / lamda

    s_lna = complex_interp_cubic(lna[:, 0], lna_complex, freq_all_hz)
    s_cable = complex_interp_cubic(cable[:, 0], cable_complex, freq_all_hz)
    s_vga = complex_interp_cubic(vga[:, 0], vga_complex, freq_all_hz)
    s_balun_a = complex_interp_cubic(balun_a[:, 0], balun_a_complex, freq_all_hz)
    s_balun_d = complex_interp_cubic(balun_d[:, 0], balun_d_complex, freq_all_hz)
    s_mn_xy = complex_interp_cubic(mn_xy[:, 0], mn_xy_complex, freq_all_hz)
    s_mn_z = complex_interp_cubic(mn_z[:, 0], mn_z_complex, freq_all_hz)
    s_load = complex_interp_cubic(adload[:, 0], load_complex, freq_all_hz)

    s_vga[:, 2] *= np.exp(1j * theta1)
    s_vga[:, 3] *= np.exp(2j * theta1)
    s_load *= np.exp(2j * theta1)

    freq_num = freq_all_hz.size
    a_total_xy = np.zeros((freq_num, 2, 2), dtype=np.complex128)
    a_total_z = np.zeros((freq_num, 2, 2), dtype=np.complex128)
    a_balun_d_cache = np.zeros((freq_num, 2, 2), dtype=np.complex128)
    chain_xy = np.zeros((freq_num, 2, 2), dtype=np.complex128)
    chain_z = np.zeros((freq_num, 2, 2), dtype=np.complex128)
    zin_xy = np.zeros(freq_num, dtype=np.complex128)
    zin_z = np.zeros(freq_num, dtype=np.complex128)
    z_ad = np.zeros(freq_num, dtype=np.complex128)
    z_load = 50.0 * (1.0 + s_load) / (1.0 - s_load)

    for i in range(freq_num):
        a_lna = a_from_s_scaled(s_lna[i, :])
        a_cable = a_from_s_scaled(s_cable[i, :])
        a_vga_i = a_from_s_scaled(s_vga[i, :])
        a_balun_a_i = a_from_s_scaled(s_balun_a[i, :])
        a_balun_d_i = a_from_s_scaled(s_balun_d[i, :])
        a_mn_xy_i = a_from_s_scaled(s_mn_xy[i, :])
        a_mn_z_i = a_from_s_scaled(s_mn_z[i, :])

        a_total_xy[i] = a_balun_a_i @ a_mn_xy_i @ a_lna @ a_cable @ a_vga_i
        a_total_z[i] = a_balun_a_i @ a_mn_z_i @ a_lna @ a_cable @ a_vga_i
        a_balun_d_cache[i] = a_balun_d_i
        chain_xy[i] = a_total_xy[i] @ a_balun_d_i
        chain_z[i] = a_total_z[i] @ a_balun_d_i

        zin_xy[i] = (a_total_xy[i, 0, 0] * z_load[i] + a_total_xy[i, 0, 1]) / (a_total_xy[i, 1, 0] * z_load[i] + a_total_xy[i, 1, 1])
        zin_z[i] = (a_total_z[i, 0, 0] * z_load[i] + a_total_z[i, 0, 1]) / (a_total_z[i, 1, 0] * z_load[i] + a_total_z[i, 1, 1])
        z_ad[i] = (a_balun_d_i[0, 1] - z_load[i] * a_balun_d_i[1, 1]) / (a_balun_d_i[1, 0] * z_load[i] - a_balun_d_i[0, 0])

    return ChainCache(
        freq_all_ghz=freq_all_ghz,
        theta=theta,
        phi=phi,
        efl_all=efl_all,
        z_ant=z_ant,
        pg=pg,
        data_total_seperate=data_total_seperate,
        zin_xy=zin_xy,
        zin_z=zin_z,
        z_ad=z_ad,
        a_total_xy=a_total_xy,
        a_total_z=a_total_z,
        a_balun_d=a_balun_d_cache,
        chain_xy=chain_xy,
        chain_z=chain_z,
    )


def pick_efl_nearest(cache: ChainCache, theta_deg: float, phi_deg: float) -> np.ndarray:
    ith = int(np.argmin(np.abs(cache.theta - theta_deg)))
    iph = int(np.argmin(np.abs(cache.phi - (phi_deg % 360.0))))
    return cache.efl_all[:, :, :, ith, iph]


def galactic_noise_221_uV(cache: ChainCache, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    psd = cache.pg[:, :, cfg.lst_hour, cfg.lst_minute]
    p = psd * 1e6
    r = np.real(cache.z_ant)
    v_amp = 2.0 * np.sqrt(np.maximum(p * r, 0.0)) * 1e6
    return rng.normal(0.0, v_amp) * np.exp(1j * (2.0 * np.pi * rng.random(v_amp.shape)))


def resample_to_length(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == n:
        return x.copy()
    t0 = np.arange(x.size, dtype=np.float64)
    t1 = np.linspace(0.0, x.size - 1, n)
    return np.interp(t1, t0, x)


def inner_noise_3ch_uV(cache: ChainCache, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = int(rng.integers(0, cache.data_total_seperate.shape[1]))
    out = np.zeros((n, 3), dtype=np.float64)
    for ch in range(3):
        x = cache.data_total_seperate[0, idx, ch + 1, :] * 1e6
        out[:, ch] = resample_to_length(np.asarray(x, dtype=np.float64), n)
    return out


def rfft_scaled(x_t: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    x_t = np.asarray(x_t, dtype=np.float64)
    if x_t.ndim == 1:
        x_t = x_t[:, None]
    n = x_t.shape[0]
    xpos = np.fft.rfft(x_t, axis=0) * (2.0 / n)
    xpos[0, :] /= 2.0
    if n % 2 == 0:
        xpos[-1, :] /= 2.0
    f_pos = np.fft.rfftfreq(n, d=1.0 / fs)
    return f_pos, xpos


def irfft_scaled(xpos_scaled: np.ndarray, n: int) -> np.ndarray:
    xpos_scaled = np.asarray(xpos_scaled, dtype=np.complex128)
    if xpos_scaled.ndim == 1:
        xpos_scaled = xpos_scaled[:, None]
    raw = xpos_scaled * (n / 2.0)
    raw[0, :] = np.real(raw[0, :]) * 2.0
    if n % 2 == 0:
        raw[-1, :] = np.real(raw[-1, :]) * 2.0
    x_t = np.fft.irfft(raw, n=n, axis=0)
    return np.real(x_t)


def choose_decimation(fs: float, band_hi_hz: float) -> Tuple[int, float]:
    target_fs = 500e6
    decim = max(1, int(round(fs / target_fs)))
    fs_ds = fs / decim
    if fs_ds < 2.0 * band_hi_hz:
        decim = max(1, int(math.floor(fs / (2.0 * band_hi_hz))))
        fs_ds = fs / decim
    return decim, fs_ds


def best_decimation_offset(x_t: np.ndarray, decim: int) -> int:
    if decim <= 1:
        return 0
    best_off = 0
    best_peak = -np.inf
    for off in range(decim):
        xs = x_t[off::decim, :]
        peak = float(np.max(np.abs(xs)))
        if peak > best_peak:
            best_peak = peak
            best_off = off
    return best_off


def decimate_time_series(t_ns: np.ndarray, x_t: np.ndarray, decim: int, offset: int) -> np.ndarray:
    return np.column_stack([t_ns[offset::decim], x_t[offset::decim, :]])


def suppress_late_echo(vadc_t: np.ndarray, fs: float) -> np.ndarray:
    vadc_t = np.asarray(vadc_t, dtype=np.float64)
    if vadc_t.size == 0:
        return vadc_t

    envelope = np.sqrt(np.sum(vadc_t**2, axis=1))
    peak_idx = int(np.argmax(envelope))
    keep_tail = max(8, int(round(fs * 320e-9)))
    taper = max(8, int(round(fs * 120e-9)))

    start = min(vadc_t.shape[0], peak_idx + keep_tail)
    stop = min(vadc_t.shape[0], start + taper)
    window = np.ones(vadc_t.shape[0], dtype=np.float64)
    if stop > start:
        x = np.linspace(0.0, 1.0, stop - start, endpoint=False)
        window[start:stop] = 0.5 + 0.5 * np.cos(np.pi * x)
    if stop < vadc_t.shape[0]:
        window[stop:] = 0.0
    return vadc_t * window[:, None]


def galactic_noise_bins_uV(
    cache: ChainCache,
    cfg: Config,
    f_pos_hz: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    psd = cache.pg[:, :, cfg.lst_hour, cfg.lst_minute] * 1e6
    r = np.real(cache.z_ant)
    amp_221 = 2.0 * np.sqrt(np.maximum(psd * r, 0.0)) * 1e6
    f_src = cache.freq_all_ghz * 1e9
    amp_bins = np.zeros((len(f_pos_hz), 3), dtype=np.float64)
    for ch in range(3):
        amp_bins[:, ch] = np.interp(f_pos_hz, f_src, amp_221[:, ch], left=0.0, right=0.0)
    noise = rng.normal(0.0, amp_bins) * np.exp(1j * (2.0 * np.pi * rng.random(amp_bins.shape)))
    noise[0, :] = np.real(noise[0, :])
    if len(f_pos_hz) > 1 and np.isclose(f_pos_hz[-1], 0.5 * (f_pos_hz[1] - f_pos_hz[0]) * (2 * (len(f_pos_hz) - 1))):
        noise[-1, :] = np.real(noise[-1, :])
    return noise


def fft_Nto221(x_t: np.ndarray, fs: float, freq_all_ghz: np.ndarray) -> np.ndarray:
    x_t = np.asarray(x_t)
    if x_t.ndim == 1:
        x_t = x_t[:, None]
    n = x_t.shape[0]
    x = np.fft.fft(x_t, axis=0)
    npos = n // 2 + 1 if n % 2 == 0 else (n + 1) // 2
    xpos = x[:npos, :] / n * 2.0
    xpos[0, :] /= 2.0
    if n % 2 == 0:
        xpos[-1, :] /= 2.0
    f_pos = np.arange(npos, dtype=np.float64) * (fs / n)
    f_tar = freq_all_ghz.reshape(-1) * 1e9
    return complex_interp_linear(f_pos, xpos, f_tar, fill_value=0j)


def ifft_221toN(x221: np.ndarray, fs: float, n: int, freq_all_ghz: np.ndarray, edge_taper_hz: float) -> np.ndarray:
    x221 = np.asarray(x221, dtype=np.complex128).reshape(-1)
    f_src = freq_all_ghz.reshape(-1) * 1e9
    if edge_taper_hz > 0:
        x221 = x221 * one_sided_band_taper(f_src, f_src[0], f_src[-1], edge_taper_hz)

    npos = n // 2 + 1 if n % 2 == 0 else (n + 1) // 2
    f_pos = np.arange(npos, dtype=np.float64) * (fs / n)
    xpos_scaled = complex_interp_linear(f_src, x221, f_pos, fill_value=0j).reshape(-1)

    xpos_raw = xpos_scaled * (n / 2.0)
    xpos_raw[0] = np.real(xpos_raw[0]) * 2.0
    if n % 2 == 0:
        xpos_raw[-1] = np.real(xpos_raw[-1]) * 2.0

    xfull = np.zeros(n, dtype=np.complex128)
    xfull[:npos] = xpos_raw
    if n % 2 == 0:
        xfull[npos:] = np.conj(xpos_raw[-2:0:-1])
    else:
        xfull[npos:] = np.conj(xpos_raw[-1:0:-1])
    return np.real(np.fft.ifft(xfull))


def fft_trace_to_221(trace: np.ndarray, fs: float, cfg: Config) -> Tuple[np.ndarray, np.ndarray, float]:
    t_ns = trace[:, 0]
    x = trace[:, 1:4]
    n = x.shape[0]

    x_full = np.fft.fft(x, axis=0)
    f = np.arange(n, dtype=np.float64) * (fs / n)
    f_alias = np.minimum(f, np.abs(fs - f))
    w = one_sided_band_taper(f_alias, cfg.band_lo_hz, cfg.band_hi_hz, cfg.edge_taper_hz)
    x_bp = np.real(np.fft.ifft(x_full * w[:, None], axis=0))

    target_fs = 500e6
    decim = max(1, int(math.floor(fs / target_fs)))
    fs_ds = fs / decim
    if fs_ds < 2.0 * cfg.band_hi_hz:
        decim = max(1, int(math.floor(fs / (2.0 * cfg.band_hi_hz))))
        fs_ds = fs / decim

    best_off = 0
    best_peak = -np.inf
    for off in range(decim):
        xs = x_bp[off::decim, :]
        peak = float(np.max(np.abs(xs)))
        if peak > best_peak:
            best_peak = peak
            best_off = off

    idx = slice(best_off, None, decim)
    ds = np.column_stack([t_ns[idx], x_bp[idx, :]])
    e221 = fft_Nto221(ds[:, 1:4], fs_ds, cfg.freq_all_ghz)
    return ds, e221, fs_ds


def raw_fft_to_freqs(x_t: np.ndarray, fs: float, freq_all_ghz: np.ndarray) -> np.ndarray:
    x_t = np.asarray(x_t, dtype=np.float64)
    if x_t.ndim == 1:
        x_t = x_t[:, None]
    n = x_t.shape[0]
    x_pos = np.fft.rfft(x_t, axis=0)
    f_pos = np.fft.rfftfreq(n, d=1.0 / fs)
    f_tar = np.asarray(freq_all_ghz, dtype=np.float64).reshape(-1) * 1e9
    return complex_interp_linear(f_pos, x_pos, f_tar, fill_value=0j)


def fft_trace_to_native(
    trace: np.ndarray,
    fs: float,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    t_ns = trace[:, 0]
    x = trace[:, 1:4]
    n = x.shape[0]

    x_full = np.fft.fft(x, axis=0)
    f = np.arange(n, dtype=np.float64) * (fs / n)
    f_alias = np.minimum(f, np.abs(fs - f))
    w = one_sided_band_taper(f_alias, cfg.band_lo_hz, cfg.band_hi_hz, cfg.edge_taper_hz)
    x_bp = np.real(np.fft.ifft(x_full * w[:, None], axis=0))

    decim, fs_ds = choose_decimation(fs, cfg.band_hi_hz)
    best_off = best_decimation_offset(x_bp, decim)
    ds = decimate_time_series(t_ns, x_bp, decim, best_off)

    e_pos = np.fft.rfft(ds[:, 1:4], axis=0)
    f_pos = np.fft.rfftfreq(ds.shape[0], d=1.0 / fs_ds)
    use_pos = (f_pos >= cfg.band_lo_hz) & (f_pos <= cfg.band_hi_hz)
    e_221 = raw_fft_to_freqs(ds[:, 1:4], fs_ds, cfg.freq_all_ghz)
    return ds, e_pos, f_pos, use_pos, e_221, fs_ds


def voc2vad_inverse_at_freqs(voc: np.ndarray, cache: ChainCache, f_hz: np.ndarray) -> np.ndarray:
    f_src = cache.freq_all_ghz * 1e9
    f_hz = np.asarray(f_hz, dtype=np.float64).reshape(-1)
    voc = np.asarray(voc, dtype=np.complex128)
    if voc.shape != (f_hz.size, 3):
        raise ValueError("voc shape must be (len(f_hz), 3)")

    z_ant = complex_interp_linear(f_src, cache.z_ant, f_hz, fill_value=0j)
    zin_xy = complex_interp_linear(f_src, cache.zin_xy, f_hz, fill_value=0j)
    zin_z = complex_interp_linear(f_src, cache.zin_z, f_hz, fill_value=0j)
    z_ad = complex_interp_linear(f_src, cache.z_ad, f_hz, fill_value=0j)
    chain_xy = complex_interp_nd(f_src, cache.chain_xy, f_hz, fill_value=0j)
    chain_z = complex_interp_nd(f_src, cache.chain_z, f_hz, fill_value=0j)

    vadc = np.zeros_like(voc)
    i1x = voc[:, 0] / (zin_xy + z_ant[:, 0])
    i1y = voc[:, 1] / (zin_xy + z_ant[:, 1])
    i1z = voc[:, 2] / (zin_z + z_ant[:, 2])

    denom_xy = chain_xy[:, 1, 0] + chain_xy[:, 1, 1] / z_ad
    denom_z = chain_z[:, 1, 0] + chain_z[:, 1, 1] / z_ad
    vadc[:, 0] = i1x / denom_xy
    vadc[:, 1] = i1y / denom_xy
    vadc[:, 2] = i1z / denom_z
    return vadc


def voc2vad_inverse(voc: np.ndarray, cache: ChainCache) -> np.ndarray:
    return voc2vad_inverse_at_freqs(voc, cache, cache.freq_all_ghz * 1e9)


def forward_trace(trace: np.ndarray, theta_deg: float, phi_deg: float, cache: ChainCache, cfg: Config, rng: np.random.Generator) -> TraceResult:
    dt_s = (trace[1, 0] - trace[0, 0]) * 1e-9
    fs = 1.0 / dt_s
    e_ds, e_pos, f_pos, use_pos, e_complex, fs_ds = fft_trace_to_native(trace, fs, cfg)
    if not np.any(use_pos):
        raise RuntimeError("no native FFT bins selected inside RF band")

    f_use = f_pos[use_pos]
    e_use = e_pos[use_pos, :]
    f_src = cache.freq_all_ghz * 1e9
    use_efl = complex_interp_nd(f_src, pick_efl_nearest(cache, theta_deg, phi_deg), f_use, fill_value=0j)
    voc_use = np.einsum("fpc,fc->fp", use_efl, e_use)

    if cfg.noiseflag:
        voc_use = voc_use + galactic_noise_bins_uV(cache, cfg, f_use, rng)

    vadc_use = voc2vad_inverse_at_freqs(voc_use, cache, f_use)

    n_ds = e_ds.shape[0]
    voc_pos = np.zeros((f_pos.size, 3), dtype=np.complex128)
    vadc_pos = np.zeros((f_pos.size, 3), dtype=np.complex128)
    voc_pos[use_pos, :] = voc_use
    vadc_pos[use_pos, :] = vadc_use

    voc_t = np.real(np.fft.irfft(voc_pos, n=n_ds, axis=0))
    vadc_t = np.real(np.fft.irfft(vadc_pos, n=n_ds, axis=0))

    if cfg.innernoise_flag:
        vadc_t = vadc_t + inner_noise_3ch_uV(cache, n_ds, rng)
        vadc_pos = np.fft.rfft(vadc_t, axis=0)

    voc = complex_interp_linear(f_pos, voc_pos, cfg.freq_all_ghz * 1e9, fill_value=0j)
    vadc = complex_interp_linear(f_pos, vadc_pos, cfg.freq_all_ghz * 1e9, fill_value=0j)

    return TraceResult(
        e_shower_downsample=e_ds,
        e_complex=e_complex,
        voc=voc,
        voc_t=voc_t,
        vadc=vadc,
        vadc_t=vadc_t,
        fs_ds=fs_ds,
        suffix=output_suffix(cfg.noiseflag, cfg.innernoise_flag, cfg.legacy_adc_names),
    )


def write_adc_trace(path: Path, t_ns: np.ndarray, vadc_uv: np.ndarray) -> None:
    data = np.column_stack([t_ns.reshape(-1), vadc_uv])
    np.savetxt(path, data, fmt="%.15g")


def read_adc_trace(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(path, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] < 4:
        return None
    return arr[:, :4]


def save_forward_mat(path: Path, result: TraceResult, cfg: Config) -> None:
    savemat(
        path,
        {
            "E_complex": result.e_complex,
            "E_shower_downsample": result.e_shower_downsample,
            "Vadc": result.vadc,
            "Vadc_t": result.vadc_t,
            "Voc": result.voc,
            "Voc_t": result.voc_t,
            "freq_ALL": cfg.freq_all_ghz.reshape(1, -1),
        },
        do_compression=False,
    )


def maybe_make_plot(
    plot_path: Path,
    raw_trace: np.ndarray,
    result: TraceResult,
    clean_ref: Optional[TraceResult],
    previous_adc: Optional[np.ndarray],
    previous_label: Optional[str],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    t_raw = raw_trace[:, 0]
    t_ds = result.e_shower_downsample[:, 0]
    channel_names = ["Ch1", "Ch2", "Ch3"]

    for ch in range(3):
        ax_l = axes[ch, 0]
        ax_r = axes[ch, 1]

        ax_l.plot(t_raw, raw_trace[:, ch + 1], color="0.75", lw=1.0, label="raw E")
        ax_l.plot(t_ds, result.e_shower_downsample[:, ch + 1], color="#1f77b4", lw=1.2, label="bandpassed/downsampled E")
        ax_l.set_ylabel(f"{channel_names[ch]}\nE (uV/m)")
        ax_l.grid(True, alpha=0.25)
        if ch == 0:
            ax_l.legend(loc="upper right", fontsize=8)

        if clean_ref is not None:
            ax_r.plot(clean_ref.e_shower_downsample[:, 0], clean_ref.vadc_t[:, ch], color="#d62728", lw=1.2, label="clean adc")
        if previous_adc is not None:
            ax_r.plot(previous_adc[:, 0], previous_adc[:, ch + 1], color="#ff7f0e", lw=1.1, alpha=0.9, label=previous_label or "previous adc")
        ax_r.plot(t_ds, result.vadc_t[:, ch], color="#2ca02c", lw=1.2, label="current adc")
        ax_r.set_ylabel(f"{channel_names[ch]}\nADC (uV)")
        ax_r.grid(True, alpha=0.25)
        if ch == 0:
            ax_r.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Time (ns)")
    axes[-1, 1].set_xlabel("Time (ns)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def load_folder_names(folders_file: Optional[Path]) -> Optional[set[str]]:
    if folders_file is None:
        return None
    names: set[str] = set()
    with folders_file.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            name = line.strip()
            if not name:
                continue
            if name.endswith(".mat"):
                name = name[:-4]
            names.add(name)
    return names


def iter_event_folders(
    sim_dir: Path,
    folder_contains: Optional[str],
    folder_names: Optional[set[str]] = None,
) -> Iterable[Path]:
    for path in sorted(sim_dir.rglob("StshpCheckAngle_*")):
        if not path.is_dir():
            continue
        if folder_contains and folder_contains not in path.name:
            continue
        if folder_names is not None and path.name not in folder_names:
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Python RF-chain forward processor")
    parser.add_argument("--sim-dir", type=Path, default=Path(__file__).resolve().parent.parent / "finish")
    parser.add_argument("--params-dir", type=Path, default=Path(__file__).resolve().parent / "params")
    parser.add_argument("--noise-galactic", action="store_true", help="inject galactic noise")
    parser.add_argument("--noise-inner", action="store_true", help="inject internal noise")
    parser.add_argument("--lst-hour", type=int, default=11)
    parser.add_argument("--lst-minute", type=int, default=0)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="save comparison figures")
    parser.add_argument("--max-plots", type=int, default=20)
    parser.add_argument("--compare-clean", action="store_true", help="when current run is noisy, also compute a clean reference in plots")
    parser.add_argument("--max-folders", type=int, default=None)
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--folder-contains", type=str, default=None)
    parser.add_argument("--folders-file", type=Path, default=None, help="process only folder names listed in this file")
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--adc-only", action="store_true", help="only save *_adc.trace, skip *_fwd.mat")
    parser.add_argument("--legacy-adc-names", action="store_true", help="save noisy ADC as a*_adc.trace instead of a*_{noise}_adc.trace")
    args = parser.parse_args()

    cfg = Config(
        sim_dir=args.sim_dir,
        params_dir=args.params_dir,
        freq_all_ghz=np.arange(0.03, 0.251, 0.001, dtype=np.float64),
        noiseflag=args.noise_galactic,
        innernoise_flag=args.noise_inner,
        lst_hour=args.lst_hour,
        lst_minute=args.lst_minute,
        overwrite=not args.no_overwrite,
        band_lo_hz=30e6,
        band_hi_hz=250e6,
        edge_taper_hz=10e6,
        visualize=args.visualize,
        max_plots=args.max_plots,
        compare_clean=args.compare_clean,
        max_folders=args.max_folders,
        max_traces=args.max_traces,
        folder_contains=args.folder_contains,
        folders_file=args.folders_file,
        adc_only=args.adc_only,
        legacy_adc_names=args.legacy_adc_names,
    )

    if not cfg.sim_dir.is_dir():
        raise FileNotFoundError(f"sim dir not found: {cfg.sim_dir}")
    if not cfg.params_dir.is_dir():
        raise FileNotFoundError(f"params dir not found: {cfg.params_dir}")

    print(f"[INFO] loading RF-chain cache from {cfg.params_dir}")
    cache = load_chain_cache(cfg)
    rng = np.random.default_rng(args.seed)
    folder_names = load_folder_names(cfg.folders_file)

    plot_count = 0
    folder_count = 0
    trace_count = 0

    for folder in iter_event_folders(cfg.sim_dir, cfg.folder_contains, folder_names):
        meta = parse_folder_meta(folder.name)
        if meta is None:
            print(f"[WARN] skip unparsable folder: {folder}")
            continue
        energy, theta_deg, phi_deg = meta
        folder_count += 1
        print(f"\n[Folder] {folder.name} | energy={energy:.3f}, zenith={theta_deg:.3f}, azimuth={phi_deg:.3f}")

        for trace_path in sorted(folder.glob("a*.trace")):
            base = trace_path.stem
            if base.endswith("_adc"):
                continue

            suffix = output_suffix(cfg.noiseflag, cfg.innernoise_flag, cfg.legacy_adc_names)
            out_trace = folder / f"{base}{suffix}_adc.trace"
            out_mat = folder / f"{base}{suffix}_fwd.mat"
            out_plot = folder / f"{base}{suffix}_compare.png"
            previous_adc = None
            previous_label = None

            outputs_exist = out_trace.exists() if cfg.adc_only else (out_trace.exists() and out_mat.exists())
            if not cfg.overwrite and outputs_exist:
                print(f"  [Skip] {trace_path.name} (outputs exist)")
                continue

            raw_trace = np.loadtxt(trace_path, dtype=np.float64)
            if raw_trace.ndim != 2 or raw_trace.shape[1] < 4:
                print(f"  [FAIL] {trace_path.name} | trace must be [t(ns), Ex, Ey, Ez]")
                continue

            if cfg.visualize and cfg.compare_clean and not (cfg.noiseflag or cfg.innernoise_flag):
                previous_adc = read_adc_trace(out_trace)
                if previous_adc is not None:
                    previous_label = f"previous {out_trace.name}"
                else:
                    print(f"      compare note: no previous {out_trace.name} found, plot will only show current adc")

            result = forward_trace(raw_trace[:, :4], theta_deg, phi_deg, cache, cfg, rng)
            if not cfg.adc_only:
                save_forward_mat(out_mat, result, cfg)
            write_adc_trace(out_trace, result.e_shower_downsample[:, 0], result.vadc_t)
            trace_count += 1
            print(f"  [OK] {trace_path.name} -> {out_trace.name} | fs_ds={result.fs_ds/1e6:.3f} MHz, Nds={result.e_shower_downsample.shape[0]}")

            if cfg.visualize and plot_count < cfg.max_plots:
                clean_ref = None
                if cfg.compare_clean and (cfg.noiseflag or cfg.innernoise_flag):
                    clean_cfg = Config(**{**cfg.__dict__, "noiseflag": False, "innernoise_flag": False, "visualize": False})
                    clean_ref = forward_trace(raw_trace[:, :4], theta_deg, phi_deg, cache, clean_cfg, np.random.default_rng(args.seed))
                maybe_make_plot(
                    out_plot,
                    raw_trace[:, :4],
                    result,
                    clean_ref,
                    previous_adc,
                    previous_label,
                    f"{folder.name} | {base}{suffix}",
                )
                plot_count += 1
                print(f"      plot -> {out_plot.name}")

            if cfg.max_traces is not None and trace_count >= cfg.max_traces:
                print("[INFO] reached --max-traces limit")
                return 0

        if cfg.max_folders is not None and folder_count >= cfg.max_folders:
            print("[INFO] reached --max-folders limit")
            break

    print(f"\n[SUMMARY] folders={folder_count}, traces={trace_count}, plots={plot_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
