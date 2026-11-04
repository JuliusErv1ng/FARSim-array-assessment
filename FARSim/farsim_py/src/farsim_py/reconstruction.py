from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def get_zhs_effective_refractive_index(
    x0: float,
    y0: float,
    z0: float,
    xant: float = 0.0,
    yant: float = 0.0,
    zant: float = 0.0,
    ns: float = 325.0,
    kr: float = -0.1218,
    step_size: float = 20000.0,
) -> float:
    rearth = 6_370_949.0
    r0_sq = x0 * x0 + y0 * y0
    h0 = (np.sqrt((z0 + rearth) ** 2 + r0_sq) - rearth) / 1e3
    modr = np.sqrt(r0_sq)

    if modr > 1000.0:
        ux = xant - x0
        uy = yant - y0
        uz = zant - z0
        nint = int(np.ceil(modr / step_size))
        kx = ux / nint
        ky = uy / nint
        kz = uz / nint

        currpx, currpy, currpz, currh = x0, y0, z0, h0
        total = 0.0
        for _ in range(nint):
            nextpx = currpx + kx
            nextpy = currpy + ky
            nextpz = currpz + kz
            next_r_sq = nextpx * nextpx + nextpy * nextpy
            nexth = (np.sqrt((nextpz + rearth) ** 2 + next_r_sq) - rearth) / 1e3
            if abs(nexth - currh) > 1e-10:
                total += (np.exp(kr * nexth) - np.exp(kr * currh)) / (kr * (nexth - currh))
            else:
                total += np.exp(kr * currh)
            currpx, currpy, currpz, currh = nextpx, nextpy, nextpz, nexth
        avn = ns * total / nint
        return 1.0 + 1e-6 * avn

    hd = zant / 1e3
    if abs(hd - h0) > 1e-10:
        avn = (ns / (kr * (hd - h0))) * (np.exp(kr * hd) - np.exp(kr * h0))
    else:
        avn = ns * np.exp(kr * h0)
    return 1.0 + 1e-6 * avn


def plane_wave_reconstruction_error(points_with_time: np.ndarray, n_eff: float, theta_true: float, phi_true: float) -> float:
    antenna_coords = points_with_time[:, :3]
    arrival_times = points_with_time[:, 3]
    c_eff = 299_792_458.0 / n_eff

    def objective(angles: np.ndarray) -> float:
        theta, phi = angles
        k = np.array(
            [
                np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
                np.cos(np.deg2rad(theta)),
            ]
        )
        diff_t = arrival_times[:, None] - arrival_times[None, :]
        diff_x = antenna_coords[:, None, :] - antenna_coords[None, :, :]
        terms = c_eff * diff_t - np.einsum("k,ijk->ij", k, diff_x)
        return float(np.sum(terms**2))

    result = minimize(
        objective,
        x0=np.array([45.0, 45.0], dtype=float),
        method="L-BFGS-B",
        bounds=((0.0, 180.0), (0.0, 360.0)),
    )
    theta_opt, phi_opt = result.x
    k_opt = -np.array(
        [
            np.sin(np.deg2rad(theta_opt)) * np.cos(np.deg2rad(phi_opt)),
            np.sin(np.deg2rad(theta_opt)) * np.sin(np.deg2rad(phi_opt)),
            np.cos(np.deg2rad(theta_opt)),
        ]
    )
    if k_opt[2] <= 0:
        k_opt[2] *= -1
    true_vec = np.array(
        [
            np.sin(np.deg2rad(theta_true)) * np.cos(np.deg2rad(phi_true)),
            np.sin(np.deg2rad(theta_true)) * np.sin(np.deg2rad(phi_true)),
            np.cos(np.deg2rad(theta_true)),
        ]
    )
    cos_angle = np.clip(np.dot(k_opt, true_vec) / (np.linalg.norm(k_opt) * np.linalg.norm(true_vec)), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))
