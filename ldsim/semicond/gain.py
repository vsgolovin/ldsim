import numpy as np


def gain_2p(n, p, g0, N_tr, return_derivatives=True):
    N = p.copy()
    mask_n = (n < p)
    N[mask_n] = n[mask_n]
    gain = g0 * np.log(N / N_tr)
    if not return_derivatives:
        return gain

    # calculate derivatives w.r.t. electron and photon densities
    dg_dn = np.zeros_like(gain)
    dg_dn[mask_n] = g0[mask_n] / N[mask_n]
    dg_dp = np.zeros_like(gain)
    dg_dp[~mask_n] = g0[~mask_n] / N[~mask_n]

    return gain, dg_dn, dg_dp
