# -*- coding: utf-8 -*-
"""
Unit values for nondimensionalization.
"""

import constants as const

t = 1e-9
E = const.kb*const.T
V = E / 1.0
q = const.q
x = q / (const.eps_0*V)
n = 1 / x**3
mu = x**2 / (V*t)
j = q / (t*x**2)
P = E / t

dct = {'Ev': E, 'Ec': E, 'Eg': E, 'Nd': n, 'Na': n, 'C_dop': n,
       'Nc': n, 'Nv': n, 'mu_n': mu, 'mu_p': mu,
       'tau_n': t, 'tau_p': t, 'B': 1 / (n * t),
       'Cn': 1 / (n**2 * t), 'Cp': 1 / (n**2 * t),
       'eps': 1, 'n_refr': 1, 'ni': n, 'wg_mode': 1 / x,
       'n0': n, 'p0': n, 'psi_lcn': V, 'psi_bi': V,
       'psi': V, 'phi_n': V, 'phi_p': V, 'n': n, 'p': n,
       'dn_dpsi': n / V, 'dn_dphin': n / V,
       'dp_dpsi': n / V, 'dp_dphip': n / V,
       'g0': 1 / x, 'N_tr': n, 'S': n * x,
       'fca_e': 1 / (n * x), 'fca_h': 1 / (n * x),
       'jn': j, 'jp': j}
