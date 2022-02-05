"""
Formulas for caclculating carrier concentrations and their derivatives.
"""

import numpy as np
from .prob_functions import fermi_approx as pf
from .prob_functions import fermi_dot_approx as pfd


def n(psi, phi_n, Nc, Ec, Vt):
    """
    Free electron density.
    """
    eta = (psi - phi_n - Ec) / Vt
    return Nc * pf(eta)


def dn_dpsi(psi, phi_n, Nc, Ec, Vt):
    eta = (psi - phi_n - Ec) / Vt
    return Nc * pfd(eta) / Vt


def dn_dphin(psi, phi_n, Nc, Ec, Vt):
    eta = (psi - phi_n - Ec) / Vt
    return -Nc * pfd(eta) / Vt


def p(psi, phi_p, Nv, Ev, Vt):
    """
    Free hole density.
    """
    eta = (-psi + phi_p + Ev) / Vt
    return Nv * pf(eta)


def dp_dpsi(psi, phi_p, Nv, Ev, Vt):
    eta = (-psi + phi_p + Ev) / Vt
    return -Nv * pfd(eta) / Vt


def dp_dphip(psi, phi_p, Nv, Ev, Vt):
    eta = (-psi + phi_p + Ev) / Vt
    return Nv * pfd(eta) / Vt
