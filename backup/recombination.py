# -*- coding: utf-8 -*-
"""
A collection of functions for calculating recombination rates and corresponding
derivatives.
"""

import numpy as np
import carrier_concentrations as cc

def srh_R(n, p, n0, p0, tau_n, tau_p):
    """
    Shockley-Read-Hall recombination rate.
    """
    R = (n*p - n0*p0) / ((n+n0)*tau_p + (p+p0)*tau_n)
    return R

def srh_Rdot(n, ndot, p, pdot, n0, p0, tau_n, tau_p):
    """
    Shockley-Read-Hall recombination rate derivative with respect to
    electrostatic potential or one of quasi-Fermi potentials.
    """
    u = n*p - n0*p0
    v = (n+n0)*tau_p + (p+p0)*tau_n
    udot = ndot*p + n*pdot
    vdot = tau_p*ndot + tau_n*pdot
    Rdot = (udot*v - u*vdot) / v**2
    return Rdot

def rad_R(n, p, n0, p0, B):
    """
    Radiative recombination rate.
    """
    R = B*(n*p-n0*p0)
    return R

def rad_Rdot(n, ndot, p, pdot, B):
    """
    Radiative recombination rate derivative with respect to electrostatic
    potential or one of quasi-Fermi potentials.
    """
    Rdot = B*(n*pdot+p*ndot)
    return Rdot

def auger_R(n, p, n0, p0, Cn, Cp):
    """
    Auger recombination rate.
    """
    R = (Cn*n+Cp*p) * (n*p-n0*p0)
    return R

def auger_Rdot(n, ndot, p, pdot, n0, p0, Cn, Cp):
    """
    Auger recombination rate derivative with respect to electrostatic
    potential or one of quasi-Fermi potentials.
    """
    delta2 = 2*n*p - n0*p0
    Rdot = Cn*(delta2*ndot + n**2*pdot) + Cp*(delta2*pdot + p**2*ndot)
    return Rdot

# testing
test_err = 1e-4

def test_derivatives():
    # parameters
    psi = -2
    phi_n = 1.2
    phi_p = 0.8
    Nc = 2.4e10
    Nv = 45e10
    Ec = 0.7
    Ev = -1
    Vt = 1.4
    n0 = 2e7
    p0 = 1e7
    tau_n = 2e-9
    tau_p = 3e-9
    B = 1e10
    Cn = 2e-30
    Cp = 5e-30

    # calculating recombination rates' derivatives
    # 1. by using implemented functions
    n = cc.n(psi, phi_n, Nc, Ec, Vt)
    ndot = cc.dn_dpsi(psi, phi_n, Nc, Ec, Vt)
    p = cc.p(psi, phi_p, Nv, Ev, Vt)
    pdot = cc.dp_dpsi(psi, phi_p, Nv, Ev, Vt)
    Rdot_srh = srh_Rdot(n, ndot, p, pdot, n0, p0, tau_n, tau_p)
    Rdot_rad = rad_Rdot(n, ndot, p, pdot, B)
    Rdot_aug = auger_Rdot(n, ndot, p, pdot, n0, p0, Cn, Cp)

    # 2. by using finite differences
    psi_1 = psi*(1-1e-4)
    psi_2 = psi*(1+1e-4)
    n1 = cc.n(psi_1, phi_n, Nc, Ec, Vt)
    p1 = cc.p(psi_1, phi_p, Nv, Ev, Vt)
    n2 = cc.n(psi_2, phi_n, Nc, Ec, Vt)
    p2 = cc.p(psi_2, phi_p, Nv, Ev, Vt)

    errs = np.zeros(3, dtype=float)

    R1 = srh_R(n1, p1, n0, p0, tau_n, tau_p)
    R2 = srh_R(n2, p2, n0, p0, tau_n, tau_p)
    Rdot = (R2-R1)/(psi_2-psi_1)
    errs[0] = np.abs(1-Rdot_srh/Rdot)

    R1 = rad_R(n1, p1, n0, p0, B)
    R2 = rad_R(n2, p2, n0, p0, B)
    Rdot = (R2-R1)/(psi_2-psi_1)
    errs[1] = np.abs(1-Rdot_rad/Rdot)

    R1 = auger_R(n1, p1, n0, p0, Cn, Cp)
    R2 = auger_R(n2, p2, n0, p0, Cn, Cp)
    Rdot = (R2-R1)/(psi_2-psi_1)
    errs[2] = np.abs(1-Rdot_aug/Rdot)

    # checking results
    print(errs)
    assert (errs<test_err).all()
