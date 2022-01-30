# -*- coding: utf-8 -*-
"""
Functions for calculating free carrier densities and their derivatives.
"""

import numpy as np
from sdf import fermi_approx as fermi
from sdf import fermi_dot_approx as fermi_dot

def n(psi, phi_n, Nc, Ec, Vt):
    eta = (psi - phi_n - Ec)/Vt
    F = fermi(eta)
    return Nc * F

def dn_dpsi(psi, phi_n, Nc, Ec, Vt):
    eta = (psi-phi_n-Ec)/Vt
    Fdot = fermi_dot(eta)
    return Nc * Fdot / Vt

def dn_dphin(psi, phi_n, Nc, Ec, Vt):
    eta = (psi-phi_n-Ec)/Vt
    Fdot = fermi_dot(eta)
    return -Nc * Fdot / Vt

def p(psi, phi_p, Nv, Ev, Vt):
    eta = (-psi + phi_p + Ev) / Vt
    F = fermi(eta)
    return Nv * F

def dp_dpsi(psi, phi_p, Nv, Ev, Vt):
    eta = (-psi + phi_p + Ev) / Vt
    Fdot = fermi_dot(eta)
    return -Nv * Fdot/Vt

def dp_dphip(psi, phi_p, Nv, Ev, Vt):
    eta = (-psi + phi_p + Ev) / Vt
    Fdot = fermi_dot(eta)
    return Nv * Fdot / Vt

# testing
test_err = 1e-2

def test_n():
    Nc = 20
    psi = np.array([-6, -4,  0.,  6])
    phi = np.array([ 2, -1,  0.,  2])
    Ec  = np.array([ 2,  0,  0., -4])
    Vt  = np.array([ 2,  1,  1.,  2])
    n_calc = n(psi, phi, Nc, Ec, Vt)
    n_real = Nc*2/np.sqrt(np.pi)*np.array([0.0060, 0.0434, 0.6781, 5.7706])
    err = np.abs(1-n_calc/n_real)
    assert (err<test_err).all()

def test_dn_dpsi():
    Nc = 5
    Ec = 1
    phi = 0
    Vt = 1
    psi_1 = -0.01
    psi_2 =  0.01
    n1 = n(psi_1, phi, Nc, Ec, Vt)
    n2 = n(psi_2, phi, Nc, Ec, Vt)
    ndot = dn_dpsi(0, phi, Nc, Ec, Vt)
    ndot_fd = (n2-n1)/(psi_2-psi_1)
    err = np.abs(1-ndot/ndot_fd)
    assert err<test_err

def test_dn_dphin():
    Nc = 2e4
    Ec = 30
    Vt = 2.5
    psi = 10
    phi_1 = -5.01
    phi_2 = -4.99
    n1 = n(psi, phi_1, Nc, Ec, Vt)
    n2 = n(psi, phi_2, Nc, Ec, Vt)
    ndot_fd = (n2-n1)/(phi_2-phi_1)
    ndot = dn_dphin(psi, -5, Nc, Ec, Vt)
    err = np.abs(1-ndot/ndot_fd)
    assert err<test_err

def test_p():
    Nv = 1
    Ev  = np.array([-3., 0, 0, 8])
    psi = np.array([3., 1, 0, -8])
    phi = np.array([1., -3, 0, -1.6])
    Vt = np.array([1., 2, 3, 4])
    p_calc = p(psi, phi, Nv, Ev, Vt)
    p_real = Nv*2/np.sqrt(np.pi)*np.array([0.0060, 0.1146, 0.6781, 5.0181])
    err = np.abs(1-p_calc/p_real)
    assert (err<test_err).all()

def test_dp_dpsi():
    Nv = 4.2e3
    Ev = -3
    Vt = 1.3
    phi = 1
    psi_1 = 2.01
    psi_2 = 1.99
    p1 = p(psi_1, phi, Nv, Ev, Vt)
    p2 = p(psi_2, phi, Nv, Ev, Vt)
    pdot_fd = (p2-p1)/(psi_2-psi_1)
    pdot = dp_dpsi(2, phi, Nv, Ev, Vt)
    err = np.abs(1-pdot/pdot_fd)
    assert err<test_err

def test_dp_dphip():
    Nv = 10
    Ev = 0
    Vt = 1
    psi = 0
    phi_1 = 2.99
    phi_2 = 3.01
    p1 = p(psi, phi_1, Nv, Ev, Vt)
    p2 = p(psi, phi_2, Nv, Ev, Vt)
    pdot_fd = (p2-p1)/(phi_2-phi_1)
    pdot = dp_dphip(psi, 3, Nv, Ev, Vt)
    err = np.abs(1-pdot/pdot_fd)
    assert err<test_err
