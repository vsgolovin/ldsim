# -*- coding: utf-8 -*-
"""
Collection of tools for solving van Roosbroeck (drift-diffusion) system.
"""

import numpy as np


# 1. Poisson's equation
def poisson_res(psi, n, p, h, w, eps, eps_0, q, C_dop):
    lhs = -eps[1:-1] * (  1/h[1:  ]          *psi[2:  ]
                        -(1/h[1:  ]+1/h[:-1])*psi[1:-1]
                        + 1/h[ :-1]          *psi[ :-2] )
    rhs = q/eps_0 * (C_dop[1:-1]-n[1:-1]+p[1:-1]) * w
    r = -lhs+rhs
    return r


def poisson_dF_dpsi(ndot, pdot, h, w, eps, eps_0, q):
    m = len(ndot)-2  # number of inner nodes
    J = np.zeros((3, m))  # Jacobian in tridiagonal form
    J[0, 1:] = eps[1:-2] / h[1:-1]
    J[1, :] = -eps[1:-1] * (1/h[1:] + 1/h[:-1]) \
              + q/eps_0 * (pdot[1:-1] - ndot[1:-1]) * w
    J[2, :-1] = eps[2:-1] / h[1:-1]
    return J


def poisson_dF_dphin(ndot, w, eps_0, q):
    J = -q / eps_0 * ndot[1:-1] * w
    return J


def poisson_dF_dphip(pdot, w, eps_0, q):
    J = q / eps_0 * pdot[1:-1] * w
    return J


# 2. Electron current density continuity equation
def jn_dF_dpsi(djn_dpsi1, djn_dpsi2, dR_dpsi, w, q, m):
    J = np.zeros((3, m))
    J[0, 1:] = -djn_dpsi2[1:-1]
    J[1, :] = q*dR_dpsi*w - (djn_dpsi1[1:] - djn_dpsi2[:-1])
    J[2, :-1] = djn_dpsi1[1:-1]
    return J


def jn_dF_dphin(djn_dphin1, djn_dphin2, dR_dphin, w, q, m):
    J = np.zeros((3, m))
    J[0, 1:] = -djn_dphin2[1:-1]
    J[1, :] = dR_dphin*w - (djn_dphin1[1:] - djn_dphin2[:-1])
    J[2, :-1] = djn_dphin1[1:-1]
    return J


def jn_dF_dphip(dR_dphip, w, q, m):
    J = np.zeros(m)
    J[:] = q*dR_dphip*w
    return J


# 3. Hole current density continuity equation
def jp_dF_dpsi(djp_dpsi1, djp_dpsi2, dR_dpsi, w, q, m):
    J = np.zeros((3, m))
    J[0, 1:] = -djp_dpsi2[1:-1]
    J[1, :] = -q*dR_dpsi*w - (djp_dpsi1[1:]-djp_dpsi2[:-1])
    J[2, :-1] = djp_dpsi1[1:-1]
    return J


def jp_dF_dphin(dR_dphin, w, q, m):
    J = np.zeros(m)
    J[:] = -q*dR_dphin*w
    return J


def jp_dF_dphip(djp_dphip1, djp_dphip2, dR_dphip, w, q, m):
    J = np.zeros((3, m))
    J[0, 1:] = -djp_dphip2[1:-1]
    J[1, :] = -q*dR_dphip*w - (djp_dphip1[1:]-djp_dphip2[:-1])
    J[2, :-1] = djp_dphip1[1:-1]
    return J
