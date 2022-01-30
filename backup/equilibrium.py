# -*- coding: utf-8 -*-
"""
A collection of functions for solving Poisson's equation at equilibrium.
"""

import numpy as np


def intrinsic_concentration(Nc, Nv, Ec, Ev, Vt):
    "Calculate intrinsic carrier concentration."
    ni = np.sqrt(Nc*Nv)*np.exp( (Ev-Ec) / (2*Vt) )
    return ni


def intrinsic_level(Nc, Nv, Ec, Ev, Vt):
    "Calculate Fermi level location in an intrinsic semiconductor."
    Ei = (Ec+Ev)/2 + Vt/2*np.log(Nv/Nc)
    return Ei


def Ef_lcn_boltzmann(C_dop, ni, Ei, Vt):
    """
    Calculate Fermi level location assuming local charge neutrality and
    Boltzmann statistics.
    """
    xi = np.arcsinh(C_dop/(2*ni))
    Ef = xi*Vt + Ei
    return Ef


def poisson_res(psi, n, p, h, w, eps, eps_0, q, C_dop):
    """
    Calculate residual of Poisson's equation at equilibrium.
    """
    lhs = -eps[1:-1] * (  1/h[1:  ]          *psi[2:  ]
                        -(1/h[1:  ]+1/h[:-1])*psi[1:-1]
                        + 1/h[ :-1]          *psi[ :-2] )
    rhs = q/eps_0 * (C_dop[1:-1] - n[1:-1] + p[1:-1]) * w
    return -lhs+rhs


def poisson_jac(psi, n, ndot, p, pdot, h, w, eps, eps_0, q, C_dop):
    """
    Calculate Jacobian of Poisson's equation at equilibrium.
    """
    m = len(psi)
    J = np.zeros((3, m-2))
    J[0, 1:  ] =  eps[1:-2]/h[1:-1]
    J[1,  :  ] = -eps[1:-1]*(1/h[1:]+1/h[:-1]) \
                 +q/eps_0*(-ndot[1:-1]+pdot[1:-1])*w
    J[2,  :-1] =  eps[2:-1]/h[1:-1]
    return J
