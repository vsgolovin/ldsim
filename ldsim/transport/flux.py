"""
Current density calculation.
"""

import numpy as np


def bernoulli(x):
    if isinstance(x, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.true_divide(x, (np.exp(x)-1))
            y[np.where(np.abs(x) < 1e-12)] = 1
    elif np.abs(x) < 1e-12:
        y = 1
    else:
        y = x/(np.exp(x)-1)
    return y


def bernoulli_dot(x):
    if isinstance(x, np.ndarray):
        enum = np.exp(x)-x*np.exp(x)-1
        denom = (np.exp(x)-1)**2
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.true_divide(enum, denom)
            y[np.where(np.abs(x) < 1e-12)] = -0.5
    elif np.abs(x) < 1e-12:
        y = -0.5
    else:
        y = (np.exp(x)-x*np.exp(x)-1) / (np.exp(x)-1)**2
    return y


#  %% Scharfetter-Gummel expressions for current density and its derivatives
def SG_jn(n1, n2, B_plus, B_minus, h, Vt, q, mu_n):
    "Scharfetter-Gummel formula for electron current density."
    return -q*mu_n*Vt/h * (n1*B_minus-n2*B_plus)


def SG_djn_dpsi1(n1, n2, ndot_1, B_minus, Bdot_plus, Bdot_minus, h,
                 Vt, q, mu_n):
    return -q*mu_n/h * (Bdot_minus*n1 + B_minus*ndot_1*Vt + Bdot_plus*n2)


def SG_djn_dpsi2(n1, n2, ndot_2, B_plus, Bdot_plus, Bdot_minus, h,
                 Vt, q, mu_n):
    return q*mu_n/h * (Bdot_minus*n1 + Bdot_plus*n2 + B_plus*ndot_2*Vt)


def SG_djn_dphin1(ndot, B_minus, h, Vt, q, mu_n):
    return -q*mu_n*Vt/h * B_minus * ndot


def SG_djn_dphin2(ndot, B_plus, h, Vt, q, mu_n):
    return q*mu_n*Vt/h * B_plus * ndot


def SG_jp(p1, p2, B_plus, B_minus, h, Vt, q, mu_p):
    "Scharfetter-Gummel formula for hole current density."
    return q*mu_p*Vt/h * (p1*B_plus - p2*B_minus)


def SG_djp_dpsi1(p1, p2, pdot_1, B_plus, Bdot_plus, Bdot_minus, h,
                 Vt, q, mu_p):
    jdot = q*mu_p/h * (-Bdot_plus*p1+B_plus*pdot_1*Vt-Bdot_minus*p2)
    return jdot


def SG_djp_dpsi2(p1, p2, pdot_2, B_minus, Bdot_plus, Bdot_minus, h,
                 Vt, q, mu_p):
    jdot = q*mu_p/h * (Bdot_plus*p1+Bdot_minus*p2-B_minus*pdot_2*Vt)
    return jdot


def SG_djp_dphip1(pdot, B_plus, h, Vt, q, mu_p):
    jdot = q*mu_p*Vt/h * B_plus * pdot
    return jdot


def SG_djp_dphip2(pdot, B_minus, h, Vt, q, mu_p):
    jdot = -q*mu_p*Vt/h * B_minus * pdot
    return jdot


# %% original Scharfetter-Gummel scheme with Boltzmann statistics
def oSG_jn(exp_eta_1, exp_eta_2, B_plus, B_minus, h, Nc, Vt, q, mu_n):
    return -q*mu_n*Vt/h * Nc * (B_minus * exp_eta_1 - B_plus * exp_eta_2)


def oSG_djn_dpsi1(exp_eta_1, exp_eta_2, B_minus, Bdot_plus, Bdot_minus,
                  h, Nc, q, mu_n):
    return -q*mu_n/h * Nc * ((Bdot_minus+B_minus)*exp_eta_1
                             + Bdot_plus*exp_eta_2)


def oSG_djn_dpsi2(exp_eta_1, exp_eta_2, B_plus, Bdot_plus, Bdot_minus,
                  h, Nc, q, mu_n):
    return q*mu_n/h * Nc * (Bdot_minus * exp_eta_1
                            + (Bdot_plus + B_plus) * exp_eta_2)


def oSG_djn_dphin1(exp_eta_1, B_minus, h, Nc, q, mu_n):
    return q * mu_n / h * Nc * B_minus * exp_eta_1


def oSG_djn_dphin2(exp_nu_2, B_plus, h, Nc, q, mu_n):
    return -q * mu_n / h * Nc * B_plus * exp_nu_2


def oSG_jp(exp_eta_1, exp_eta_2, B_plus, B_minus, h, Nv, Vt, q, mu_p):
    return q*mu_p*Vt/h * Nv * (B_plus*exp_eta_1 - B_minus*exp_eta_2)


def oSG_djp_dpsi1(exp_eta_1, exp_eta_2, B_plus, Bdot_plus, Bdot_minus,
                  h, Nv, q, mu_p):
    return -q*mu_p/h * Nv * ((Bdot_plus+B_plus)*exp_eta_1
                             + Bdot_minus*exp_eta_2)


def oSG_djp_dpsi2(exp_eta_1, exp_eta_2, B_minus, Bdot_plus, Bdot_minus,
                  h, Nv, q, mu_p):
    return q*mu_p/h * Nv * (Bdot_plus*exp_eta_1
                            + (Bdot_minus+B_minus)*exp_eta_2)


def oSG_djp_dphip1(exp_eta_1, B_plus, h, Nv, q, mu_p):
    return q*mu_p/h * Nv * B_plus*exp_eta_1


def oSG_djp_dphip2(exp_eta_2, B_minus, h, Nv, q, mu_p):
    return -q*mu_p/h * Nv * B_minus*exp_eta_2


# %% Modified Scharfetter-Gummel scheme
def g(eta_1, eta_2, F):
    "Diffusion enhancement factor."
    return np.sqrt((F(eta_1) * F(eta_2)) / (np.exp(eta_1) * np.exp(eta_2)))


def gdot(g, eta, F, F_dot):
    "Diffusion enhancement factor `g` derivative with respect to `nu`."
    return g / 2 * (F_dot(eta) / F(eta) - 1)


def mSG_jdot(j_SG, jdot_SG, g, gdot):
    return jdot_SG * g + j_SG * gdot
