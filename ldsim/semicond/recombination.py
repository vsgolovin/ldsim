"""
A collection of functions for calculating recombination rates and corresponding
derivatives.
"""


def srh_R(n, p, n0, p0, tau_n, tau_p):
    """
    Shockley-Read-Hall recombination rate.
    """
    R = (n*p - n0*p0) / ((n+n0) * tau_p + (p+p0) * tau_n)
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
