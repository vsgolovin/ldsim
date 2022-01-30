# -*- coding: utf-8 -*-

"""
A collection of statistical distribution functions for carrier density
calculation.
"""

import numpy as np


def fermi_approx(eta):
    """
    Fermi-Dirac integral of order 1/2, uses approximate formula from
    "Physics of Photonic Devices" by S.L.Chuang.
    """
    denom = eta + 2.13 + np.power(
        np.power(np.abs(eta - 2.13), 12./5) + 9.6,
        5./12)
    C = 3 * np.sqrt(np.pi / 2) / np.power(denom, 1.5)
    F = 1 / (np.exp(-eta) + C)
    return F


def fermi_dot_approx(eta):
    """
    Derivative of Fermi-Dirac integral approximation.
    """
    C = 3 * np.sqrt(np.pi / 2) / np.power(
        eta + 2.13 + np.power(
            np.power(np.abs(eta - 2.13), 12./5) + 9.6,
            5./12),
        1.5)
    eta_abs_dot = 12./5 * np.power(np.abs(eta - 2.13), 7./5)
    if isinstance(eta, np.ndarray):
        ix = eta < 2.13
        eta_abs_dot[ix] *= -1
    elif eta < 2.13:
        eta_abs_dot *= -1
    dC = -9./2 * np.sqrt(np.pi / 2) * np.power(
        eta + 2.13 + np.power(
            np.power(np.abs(eta - 2.13), 12./5) + 9.6, 5./12), -5./2) * (
        1 + 5./12 * np.power(np.power(np.abs(eta - 2.13), 12./5) + 9.6,
                             -7./12) * eta_abs_dot)
    return 1 / (np.exp(-eta) + C)**2 * (np.exp(-eta) - dC)


def blakemore(eta):
    """
    Use Blakemore approximation of Fermi-Dirac integral of order 1/2.
    Valid for `eta`<=1.3.
    """
    return 1 / (np.exp(-eta) + 0.27)


def boltzmann(eta):
    """
    Use Boltzmann approximation of Fermi-Dirac integral of order 1/2.
    Valid for `eta`<=-2.
    """
    return np.exp(eta)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(-5, 5, 200)
    y = fermi_approx(x)
    y_dot = fermi_dot_approx(x)
    y_dot_fd = (y[1:] - y[:-1]) / (x[1:] - x[:-1])

    plt.figure('function')
    plt.plot(x, y)

    plt.figure('derivative')
    plt.plot(x[:-1], y_dot_fd)
    plt.plot(x, y_dot)
    plt.show()
