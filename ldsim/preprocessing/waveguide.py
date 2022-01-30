"""
1D waveguide equation.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs


def solve_wg(x, n, lam, n_modes):
    """
    Solve eigenvalue problem for a 1D waveguide.

    Parameters
    ----------
    x : numpy.ndarray
        x coordinate.
    n : numpy.ndarray
        Refractive index values. `n.shape==x.shape`
    lam : number
        Wavelength, same units as `x`.
    n_modes : int
        Number of eigenvalues/eigenvectors to be calculated.

    Returns
    -------
    n_eff : numpy.ndarray
        Calculated effective refractive index values.
    modes : numpy.ndarray
        Calculated mode profiles.

    """
    # creating matrix A for the eigenvalue problem
    k0 = 2*np.pi / lam
    delta_x = x[1]-x[0]  # uniform mesh
    delta_chi2 = (delta_x*k0)**2
    main_diag = n**2 - 2/delta_chi2
    off_diag = np.full(x[:-1].shape, 1/delta_chi2)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1])

    # solving the eigenproblem
    n_max = n.max()
    w, v = eigs(A, k=n_modes, which='SR', sigma=n_max**2)

    # converting eigenvalues to effective refractive indices
    n_eff = np.sqrt(np.real(w))
    # and eigenvectors to mode profiles
    modes = np.real(v)**2

    # normalizing modes
    for i in range(n_modes):
        integral = np.sum(modes[:, i])*delta_x
        modes[:, i] /= integral

    return n_eff, modes
