"""
Defines a class for solving systems of equations using Newton's method.
"""

import numpy as np
from scipy.linalg import solve_banded
from ldsim import semicond
from ldsim.semicond import equilibrium as eq


def l2_norm(x):
    "Calculate L2 (Euclidean) norm of vector `x`."
    x = np.asarray(x)
    return np.sqrt(np.sum(x*x))


class NewtonSolver(object):
    def __init__(self, res, jac, x0, linalg_solver, inds=None):
        """
        Class for solving a system of equations using Newton's method.

        Parameters
        ----------
        res : callable
            Right-hand side (residual) of the system.
        jac : callable
            Jacobian of the system.
        x0 : array-like
            Initial guess. Creates a copy of the passed array.
        linalg_solver : callable
            Method for solving the 'A*x = b` system.
        inds : iterable or NoneType
            At which indices the solution needs to be updated at every
            iteration. `None` is equivalent to `np.arange(len(x0))`,
            i.e., the whole solution will be updated.

        """
        self.rfun = res
        self.jfun = jac
        self.x = np.array(x0, dtype=np.float64)
        self.la_solver = linalg_solver
        if inds is None:
            self.inds = np.arange(len(x0))
        else:
            self.inds = inds
        self.i = 0  # iteration number
        self.rnorms = list()  # L2 norms of residuals
        self.fluct = list()   # fluctuation values -- ||dx|| / ||x||

    def step(self, omega=1.0):
        """
        Perform single iteration and update x.

        Parameters
        ----------
        omega : float
            Damping parameter, 0<`omega`<=1.

        """
        # check omega value
        assert omega > 0 and omega <= 1.0

        # calculate residual and Jacobian
        self.rvec = self.rfun(self.x)
        self.rnorms.append(l2_norm(self.rvec))
        self.jac = self.jfun(self.x)

        # solve J*dx = -r system and update x
        dx = self.la_solver(self.jac, -self.rvec)
        self.fluct.append(l2_norm(dx)/l2_norm(self.x))
        self.x[self.inds] += dx*omega
        self.i += 1

    def solve(self, maxiter=500, fluct=1e-7, omega=1.0):
        """
        Solve the problem by iterating at most `maxiter` steps (or until
        solution fluctuation is below `fluct`).

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations.
        fluct : float
            Fluctuation of solution needed to stop iterating.
        omega : float
            Damping parameter.

        """
        for _ in range(maxiter):
            self.step(omega)
            if self.fluct[-1] < fluct:
                break


class LCNSolver1d(NewtonSolver):
    def __init__(self, params: dict, i: int = 0):
        """
        Newton solver for 1D electrostatic potential distribution along the
        vertical axis (x) at equilibrium assuming local charge neutrality.
        If using 2D model, specify the vertical slice index `i`.
        """
        Ec = params['Ec']
        if Ec.ndim == 1:
            inds = np.arange(len(Ec))
        else:
            inds = (i, np.arange(Ec.shape[1]))
        self.Ec = params['Ec'][inds]
        self.Ev = params['Ev'][inds]
        self.Nc = params['Nc'][inds]
        self.Nv = params['Nv'][inds]
        self.Vt = params['Vt'][inds]
        self.C_dop = params['C_dop'][inds]

        # initial guess with Boltzmann statistics
        ni = eq.intrinsic_concentration(self.Nc, self.Nv, self.Ec, self.Ev,
                                        self.Vt)
        Ei = eq.intrinsic_level(self.Nc, self.Nv, self.Ec, self.Ev,
                                self.Vt)
        Ef_i = eq.Ef_lcn_boltzmann(self.C_dop, ni, Ei, self.Vt)

        # Jacobian is a vector -> element-wise division
        super().__init__(res=self._residual, jac=self._jacobian, x0=Ef_i,
                         linalg_solver=lambda A, b: b / A)

    def _residual(self, psi: np.ndarray):
        n = semicond.n(psi, 0, self.Nc, self.Ec, self.Vt)
        p = semicond.p(psi, 0, self.Nv, self.Ev, self.Vt)
        return self.C_dop - n + p

    def _jacobian(self, psi: np.ndarray):
        ndot = semicond.dn_dpsi(psi, 0, self.Nc, self.Ec, self.Vt)
        pdot = semicond.dp_dpsi(psi, 0, self.Nv, self.Ev, self.Vt)
        return pdot - ndot


class EquilibriumSolver1d(NewtonSolver):
    def __init__(self, params: dict, xn: np.ndarray, xb: np.ndarray,
                 q: float, eps_0: float, i: int = 0):
        # select slice (2D model) or whole arrays (1D)
        if xn.ndim == 1:
            inds = np.arange(len(xn))
        else:
            inds = (i, np.arange(xn.shape[1]))
        xn = xn[inds]
        self.h = xn[1:] - xn[:-1]
        if isinstance(inds, tuple):
            xb = xb[i]
        self.w = xb[1:] - xb[:-1]
        self.Nc = params['Nc'][inds]
        self.Nv = params['Nv'][inds]
        self.Ec = params['Ec'][inds]
        self.Ev = params['Ev'][inds]
        self.Vt = params['Vt'][inds]
        self.C_dop = params['C_dop'][inds]
        self.eps = params['eps'][inds]
        self.q = q
        self.eps_0 = eps_0

        # save n and p as they are needed for both residual and Jacobian
        self.n = None
        self.p = None

        # initialize solver
        super().__init__(
            res=self._residual,
            jac=self._jacobian,
            x0=params['psi_lcn'][inds],
            linalg_solver=lambda A, b: solve_banded((1, 1), A, b),
            inds=np.arange(1, len(xn) - 1)
        )

    def _residual(self, psi: np.ndarray):
        self.n = semicond.n(psi, 0, self.Nc, self.Ec, self.Vt)
        self.p = semicond.p(psi, 0, self.Nv, self.Ev, self.Vt)
        return eq.poisson_res(psi, self.n, self.p, self.h, self.w,
                              self.eps, self.eps_0, self.q, self.C_dop)

    def _jacobian(self, psi: np.ndarray):
        ndot = semicond.dn_dpsi(psi, 0, self.Nc, self.Ec, self.Vt)
        pdot = semicond.dp_dpsi(psi, 0, self.Nv, self.Ev, self.Vt)
        return eq.poisson_jac(psi, self.n, ndot, self.p, pdot, self.h, self.w,
                              self.eps, self.eps_0, self.q, self.C_dop)
