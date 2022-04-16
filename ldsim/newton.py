"""
Defines a class for solving systems of equations using Newton's method.
"""

import numpy as np
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
