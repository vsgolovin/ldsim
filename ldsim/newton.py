"""
Defines a class for solving systems of equations using Newton's method.
"""

import numpy as np


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
            At which indices of solution need to be updated at every
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
