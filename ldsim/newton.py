"""
Defines a class for solving systems of equations using Newton's method.
"""

from typing import Union, Tuple
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


def transport_system(xn: np.ndarray, vn: dict, xb: np.ndarray, vb: dict,
                     sol: dict, q: float, eps_0: float,
                     index: Union[int, None] = None
                     ) -> Tuple[np.ndarray, list, np.ndarray]:
    # shorthand function to access either array itself (1D model)
    # or 1D slice of 2D array (2D model)
    def get(array):
        return array[index] if index is not None else array

    h = get(xn)[1:] - get(xn)[:-1]  # mesh steps
    w = get(xb)[1:] - get(xb)[:-1]  # 1D volumes
    m = len(w)                      # number of inner nodes

    # recombination rate
    R = (get(vn['R_srh']) + get(vn['R_rad']) + get(vn['R_aug']))[1:-1]
    dR_dpsi = sum(get(vn[f'd{r}_dpsi'])
                  for r in ['Rsrh', 'Rrad', 'Raug'])[1:-1]
    dR_dphin = sum(get(vn[f'd{r}_dphin'])
                   for r in ['Rsrh', 'Rrad', 'Raug'])[1:-1]
    dR_dphip = sum(get(vn[f'd{r}_dphip'])
                   for r in ['Rsrh', 'Rrad', 'Raug'])[1:-1]

    # van Roosbroeck system residual vector
    residuals = np.zeros(m * 3)
    residuals[:m] = poisson_residual(
        psi=get(sol['psi']), n=get(sol['n']), p=get(sol['p']), h=h, w=w,
        eps=get(vn['eps']), eps_0=eps_0, q=q, C_dop=get(vn['C_dop'])
    )
    residuals[m:2*m] = jn_residual(jn=get(vb['jn']), R=R, w=w, q=q)
    residuals[2*m:] = jp_residual(jp=get(vb['jp']), R=R, w=w, q=q)

    # van Roosbroech system Jacobian
    # 1. Poisson's equation
    j11 = poisson_dF_dpsi(ndot=get(sol['dn_dpsi']), pdot=get(sol['dp_dpsi']),
                          h=h, w=w, eps=get(vn['eps']), eps_0=eps_0, q=q)
    j12 = poisson_dF_dphin(ndot=get(sol['dn_dphin']), w=w, eps_0=eps_0, q=q)
    j13 = poisson_dF_dphip(pdot=get(sol['dp_dphip']), w=w, eps_0=eps_0, q=q)

    # 2. Electron current density continuity equation
    j21 = jn_dF_dpsi(get(vb['djn_dpsi1']), get(vb['djn_dpsi2']), dR_dpsi, w, q)
    j22 = jn_dF_dphin(get(vb['djn_dphin1']), get(vb['djn_dphin2']),
                      dR_dphin=dR_dphin, w=w, q=q)
    j23 = jn_dF_dphip(dR_dphip=dR_dphip, w=w, q=q)

    # 3. Hole current density continuity equation
    j31 = jp_dF_dpsi(get(vb['djp_dpsi1']), get(vb['djp_dpsi2']), dR_dpsi, w, q)
    j32 = jp_dF_dphin(dR_dphin=dR_dphin, w=w, q=q)
    j33 = jp_dF_dphip(get(vb['djp_dphip1']), get(vb['djp_dphip2']),
                      dR_dphip=dR_dphip, w=w, q=q)

    # collect Jacobian diagonals
    data = np.zeros((11, 3*m))
    data[0, 2*m:   ] = j13
    data[1,   m:2*m] = j12
    data[1, 2*m:   ] = j23
    data[2,    :m  ] = j11[0]
    data[2,   m:2*m] = j22[0]
    data[2, 2*m:   ] = j33[0]
    data[3,    :m  ] = j11[1]
    data[3,   m:2*m] = j22[1]
    data[3, 2*m:   ] = j33[1]
    data[4,    :m  ] = j11[2]
    data[4,   m:2*m] = j22[2]
    data[4, 2*m:   ] = j33[2]
    data[5,    :m  ] = j21[0]
    data[6,    :m  ] = j21[1]
    data[6,   m:2*m] = j32
    data[7,    :m  ] = j21[2]
    data[8,    :m  ] = j31[0]
    data[9,    :m  ] = j31[1]
    data[10,   :m  ] = j31[2]

    # indices of diagonals (values stored in `data`)
    diags = [2*m, m, 1, 0, -1, -m+1, -m, -m-1, -2*m+1, -2*m, -2*m-1]

    return data, diags, residuals


# Poisson's equation
def poisson_residual(psi, n, p, h, w, eps, eps_0, q, C_dop):
    lhs = -eps[1:-1] * (
        1 / h[1:] * psi[2:]
        - (1 / h[1:] + 1 / h[:-1]) * psi[1:-1]
        + 1 / h[:-1] * psi[:-2]
    )
    rhs = q / eps_0 * (C_dop[1:-1] - n[1:-1] + p[1:-1]) * w
    return rhs - lhs


def poisson_dF_dpsi(ndot, pdot, h, w, eps, eps_0, q):
    m = len(ndot) - 2     # number of inner nodes
    J = np.zeros((3, m))  # Jacobian in tridiagonal form
    J[0, 1:] = eps[1:-2] / h[1:-1]
    J[1, :] = (-eps[1:-1] * (1/h[1:] + 1/h[:-1])
               + q/eps_0 * (pdot[1:-1] - ndot[1:-1]) * w)
    J[2, :-1] = eps[2:-1] / h[1:-1]
    return J


def poisson_dF_dphin(ndot, w, eps_0, q):
    return -q / eps_0 * ndot[1:-1] * w


def poisson_dF_dphip(pdot, w, eps_0, q):
    return q / eps_0 * pdot[1:-1] * w


# Electron current density continuity equation
def jn_residual(jn, R, w, q):
    return q * R * w - (jn[1:] - jn[:-1])


def jn_dF_dpsi(djn_dpsi1, djn_dpsi2, dR_dpsi, w, q):
    J = np.zeros((3, len(w)))
    J[0, 1:] = -djn_dpsi2[1:-1]
    J[1, :] = q * dR_dpsi * w - (djn_dpsi1[1:] - djn_dpsi2[:-1])
    J[2, :-1] = djn_dpsi1[1:-1]
    return J


def jn_dF_dphin(djn_dphin1, djn_dphin2, dR_dphin, w, q):
    J = np.zeros((3, len(w)))
    J[0, 1:] = -djn_dphin2[1:-1]
    J[1, :] = q * dR_dphin * w - (djn_dphin1[1:] - djn_dphin2[:-1])
    J[2, :-1] = djn_dphin1[1:-1]
    return J


def jn_dF_dphip(dR_dphip, w, q):
    J = np.zeros(len(w))
    J[:] = q * dR_dphip * w
    return J


# Hole current density continuity equation
def jp_residual(jp, R, w, q):
    return -q * R * w - (jp[1:] - jp[:-1])


def jp_dF_dpsi(djp_dpsi1, djp_dpsi2, dR_dpsi, w, q):
    J = np.zeros((3, len(w)))
    J[0, 1:] = -djp_dpsi2[1:-1]
    J[1, :] = -q * dR_dpsi * w - (djp_dpsi1[1:] - djp_dpsi2[:-1])
    J[2, :-1] = djp_dpsi1[1:-1]
    return J


def jp_dF_dphin(dR_dphin, w, q):
    J = np.zeros(len(w))
    J[:] = -q * dR_dphin * w
    return J


def jp_dF_dphip(djp_dphip1, djp_dphip2, dR_dphip, w, q):
    J = np.zeros((3, len(w)))
    J[0, 1:] = -djp_dphip2[1:-1]
    J[1, :] = -q * dR_dphip * w - (djp_dphip1[1:] - djp_dphip2[:-1])
    J[2, :-1] = djp_dphip1[1:-1]
    return J
