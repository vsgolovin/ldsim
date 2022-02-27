import warnings
import numpy as np
from scipy.linalg import solve_banded
from ldsim.preprocessing.design import LaserDiode
from ldsim.mesh import generate_nonuniform_mesh
from ldsim import units, newton, semicond
import ldsim.semicond.equilibrium as eq


class LaserDiodeModel1d(LaserDiode):
    input_params_nodes = [
        'Ev', 'Ec', 'Eg', 'Nd', 'Na', 'C_dop', 'Nc', 'Nv', 'mu_n', 'mu_p',
        'tau_n', 'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr', 'g0', 'N_tr',
        'fca_e', 'fca_h', 'T']
    input_params_boundaries = ['Ev', 'Ec', 'Nc', 'Nv', 'mu_n', 'mu_p']
    params_active = ['g0', 'N_tr']
    calculated_params_nodes = [
        'Vt', 'psi_lcn', 'n0', 'p0', 'psi_bi']
    solution_arrays = ['psi', 'phi_n', 'phi_p', 'n', 'p']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # mesh
        self.xn = None     # nodes
        self.xb = None     # volume boundaries
        self.ar_ix = None  # active region mask for xn

        # parameters at mesh nodes and volume boundaries
        self.vn = dict.fromkeys(self.input_params_nodes
                                + self.calculated_params_nodes)
        self.vb = dict.fromkeys(self.input_params_boundaries)
        self.sol = dict.fromkeys(self.solution_arrays)
        self.S = 1e-12

    def _update_Vt(self):
        self.vn['Vt'] = self.vn['T'] * self.kb

    def _update_densities(self):
        s = self.sol
        v = self.vn
        self.sol['n'] = semicond.n(s['psi'], s['phi_n'], v['Nc'], v['Ec'],
                                   v['Vt'])
        self.sol['p'] = semicond.p(s['psi'], s['phi_p'], v['Nv'], v['Ev'],
                                   v['Vt'])

    def generate_nonuniform_mesh(self, step_uni=5e-8, step_min=1e-7,
                                 step_max=20e-7, sigma=100e-7,
                                 y_ext=[None, None]):
        """
        Generate nonuniform mesh in which step size is proportional to local
        change in bandgap (Eg). Initially creates uniform mesh with step size
        `step_uni`. See `ldsim.mesh.generate_nonuiform_mesh` for detailed
        description.
        """
        assert not self.is_dimensionless
        thickness = self.get_thickness()
        num_points = int(round(thickness // step_uni))
        x = np.linspace(0, thickness, num_points)
        y = self.calculate('Eg', x)
        self.xn = generate_nonuniform_mesh(x, y, step_min, step_max,
                                           sigma, y_ext)[0]
        self.xb = (self.xn[1:] + self.xn[:-1]) / 2
        self.ar_ix = self._get_ar_mask(self.xn)
        self._calculate_all_params()

    def _calculate_all_params(self):
        """
        Calculate values of all parameters at mesh nodes and boundaries.
        """
        # nodes
        inds, dx = self._inds_dx(self.xn)
        for param in self.input_params_nodes:
            if param in self.params_active:
                continue
            self.vn[param] = self.calculate(
                param, self.xn, z=0, inds=inds, dx=dx)
        # active region
        inds, dx = inds[self.ar_ix], dx[self.ar_ix]
        for param in self.params_active:
            self.vn[param] = self.calculate(
                param, self.xn[self.ar_ix], z=0, inds=inds, dx=dx)

        # boundaries
        inds, dx = self._inds_dx(self.xb)
        for param in self.vb:
            self.vb[param] = self.calculate(
                param, self.xb, z=0, inds=inds, dx=dx)

        self._update_Vt()

    def make_dimensionless(self):
        "Make every parameter dimensionless."
        self.xn /= units.x
        self.xb /= units.x
        for d in (self.vn, self.vb, self.sol):
            for param in d:
                if d[param] is not None:
                    d[param] /= units.dct[param]
        return super().make_dimensionless()

    def original_units(self):
        "Convert all values back to original units."
        self.xn *= units.x
        self.xb *= units.x
        for d in (self.vn, self.vb, self.sol):
            for param in d:
                if d[param] is not None:
                    d[param] *= units.dct[param]
        return super().original_units()

    def make_lcn_solver(self):
        """
        Make solver for electrostatic potential distribution along the
        vertical axis at equilibrium assuming local charge neutrality.
        """
        Nc = self.vn['Nc']
        Nv = self.vn['Nv']
        Ec = self.vn['Ec']
        Ev = self.vn['Ev']
        Vt = self.vn['Vt']

        def f(psi):
            n = semicond.n(psi, 0, Nc, Ec, Vt)
            p = semicond.p(psi, 0, Nv, Ev, Vt)
            return self.vn['C_dop'] - n + p

        def fdot(psi):
            ndot = semicond.dn_dpsi(psi, 0, Nc, Ec, Vt)
            pdot = semicond.dp_dpsi(psi, 0, Nv, Ev, Vt)
            return pdot - ndot

        # initial guess with Boltzmann statistics
        ni = eq.intrinsic_concentration(Nc, Nv, Ec, Ev, Vt)
        Ei = eq.intrinsic_level(Nc, Nv, Ec, Ev, Vt)
        Ef_i = eq.Ef_lcn_boltzmann(self.vn['C_dop'], ni, Ei, Vt)

        # Jacobian is a vector -> element-wise division
        sol = newton.NewtonSolver(res=f, jac=fdot, x0=Ef_i,
                                  linalg_solver=lambda A, b: b / A)
        return sol

    def solve_lcn(self, maxiter=100, fluct=1e-8, omega=1.0):
        """
        Find potential distribution at zero external bias assuming local
        charge neutrality. Uses Newton's method implemented in `NewtonSolver`.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of Newton's method iterations.
        fluct : float, optional
            Fluctuation of solution that is needed to stop iterating before
            reaching `maxiter` steps.
        omega : float, optional
            Damping parameter.

        """
        solver = self.make_lcn_solver()
        solver.solve(maxiter, fluct, omega)
        if solver.fluct[-1] > fluct:
            warnings.warn('LaserDiode1D.solve_lcn(): fluctuation ' +
                          f'{solver.fluct[-1]:.3e} exceeds {fluct:.3e}.')

        self.vn['psi_lcn'] = solver.x.copy()
        self.vn['n0'] = semicond.n(psi=self.vn['psi_lcn'], phi_n=0,
                                   Nc=self.vn['Nc'], Ec=self.vn['Ec'],
                                   Vt=self.vn['Vt'])
        self.vn['p0'] = semicond.p(psi=self.vn['psi_lcn'], phi_p=0,
                                   Nv=self.vn['Nv'], Ev=self.vn['Ev'],
                                   Vt=self.vn['Vt'])

    def make_equilibrium_solver(self):
        """
        Generate solver for electrostatic potential distribution along
        vertical axis at equilibrium.
        """
        if self.vn['psi_lcn'] is None:
            self.solve_lcn()

        h = self.xn[1:] - self.xn[:-1]
        w = self.xb[1:] - self.xb[:-1]
        v = self.vn

        def res(psi):
            n = semicond.n(psi, 0, v['Nc'], v['Ec'], v['Vt'])
            p = semicond.p(psi, 0, v['Nv'], v['Ev'], v['Vt'])
            r = eq.poisson_res(psi, n, p, h, w, v['eps'], self.eps_0, self.q,
                               v['C_dop'])
            return r

        def jac(psi):
            n = semicond.n(psi, 0, v['Nc'], v['Ec'], v['Vt'])
            ndot = semicond.dn_dpsi(psi, 0, v['Nc'], v['Ec'], v['Vt'])
            p = semicond.p(psi, 0, v['Nv'], v['Ev'], v['Vt'])
            pdot = semicond.dp_dpsi(psi, 0, v['Nv'], v['Ev'], v['Vt'])
            j = eq.poisson_jac(psi, n, ndot, p, pdot, h, w, v['eps'],
                               self.eps_0, self.q, v['C_dop'])
            return j

        psi_init = v['psi_lcn'].copy()
        sol = newton.NewtonSolver(
            res=res, jac=jac, x0=psi_init,
            linalg_solver=lambda A, b: solve_banded((1, 1), A, b),
            inds=np.arange(1, len(psi_init) - 1))
        return sol

    def solve_equilibrium(self, maxiter=100, fluct=1e-8, omega=1.0):
        """
        Calculate electrostatic potential distribution at equilibrium (zero
        external bias). Uses Newton's method implemented in `NewtonSolver`.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of Newton's method iterations.
        fluct : float, optional
            Fluctuation of solution that is needed to stop iterating before
            reacing `maxiter` steps.
        omega : float, optional
            Damping parameter.

        """
        sol = self.make_equilibrium_solver()
        sol.solve(maxiter, fluct, omega)
        if sol.fluct[-1] > fluct:
            warnings.warn('LaserDiode1D.solve_equilibrium(): fluctuation ' +
                          f'{sol.fluct[-1]:.3e} exceeds {fluct:.3e}.')
        self.vn['psi_bi'] = sol.x.copy()
        self.sol['psi'] = sol.x.copy()
        self.sol['phi_n'] = np.zeros_like(sol.x)
        self.sol['phi_p'] = np.zeros_like(sol.x)
        self._update_densities()
