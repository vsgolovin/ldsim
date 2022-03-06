import warnings
import numpy as np
from scipy.linalg import solve_banded
from ldsim.preprocessing.design import LaserDiode
from ldsim.mesh import generate_nonuniform_mesh
from ldsim import units, newton, semicond
import ldsim.semicond.equilibrium as eq
from ldsim.transport import flux


class LaserDiodeModel1d(LaserDiode):
    input_params_nodes = [
        'Ev', 'Ec', 'Eg', 'Nd', 'Na', 'C_dop', 'Nc', 'Nv', 'mu_n', 'mu_p',
        'tau_n', 'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr', 'g0', 'N_tr',
        'fca_e', 'fca_h', 'T']
    input_params_boundaries = ['Ev', 'Ec', 'Nc', 'Nv', 'mu_n', 'mu_p']
    params_active = ['g0', 'N_tr']
    calculated_params_nodes = [
        'Vt', 'psi_lcn', 'n0', 'p0', 'psi_bi', 'wg_mode']
    calculated_params_boundaries = [
        'jn', 'djn_dpsi1', 'djn_dpsi2', 'djn_dphin1', 'djn_dphin2',
        'jp', 'djp_dpsi1', 'djp_dpsi2', 'djp_dphip1', 'djp_dphip2']
    solution_arrays = [
        'psi', 'phi_n', 'phi_p', 'n', 'p', 'dn_dpsi', 'dn_dphin',
        'dp_dpsi', 'dp_dphip']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # mesh
        self.xn = None     # nodes
        self.xb = None     # volume boundaries
        self.ar_ix = None  # active region mask for xn

        # parameters at mesh nodes and volume boundaries
        self.vn = dict.fromkeys(self.input_params_nodes
                                + self.calculated_params_nodes)
        self.vb = dict.fromkeys(self.input_params_boundaries
                                + self.calculated_params_boundaries)
        self.sol = dict.fromkeys(self.solution_arrays)
        self.S = 1e-12
        self.iterations = None
        self.fluct = None
        self.voltage = 0.0

        # current density discretization scheme
        self.j_discr = 'mSG'

    def _update_Vt(self):
        self.vn['Vt'] = self.vn['T'] * self.kb

    def _update_current_densities(self):
        psi = self.sol['psi']
        Vt = self.vn['Vt']
        B_plus = flux.bernoulli((psi[1:] - psi[:-1]) / Vt)
        B_minus = flux.bernoulli(-(psi[1:] - psi[:-1]) / Vt)
        Bdot_plus = flux.bernoulli_dot((psi[1:] - psi[:-1]) / Vt)
        Bdot_minus = flux.bernoulli_dot(-(psi[1:] - psi[:-1]) / Vt)
        h = self.xn[1:] - self.xn[:-1]

        if self.j_discr == 'mSG':
            jn_function = self._jn_mSG
            jp_function = self._jp_mSG
        else:
            assert self.j_discr == 'SG'
            jn_function = self._jn_SG
            jp_function = self._jp_SG
        jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2 = jn_function(
            B_plus, B_minus, Bdot_plus, Bdot_minus, h
        )
        jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2 = jp_function(
            B_plus, B_minus, Bdot_plus, Bdot_minus, h
        )
        self.vb['jn'] = jn
        self.vb['djn_dpsi1'] = djn_dpsi1
        self.vb['djn_dpsi2'] = djn_dpsi2
        self.vb['djn_dphin1'] = djn_dphin1
        self.vb['djn_dphin2'] = djn_dphin2
        self.vb['jp'] = jp
        self.vb['djp_dpsi1'] = djp_dpsi1
        self.vb['djp_dpsi2'] = djp_dpsi2
        self.vb['djp_dphip1'] = djp_dphip1
        self.vb['djp_dphip2'] = djp_dphip2

    def _jn_SG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h):
        "Electron current density, Scharfetter-Gummel scheme."
        psi = self.sol['psi']
        phi_n = self.sol['phi_n']
        Nc = self.vb['Nc']
        Ec = self.vb['Ec']
        Vt = self.vn['Vt']
        mu_n = self.vb['mu_n']
        q = self.q

        # electron densities at finite volume boundaries
        n1 = semicond.n(psi[:-1], phi_n[:-1], Nc, Ec, Vt[:-1])
        n2 = semicond.n(psi[1:], phi_n[1:], Nc, Ec, Vt[1:])

        # forward (2-2) and backward (1-1) derivatives w.r.t. potentials
        dn1_dpsi1 = semicond.dn_dpsi(psi[:-1], phi_n[:-1], Nc, Ec, Vt[:-1])
        dn2_dpsi2 = semicond.dn_dpsi(psi[1:], phi_n[1:], Nc, Ec, Vt[1:])
        dn1_dphin1 = semicond.dn_dphin(psi[:-1], phi_n[:-1], Nc, Ec, Vt[:-1])
        dn2_dphin2 = semicond.dn_dphin(psi[1:], phi_n[1:], Nc, Ec, Vt[1:])

        # electron current density and its derivatives
        jn = flux.SG_jn(n1, n2, B_plus, B_minus, h)
        djn_dpsi1 = flux.SG_djn_dpsi1(n1, n2, dn1_dpsi1, B_minus, Bdot_plus,
                                      Bdot_minus, h, Vt, q, mu_n)
        djn_dpsi2 = flux.SG_djn_dpsi2(n1, n2, dn2_dpsi2, B_plus, Bdot_plus,
                                      Bdot_minus, h, Vt, q, mu_n)
        djn_dphin1 = flux.SG_djn_dphin1(dn1_dphin1, B_minus, h, Vt, q, mu_n)
        djn_dphin2 = flux.SG_djn_dphin2(dn2_dphin2, B_plus, h, Vt, q, mu_n)
        return jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2

    def _jp_SG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h):
        "Hole current density, Scharfetter-Gummel scheme."
        psi = self.sol['psi']
        phi_p = self.sol['phi_p']
        Nv = self.vb['Nv']
        Ev = self.vb['Ev']
        Vt = self.vn['Vt']
        mu_p = self.vb['mu_p']
        q = self.q

        # hole densities at finite bolume boundaries
        p1 = semicond.p(psi[:-1], phi_p[:-1], Nv, Ev, Vt[:-1])
        p2 = semicond.p(psi[1:], phi_p[1:], Nv, Ev, Vt[1:])

        # forward (2-2) and backward derivatives w.r.t. potentials
        dp1_dpsi1 = semicond.dp_dpsi(psi[:-1], phi_p[:-1], Nv, Ev, Vt[:-1])
        dp2_dpsi2 = semicond.dp_dpsi(psi[1:], phi_p[1:], Nv, Ev, Vt[1:])
        dp1_dphip1 = semicond.dp_dphip(psi[:-1], phi_p[:-1], Nv, Ev, Vt[:-1])
        dp2_dphip2 = semicond.dp_dphip(psi[1:], phi_p[1:], Nv, Ev, Vt[1:])

        # hole current density and its derivatives
        jp = flux.SG_jp(p1, p2, B_plus, B_minus, h, Vt, q, mu_p)
        djp_dpsi1 = flux.SG_djp_dpsi1(p1, p2, dp1_dpsi1, B_plus, Bdot_plus,
                                      Bdot_minus, h, Vt, q, mu_p)
        djp_dpsi2 = flux.SG_djp_dpsi2(p1, p2, dp2_dpsi2, B_minus, Bdot_plus,
                                      Bdot_minus, h, Vt, q, mu_p)
        djp_dphip1 = flux.SG_djp_dphip1(dp1_dphip1, B_plus, h, Vt, q, mu_p)
        djp_dphip2 = flux.SG_djp_dphip2(dp2_dphip2, B_minus, h, Vt, q, mu_p)
        return jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2

    def _jn_mSG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h):
        "Electron current density, modified Scharfetter-Gummel scheme."
        # aliases
        psi = self.sol['psi']
        phi_n = self.sol['phi_n']
        Nc = self.vb['Nc']
        Ec = self.vb['Ec']
        Vt = self.vn['Vt']
        mu_n = self.vb['mu_n']
        q = self.q
        F = semicond.prob_functions.fermi_approx
        F_dot = semicond.prob_functions.fermi_dot_approx

        # electron current density
        eta_n1 = (psi[:-1] - phi_n[:-1] - Ec) / Vt[:-1]
        eta_n2 = (psi[1:] - phi_n[1:] - Ec) / Vt[1:]
        exp_eta_n1 = np.exp(eta_n1)
        exp_eta_n2 = np.exp(eta_n2)
        gn = flux.g(eta_n1, eta_n2, F)
        jn_SG = flux.oSG_jn(exp_eta_n1, exp_eta_n2, B_plus, B_minus, h, Nc,
                            Vt, q, mu_n)
        jn = jn_SG * gn

        # derivatives
        gdot_n1 = flux.gdot(gn, eta_n1, F, F_dot) / Vt[:-1]
        gdot_n2 = flux.gdot(gn, eta_n2, F, F_dot) / Vt[1:]
        djn_dpsi1_SG = flux.oSG_djn_dpsi1(
            exp_eta_n1, exp_eta_n2, B_minus, Bdot_plus, Bdot_minus,
            h, Nc, q, mu_n
        )
        djn_dpsi1 = flux.mSG_jdot(jn_SG, djn_dpsi1_SG, gn, gdot_n1)
        djn_dpsi2_SG = flux.oSG_djn_dpsi2(
            exp_eta_n1, exp_eta_n2, B_plus, Bdot_plus, Bdot_minus,
            h, Nc, q, mu_n
        )
        djn_dpsi2 = flux.mSG_jdot(jn_SG, djn_dpsi2_SG, gn, gdot_n2)
        djn_dphin1_SG = flux.oSG_djn_dphin1(exp_eta_n1, B_minus, h,
                                            Nc, q, mu_n)
        djn_dphin1 = flux.mSG_jdot(jn_SG, djn_dphin1_SG, gn, -gdot_n1)
        djn_dphin2_SG = flux.oSG_djn_dphin2(exp_eta_n2, B_plus, h,
                                            Nc, q, mu_n)
        djn_dphin2 = flux.mSG_jdot(jn_SG, djn_dphin2_SG, gn, -gdot_n2)
        return jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2

    def _jp_mSG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h):
        "Hole current density, modified Scharfetter-Gummel scheme."
        # aliases
        psi = self.sol['psi']
        phi_p = self.sol['phi_p']
        Nv = self.vb['Nv']
        Ev = self.vb['Ev']
        Vt = self.vn['Vt']
        mu_p = self.vb['mu_p']
        q = self.q
        F = semicond.prob_functions.fermi_approx
        F_dot = semicond.prob_functions.fermi_dot_approx

        # hole current density
        eta_1 = (-psi[:-1] + phi_p[:-1] + Ev) / Vt[:-1]
        eta_2 = (-psi[1:] + phi_p[1:] + Ev) / Vt[1:]
        exp_eta_p1 = np.exp(eta_1)
        exp_eta_p2 = np.exp(eta_2)
        gp = flux.g(eta_1, eta_2, F)
        jp_SG = flux.osG_jp(exp_eta_p1, exp_eta_p2, B_plus, B_minus,
                            h, Nv, q, mu_p)
        jp = jp_SG * gp

        # derivatives
        gdot_p1 = flux.gdot(gp, eta_1, F, F_dot) / Vt[:-1]
        gdot_p2 = flux.gdot(gp, eta_2, F, F_dot) / Vt[1:]
        djp_dpsi1_SG = flux.oSG_djp_dpsi(
            exp_eta_p1, exp_eta_p2, B_plus, Bdot_plus, Bdot_minus,
            h, Nv, q, mu_p
        )
        djp_dpsi1 = flux.mSG_jdot(jp_SG, djp_dpsi1_SG, gp, -gdot_p1)
        djp_dpsi2_SG = flux.oSG_djp_dpsi2(
            exp_eta_p1, exp_eta_p2, B_minus, Bdot_plus, Bdot_minus,
            h, Nv, q, mu_p
        )
        djp_dpsi2 = flux.mSG_jdot(jp_SG, djp_dpsi2_SG, gp, -gdot_p2)
        djp_dphip1_SG = flux.oSG_djp_dphip1(exp_eta_p1, B_plus,
                                            h, Nv, q, mu_p)
        djp_dphip1 = flux.mSG_jdot(jp_SG, djp_dphip1_SG, gp, gdot_p1)
        djp_dphip2_SG = flux.oSG_djp_dphip2(exp_eta_p2, B_minus,
                                            h, Nv, q, mu_p)
        djp_dphip2 = flux.mSG_jdot(jp_SG, djp_dphip2_SG, gp, gdot_p2)
        return jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2

    def _update_densities(self):
        # aliases (pointers)
        psi = self.sol['psi']
        phi_n = self.sol['phi_n']
        phi_p = self.sol['phi_p']
        Nc = self.vn['Nc']
        Nv = self.vn['Nv']
        Ec = self.vn['Ec']
        Ev = self.vn['Ev']
        Vt = self.vn['Vt']
        # densities
        self.sol['n'] = semicond.n(psi, phi_n, Ev, Ec, Vt)
        self.sol['p'] = semicond.p(psi, phi_p, Nv, Ev, Vt)
        # derivatives
        self.sol['dn_dpsi'] = semicond.dn_dpsi(psi, phi_n, Nc, Ec, Vt)
        self.sol['dn_dphin'] = semicond.dn_dphin(psi, phi_n, Nc, Ec, Vt)
        self.sol['dp_dpsi'] = semicond.dp_dpsi(psi, phi_p, Nv, Ev, Vt)
        self.sol['dp_dphip'] = semicond.dp_dphip(psi, phi_p, Nv, Ev, Vt)

    def solve_waveguide(self, step=1e-7, n_modes=3, remove_layers=(0, 0)):
        rv = super().solve_waveguide(step, n_modes, remove_layers)
        if self.xn is not None:
            self.vn['wg_mode'] = self.get_waveguide_mode(self.xn,
                                                         self.is_dimensionless)
        return rv

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

        # waveguide mode
        if self.gamma:  # not None -> calculated waveguide mode profile
            self.vn['wg_mode'] = self.get_waveguide_mode(self.xn,
                                                         self.is_dimensionless)
        self._update_Vt()

    def make_dimensionless(self):
        "Make every parameter dimensionless."
        self.xn /= units.x
        self.xb /= units.x
        for d in (self.vn, self.vb, self.sol):
            for param in d:
                if d[param] is not None and param in units.dct:
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

    # nonequilibrium drift-diffusion
    def apply_voltage(self, V):
        """
        Modify current solution so that boundary conditions at external
        voltage `V` are satisfied.
        """
        # scale voltage
        if self.is_dimensionless:
            V /= units.V

        # apply boundary conditions
        self.sol['psi'][0] = self.vn['psi_bi'][0] - V / 2
        self.sol['psi'][-1] = self.vn['psi_bi'][-1] + V / 2
        for phi in ('phi_n', 'phi_p'):
            self.sol[phi][0] = -V / 2
            self.sol[phi][-1] = V / 2
        # carrier densities at boundaries should not change

        # track solution convergence
        self.iterations = 0
        self.fluct = []  # fluctuation for every Newton iteration
        self.voltage = V
