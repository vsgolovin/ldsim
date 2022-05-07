from typing import Iterable, NoReturn
from copy import deepcopy
import warnings
import itertools
import numpy as np
from scipy import sparse
from ldsim.preprocessing.design import LaserDiode
from ldsim.mesh import generate_nonuniform_mesh
from ldsim import units, newton, semicond
import ldsim.semicond.recombination as rec
from ldsim.semicond.gain import gain_2p
from ldsim.transport import flux


input_params_nodes = [
    'Ev', 'Ec', 'Eg', 'Nd', 'Na', 'C_dop', 'Nc', 'Nv', 'mu_n', 'mu_p',
    'tau_n', 'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr', 'g0', 'N_tr',
    'fca_e', 'fca_h', 'T']
input_params_boundaries = ['Ev', 'Ec', 'Nc', 'Nv', 'mu_n', 'mu_p', 'T']
params_active = ['g0', 'N_tr']
calculated_params_nodes = [
    'Vt', 'psi_lcn', 'n0', 'p0', 'psi_bi', 'wg_mode',
    'R_srh', 'dRsrh_dpsi', 'dRsrh_dphin', 'dRsrh_dphip',
    'R_rad', 'dRrad_dpsi', 'dRrad_dphin', 'dRrad_dphip',
    'R_aug', 'dRrad_dpsi', 'dRaug_dphin', 'dRaug_dphip',
    'gain', 'dg_dpsi', 'dg_dphin', 'dg_dphip', 'fca',
    'R_st', 'dRst_dpsi', 'dRst_dphin', 'dRst_dphip', 'dRst_dS']
calculated_params_boundaries = [
    'Vt',
    'jn', 'djn_dpsi1', 'djn_dpsi2', 'djn_dphin1', 'djn_dphin2',
    'jp', 'djp_dpsi1', 'djp_dpsi2', 'djp_dphip1', 'djp_dphip2']
solution_arrays = [
    'psi', 'phi_n', 'phi_p', 'n', 'p', 'dn_dpsi', 'dn_dphin',
    'dp_dpsi', 'dp_dphip']


class LaserDiodeModel1d(LaserDiode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # mesh
        self.xn = None     # nodes
        self.xb = None     # volume boundaries
        self.ar_ix = None  # active region mask for xn

        # parameters at mesh nodes and volume boundaries
        self.vn = dict.fromkeys(input_params_nodes
                                + calculated_params_nodes
                                + params_active)
        self.vb = dict.fromkeys(input_params_boundaries
                                + calculated_params_boundaries)
        self.sol = dict.fromkeys(solution_arrays)
        self.gamma = None
        self.n_eff = None
        self.waveguide_function = None
        self.S = 1e-12
        self.alpha_fca = 0.0
        self.Gain = 0.0
        self.iterations = None
        self.fluct = None
        self.voltage = 0.0

        # current density discretization scheme
        self.j_discr = 'mSG'

    # preprocessing methods
    def solve_waveguide(self, step=1e-7, n_modes=3, remove_layers=(0, 0)):
        rv = super().solve_waveguide(step=step, n_modes=n_modes,
                                     remove_layers=remove_layers)
        i = np.argmax(rv['gammas'])
        self.gamma = rv['gammas'][i]
        self.n_eff = rv['n_eff'][i]
        self.waveguide_function = rv['waveguide_function']
        if self.xn is not None:
            self._update_waveguide_mode()
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
        for param in input_params_nodes:
            if param in params_active:
                continue
            self.vn[param] = self.calculate(
                param, self.xn, z=0, inds=inds, dx=dx)
        # active region
        inds, dx = inds[self.ar_ix], dx[self.ar_ix]
        for param in params_active:
            self.vn[param] = self.calculate(
                param, self.xn[self.ar_ix], z=0, inds=inds, dx=dx)

        # boundaries
        inds, dx = self._inds_dx(self.xb)
        for param in input_params_boundaries:
            self.vb[param] = self.calculate(
                param, self.xb, z=0, inds=inds, dx=dx)

        # waveguide mode
        if self.gamma is not None:  # calculated waveguide mode profile
            self._update_waveguide_mode()
        self._update_Vt()

    def solve_lcn(self, maxiter=100, fluct=1e-8, omega=1.0):
        """
        Find potential distribution at zero external bias assuming local
        charge neutrality via Newton's method.

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
        solver = newton.LCNSolver1d(self.vn)
        solver.solve(maxiter, fluct, omega)
        if solver.fluct[-1] > fluct:
            warnings.warn('LaserDiode1D.solve_lcn(): fluctuation '
                          + f'{solver.fluct[-1]:.3e} exceeds {fluct:.3e}.')

        self.vn['psi_lcn'] = solver.x.copy()
        self.vn['n0'] = semicond.n(psi=self.vn['psi_lcn'], phi_n=0,
                                   Nc=self.vn['Nc'], Ec=self.vn['Ec'],
                                   Vt=self.vn['Vt'])
        self.vn['p0'] = semicond.p(psi=self.vn['psi_lcn'], phi_p=0,
                                   Nv=self.vn['Nv'], Ev=self.vn['Ev'],
                                   Vt=self.vn['Vt'])

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
        if self.vn['psi_lcn'] is None:
            self.solve_lcn()
        sol = newton.EquilibriumSolver1d(self.vn, self.xn, self.xb, self.q,
                                         self.eps_0)
        sol.solve(maxiter, fluct, omega)
        if sol.fluct[-1] > fluct:
            warnings.warn('LaserDiode1D.solve_equilibrium(): fluctuation '
                          + f'{sol.fluct[-1]:.3e} exceeds {fluct:.3e}.')
        self.vn['psi_bi'] = sol.x.copy()
        self.sol['psi'] = sol.x.copy()
        self.sol['phi_n'] = np.zeros_like(sol.x)
        self.sol['phi_p'] = np.zeros_like(sol.x)
        self._update_densities()

    # scale all parameters
    def make_dimensionless(self):
        "Make every parameter dimensionless."
        super().make_dimensionless()
        self.xn /= units.x
        self.xb /= units.x
        self.S /= units.dct['S']
        self.alpha_fca /= 1 / units.x
        self.Gain /= 1 / units.x
        self.voltage /= units.V
        for d in (self.vn, self.vb, self.sol):
            for param in d:
                if d[param] is not None and param in units.dct:
                    d[param] /= units.dct[param]

    def original_units(self):
        "Convert all values back to original units."
        super().original_units()
        self.xn *= units.x
        self.xb *= units.x
        self.S *= units.dct['S']
        self.alpha_fca *= 1 / units.x
        self.Gain *= 1 / units.x
        self.voltage *= units.V
        for d in (self.vn, self.vb, self.sol):
            for param in d:
                if d[param] is not None and param in units.dct:
                    d[param] *= units.dct[param]

    # methods for updating arrays usign currently defined potentials
    def _update_waveguide_mode(self):
        if self.is_dimensionless:
            self.vn['wg_mode'] = self.waveguide_function(
                self.xn * units.x) * units.x
        else:
            self.vn['wg_mode'] = self.waveguide_function(self.xn)

    def _update_Vt(self):
        self.vn['Vt'] = self.vn['T'] * self.kb
        self.vb['Vt'] = self.vb['T'] * self.kb

    def _update_densities(self):
        # function arguments
        kwargs_n = dict(psi=self.sol['psi'], phi_n=self.sol['phi_n'],
                        Nc=self.vn['Nc'], Ec=self.vn['Ec'], Vt=self.vn['Vt'])
        kwargs_p = dict(psi=self.sol['psi'], phi_p=self.sol['phi_p'],
                        Nv=self.vn['Nv'], Ev=self.vn['Ev'], Vt=self.vn['Vt'])
        # densities
        self.sol['n'] = semicond.n(**kwargs_n)
        self.sol['p'] = semicond.p(**kwargs_p)
        # derivatives
        self.sol['dn_dpsi'] = semicond.dn_dpsi(**kwargs_n)
        self.sol['dn_dphin'] = semicond.dn_dphin(**kwargs_n)
        self.sol['dp_dpsi'] = semicond.dp_dpsi(**kwargs_p)
        self.sol['dp_dphip'] = semicond.dp_dphip(**kwargs_p)

    def _update_current_densities(self):
        psi = self.sol['psi']
        delta_psi = psi[..., 1:] - psi[..., :-1]
        B_plus = flux.bernoulli(delta_psi / self.vb['Vt'])
        B_minus = flux.bernoulli(-delta_psi / self.vb['Vt'])
        Bdot_plus = flux.bernoulli_dot(delta_psi / self.vb['Vt'])
        Bdot_minus = flux.bernoulli_dot(-delta_psi / self.vb['Vt'])
        h = self.xn[..., 1:] - self.xn[..., :-1]

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
        psi_1 = psi[..., :-1]
        psi_2 = psi[..., 1:]
        phi_n = self.sol['phi_n']
        phi_n_1 = phi_n[..., :-1]
        phi_n_2 = phi_n[..., 1:]
        Nc = self.vb['Nc']
        Ec = self.vb['Ec']
        Vt = self.vb['Vt']
        mu_n = self.vb['mu_n']
        q = self.q

        # electron densities at finite volume boundaries
        n1 = semicond.n(psi_1, phi_n_1, Nc, Ec, Vt)
        n2 = semicond.n(psi_2, phi_n_2, Nc, Ec, Vt)

        # forward (2-2) and backward (1-1) derivatives w.r.t. potentials
        dn1_dpsi1 = semicond.dn_dpsi(psi_1, phi_n_1, Nc, Ec, Vt)
        dn2_dpsi2 = semicond.dn_dpsi(psi_2, phi_n_2, Nc, Ec, Vt)
        dn1_dphin1 = semicond.dn_dphin(psi_1, phi_n_1, Nc, Ec, Vt)
        dn2_dphin2 = semicond.dn_dphin(psi_2, phi_n_2, Nc, Ec, Vt)

        # electron current density and its derivatives
        jn = flux.SG_jn(n1, n2, B_plus, B_minus, h, Vt, q, mu_n)
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
        psi_1 = psi[..., :-1]
        psi_2 = psi[..., 1:]
        phi_p = self.sol['phi_p']
        phi_p_1 = phi_p[..., :-1]
        phi_p_2 = phi_p[..., 1:]
        Nv = self.vb['Nv']
        Ev = self.vb['Ev']
        Vt = self.vb['Vt']
        mu_p = self.vb['mu_p']
        q = self.q

        # hole densities at finite bolume boundaries
        p1 = semicond.p(psi_1, phi_p_1, Nv, Ev, Vt)
        p2 = semicond.p(psi_2, phi_p_2, Nv, Ev, Vt)

        # forward (2-2) and backward derivatives w.r.t. potentials
        dp1_dpsi1 = semicond.dp_dpsi(psi_1, phi_p_1, Nv, Ev, Vt)
        dp2_dpsi2 = semicond.dp_dpsi(psi_2, phi_p_2, Nv, Ev, Vt)
        dp1_dphip1 = semicond.dp_dphip(psi_1, phi_p_1, Nv, Ev, Vt)
        dp2_dphip2 = semicond.dp_dphip(psi_2, phi_p_2, Nv, Ev, Vt)

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
        mu_n = self.vb['mu_n']
        q = self.q
        F = semicond.prob_functions.fermi_approx
        F_dot = semicond.prob_functions.fermi_dot_approx

        # electron current density
        eta_n1 = (psi[..., :-1]
                  - phi_n[..., :-1] - Ec) / self.vn['Vt'][..., :-1]
        eta_n2 = (psi[..., 1:]
                  - phi_n[..., 1:] - Ec) / self.vn['Vt'][..., 1:]
        exp_eta_n1 = np.exp(eta_n1)
        exp_eta_n2 = np.exp(eta_n2)
        gn = flux.g(eta_n1, eta_n2, F)
        jn_SG = flux.oSG_jn(exp_eta_n1, exp_eta_n2, B_plus, B_minus, h, Nc,
                            self.vb['Vt'], q, mu_n)
        jn = jn_SG * gn

        # derivatives
        gdot_n1 = flux.gdot(gn, eta_n1, F, F_dot) / self.vn['Vt'][..., :-1]
        gdot_n2 = flux.gdot(gn, eta_n2, F, F_dot) / self.vn['Vt'][..., 1:]
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
        mu_p = self.vb['mu_p']
        q = self.q
        F = semicond.prob_functions.fermi_approx
        F_dot = semicond.prob_functions.fermi_dot_approx

        # hole current density
        eta_1 = (-psi[..., :-1]
                 + phi_p[..., :-1] + Ev) / self.vn['Vt'][..., :-1]
        eta_2 = (-psi[..., 1:]
                 + phi_p[..., 1:] + Ev) / self.vn['Vt'][..., 1:]
        exp_eta_p1 = np.exp(eta_1)
        exp_eta_p2 = np.exp(eta_2)
        gp = flux.g(eta_1, eta_2, F)
        jp_SG = flux.oSG_jp(exp_eta_p1, exp_eta_p2, B_plus, B_minus,
                            h, Nv, self.vb['Vt'], q, mu_p)
        jp = jp_SG * gp

        # derivatives
        gdot_p1 = flux.gdot(gp, eta_1, F, F_dot) / self.vn['Vt'][..., :-1]
        gdot_p2 = flux.gdot(gp, eta_2, F, F_dot) / self.vn['Vt'][..., 1:]
        djp_dpsi1_SG = flux.oSG_djp_dpsi1(
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

    def _update_recombination_rates(self):
        """
        Calculate recombination (Shockley-Read-Hall, radiative and Auger)
        rates, as well as their derivatives w.r.t. potentials.
        """
        # aliases
        n = self.sol['n']
        dn_dpsi = self.sol['dn_dpsi']
        dn_dphin = self.sol['dn_dphin']
        p = self.sol['p']
        dp_dpsi = self.sol['dp_dpsi']
        dp_dphip = self.sol['dp_dphip']
        n0 = self.vn['n0']
        p0 = self.vn['p0']
        B = self.vn['B']

        # Shockley-Read-Hall recombination
        self.vn['R_srh'] = rec.srh_R(
            n, p, n0, p0, self.vn['tau_n'], self.vn['tau_p']
        )
        self.vn['dRsrh_dpsi'] = rec.srh_Rdot(
            n, dn_dpsi, p, dp_dpsi, n0, p0, self.vn['tau_n'], self.vn['tau_p']
        )
        self.vn['dRsrh_dphin'] = rec.srh_Rdot(
            n, dn_dphin, p, 0, n0, p0, self.vn['tau_n'], self.vn['tau_p']
        )
        self.vn['dRsrh_dphip'] = rec.srh_Rdot(
            n, 0, p, dp_dphip, n0, p0, self.vn['tau_n'], self.vn['tau_p']
        )

        # radiative recombination
        self.vn['R_rad'] = rec.rad_R(n, p, n0, p, B)
        self.vn['dRrad_dpsi'] = rec.rad_Rdot(n, dn_dpsi, p, dp_dpsi, B)
        self.vn['dRrad_dphin'] = rec.rad_Rdot(n, dn_dphin, p, 0, B)
        self.vn['dRrad_dphip'] = rec.rad_Rdot(n, 0, p, dp_dphip, B)

        # Auger recombination
        self.vn['R_aug'] = rec.auger_R(
            n, p, n0, p0, self.vn['Cn'], self.vn['Cp']
        )
        self.vn['dRaug_dpsi'] = rec.auger_Rdot(
            n, dn_dpsi, p, dp_dpsi, n0, p0, self.vn['Cn'], self.vn['Cp']
        )
        self.vn['dRaug_dphin'] = rec.auger_Rdot(
            n, dn_dphin, p, 0, n0, p0, self.vn['Cn'], self.vn['Cp']
        )
        self.vn['dRaug_dphip'] = rec.auger_Rdot(
            n, 0, p, dp_dphip, n0, p0, self.vn['Cn'], self.vn['Cp']
        )

    def _update_gain(self):
        "Calculate gain ind its derivatives. Compatible with 2D model."
        gain, dg_dn, dg_dp = gain_2p(
            n=self.sol['n'][self.ar_ix],
            p=self.sol['p'][self.ar_ix],
            g0=self.vn['g0'],
            N_tr=self.vn['N_tr'],
            return_derivatives=True
        )
        # discard negative gain (absorption)
        mask_abs = (gain < 0)
        gain[mask_abs] = 0.0
        dg_dn[mask_abs] = 0.0
        dg_dp[mask_abs] = 0.0

        # calculate derivatives w.r.t. to potentials and save results
        self.vn['gain'] = gain
        self.vn['dg_dpsi'] = (dg_dn * self.sol['dn_dpsi'][self.ar_ix]
                              + dg_dp * self.sol['dp_dpsi'][self.ar_ix])
        self.vn['dg_dphin'] = dg_dn * self.sol['dn_dphin'][self.ar_ix]
        self.vn['dg_dphip'] = dg_dp * self.sol['dp_dphip'][self.ar_ix]

    def _update_fca(self):
        "Calculate free-carrier absorption. Compatible with 2D model."
        self.vn['fca'] = (
            (self.sol['n']*self.vn['fca_e'] + self.sol['p']*self.vn['fca_h'])
            * self.vn['wg_mode'])
        self.alpha_fca = np.sum(self.vn['fca'][..., 1:-1] * (self.xb[..., 1:]
                                - self.xb[..., :-1]), axis=-1)

    def _update_Rst(self, S=None):
        "Calculate stimulated recombination rate and its derivatives."
        dx = (self.xb[..., 1:] - self.xb[..., :-1])[self.ar_ix[..., 1:-1]]
        k = self.vg * dx * self.vn['wg_mode'][self.ar_ix]
        if S is None:
            S = self.S
        self.vn['R_st'] = k * self.vn['gain'] * S
        self.vn['dRst_dpsi'] = k * self.vn['dg_dpsi'] * S
        self.vn['dRst_dphin'] = k * self.vn['dg_dphin'] * S
        self.vn['dRst_dphip'] = k * self.vn['dg_dphip'] * S
        self.vn['dRst_dS'] = k * self.vn['gain']

    def _update_gain_fca_Rst(self):
        self._update_gain()
        self.Gain = np.sum(self.vn['gain'] * self.vn['wg_mode'][self.ar_ix]
                           * (self.xb[1:] - self.xb[:-1])[self.ar_ix[1:-1]])
        self._update_fca()
        self._update_Rst()

    # methods for solving the drift-diffusion system
    def _lasing_system(self):
        m = len(self.xn) - 2
        # [poisson, electron current density, hole -/-,
        #  photon density rate equation]
        residuals = np.empty(m * 3 + 1)  # [poisson, electron current]
        data, diags, residuals[:-1] = newton.transport_system(
            xn=self.xn, vn=self.vn, xb=self.xb, vb=self.vb, sol=self.sol,
            q=self.q, eps_0=self.eps_0, index=None
        )

        # update vector of residuals
        inds = np.where(self.ar_ix)[0] - 1
        net_gain = self.Gain - self.alpha_i - self.alpha_m - self.alpha_fca
        R_rad_ar_int = np.sum(
            self.vn['R_rad'][self.ar_ix]
            * (self.xb[1:] - self.xb[:-1])[self.ar_ix[1:-1]]
            * self.vn['wg_mode'][self.ar_ix]
        )
        residuals[m + inds] += self.q * self.vn['R_st']
        residuals[2 * m + inds] += -self.q * self.vn['R_st']
        residuals[-1] = (self.vg * net_gain * self.S
                         + self.beta_sp * R_rad_ar_int)

        # construct Jacobian from diagonals
        data[6, inds] += self.q * self.vn['dRst_dpsi']        # j21
        data[3, m+inds] += self.q * self.vn['dRst_dphin']     # j22
        data[1, 2*m+inds] += self.q * self.vn['dRst_dphip']   # j23
        data[9, inds] += -self.q * self.vn['dRst_dpsi']       # j31
        data[6, m+inds] += -self.q * self.vn['dRst_dphin']    # j32
        data[3, 2*m+inds] += -self.q * self.vn['dRst_dphip']  # j33
        J = sparse.spdiags(data, diags, m=3*m+1, n=3*m+1, format='lil')

        # fill rightmost column
        J[m+inds, -1] = self.q * self.vn['dRst_dS']
        J[2*m+inds, -1] = -self.q * self.vn['dRst_dS']

        # fill bottom row
        w = (self.xb[1:] - self.xb[:-1])[self.ar_ix[1:-1]]
        T = self.vn['wg_mode'][self.ar_ix]
        J[-1, inds] = (
            self.vg * self.vn['dg_dpsi'] * self.S
            + self.beta_sp * self.vn['dRrad_dpsi'][self.ar_ix]
        ) * w * T
        J[-1, m+inds] = (
            self.vg * self.vn['dg_dphin'] * self.S
            + self.beta_sp * self.vn['dRrad_dphin'][self.ar_ix]
        ) * w * T
        J[-1, 2*m+inds] = (
            self.vg * self.vn['dg_dphip'] * self.S
            + self.beta_sp * self.vn['dRrad_dphip'][self.ar_ix]
        ) * w * T
        J[-1, -1] = self.vg * net_gain

        return J.tocsc(), residuals

    def apply_voltage(self, V):
        """
        Modify current solution so that boundary conditions at external
        voltage `V` are satisfied.
        """
        # scale voltage
        if self.is_dimensionless:
            V /= units.V

        # apply boundary conditions
        self.sol['psi'][..., 0] = self.vn['psi_bi'][..., 0] - V / 2
        self.sol['psi'][..., -1] = self.vn['psi_bi'][..., -1] + V / 2
        for phi in ('phi_n', 'phi_p'):
            self.sol[phi][..., 0] = -V / 2
            self.sol[phi][..., -1] = V / 2
        # carrier densities at boundaries should not change

        # track solution convergence
        self.iterations = 0
        self.fluct = []  # fluctuation for every Newton iteration
        self.voltage = V

        # update calculated arrays
        self._update_current_densities()
        self._update_recombination_rates()

    def _solve_transport_system(self):
        m = len(self.xn) - 2
        data, diags, residuals = newton.transport_system(
            xn=self.xn, vn=self.vn, xb=self.xb, vb=self.vb, sol=self.sol,
            q=self.q, eps_0=self.eps_0, index=None
        )
        J = sparse.spdiags(data=data, diags=diags, m=m*3, n=m*3, format='csc')
        return sparse.linalg.spsolve(J, -residuals)

    def transport_step(self, omega=0.1):
        """
        Perform a single Newton step for the transport problem.
        As a result,  solution vector `x` (concatenated `psi`, `phi_n` and
        `phi_p' vectors) is updated by `dx` with damping parameter `omega`,
        i.e. `x += omega * dx`.

        Parameters
        ----------
        omega : float
            Damping parameter.

        Returns
        -------
        fluct : float
            Solution fluctuation, i.e. ratio of `dx` and `x` L2 norms.

        """
        # solve the system
        dx = self._solve_transport_system()
        m = dx.shape[-1] // 3

        # calculate and save fluctuation
        fluct = np.sqrt(
            np.sum(dx**2) / (
                np.sum(self.sol['psi'][..., 1:-1]**2)
                + np.sum(self.sol['phi_n'][..., 1:-1]**2)
                + np.sum(self.sol['phi_p'][..., 1:-1]**2)
            )
        )
        self.fluct.append(fluct)

        # update current solution
        self.sol['psi'][..., 1:-1] += dx[..., :m] * omega
        self.sol['phi_n'][..., 1:-1] += dx[..., m:2*m] * omega
        self.sol['phi_p'][..., 1:-1] += dx[..., 2*m:] * omega
        self._update_densities()
        self._update_current_densities()
        self._update_recombination_rates()
        self.iterations += 1

        return fluct

    def lasing_step(self, omega=0.1, omega_S=(1.0, 0.1)):
        """
        Perform a single Newton step for the lasing problem -- combination
        of the transport problem with the photon density rate equation.
        As a result,  solution vector `x` (all potentials and photon
        densities) is updated by `dx` with damping parameter `omega`,
        i.e. `x += omega * dx`.

        Parameters
        ----------
        omega : float
            Damping parameter for potentials.
        omega_S : (float, float)
            Damping parameter for photon density `S`. First value is used
            for increasing `S`, second -- for decreasing `S`.
            Separate values are needed to prevent `S` converging to a
            negative value near threshold.

        Returns
        -------
        fluct : number
            Solution fluctuation, i.e. ratio of `dx` and `x` L2 norms.

        """
        # `apply_voltage` does not calculate gain
        if len(self.fluct) == 0:
            self._update_gain_fca_Rst()

        # solve the system
        J, residuals = self._lasing_system()
        dx = sparse.linalg.spsolve(J, -residuals)

        # calculate and save fluctuation
        fluct = np.sqrt(
            np.sum(dx**2) / (
                np.sum(self.sol['psi'][1:-1]**2)
                + np.sum(self.sol['phi_n'][1:-1]**2)
                + np.sum(self.sol['phi_p'][1:-1]**2)
                + self.S**2
            )
        )
        self.fluct.append(fluct)

        # update current solution
        m = len(self.xn) - 2
        self.sol['psi'][1:-1] += dx[:m] * omega
        self.sol['phi_n'][1:-1] += dx[m:2*m] * omega
        self.sol['phi_p'][1:-1] += dx[2*m:3*m] * omega
        if dx[-1] > 0:
            self.S += dx[-1] * omega_S[0]
        else:
            self.S += dx[-1] * omega_S[1]
        self._update_densities()
        self._update_current_densities()
        self._update_recombination_rates()
        self._update_gain_fca_Rst()
        self.iterations += 1

        return fluct

    # getters and methods for exporting data
    # all returned values are in original units
    def get_current_density(self):
        "Get current density through device (A/cm2)."
        j = (self.vb['jn'] + self.vb['jp']).mean(axis=-1)
        if self.is_dimensionless:
            j *= units.j
        return j

    def get_current(self):
        "Get current through device (A)."
        area = self.w * self.L
        if self.is_dimensionless:
            area *= units.x**2
        return self.get_current_density() * area

    def get_output_power(self):
        "Get output power from both facets (W)."
        P = self.photon_energy * self.S * self.vg*self.alpha_m * self.w*self.L
        if self.is_dimensionless:
            P *= units.P
        return P

    def get_recombination_current_density(self, s: str = 'rad'):
        dx = self.xb[..., 1:] - self.xb[..., :-1]
        if s == 'srh':
            R = self.vn['R_srh'][..., 1:-1]
        elif s == 'rad':
            R = self.vn['R_rad'][..., 1:-1]
        elif s == 'aug':
            R = self.vn['R_aug'][..., 1:-1]
        else:
            raise ValueError('Unknown spontaneous recombination mechanism '
                             + f'{s}.')
        j = self.q * np.sum(R * dx, axis=-1)
        if self.is_dimensionless:
            j *= units.j
        return j

    def get_recombination_current(self, s: str = 'rad'):
        """
        Get current (A) associated with spontaneous recombination
        mechanism `s`. This parameter should be one of:
          - `'srh'` -- Shockley-Read-Hall recombination,
          - `'rad'` -- radiative recombination,
          - `'aug'` -- Auger recombination.
        """
        area = self.w * self.L
        if self.is_dimensionless:
            area *= units.x**2
        return self.get_recombination_current_density(s) * area

    def get_loss(self):
        """
        Get total internal absorption (1/cm), i.e. user-specified base
        absorption level `alpha_i` plus free-carrier absorbtion `alpha_fca`.
        Does not include output loss.
        """
        alpha = self.alpha_fca + self.alpha_i
        return alpha * 1/units.x if self.is_dimensionless else alpha

    def get_nodes(self):
        "Get coordinates of mesh nodes (cm)."
        return self.xn*units.x if self.is_dimensionless else deepcopy(self.xn)

    def get_potential(self):
        "Get electrostatic potential (V)."
        if self.is_dimensionless:
            return self.sol['psi'] * units.V
        return deepcopy(self.sol['psi'])

    def get_Ec_Ev(self):
        "Get conduction band bottom and valence band top (eV)."
        Ec = self.vn['Ec'] - self.sol['psi']
        Ev = self.vn['Ev'] - self.sol['psi']
        if self.is_dimensionless:
            Ec *= units.E
            Ev *= units.E
        return Ec, Ev

    def get_carrier_densities(self):
        "Get distributions of electron and hole densities."
        k = 1 / (units.x**3) if self.is_dimensionless else 1.0
        return (self.sol['n']*k, self.sol['p']*k)

    def get_fermi_levels(self):
        "Get electron and hole quasi-Fermi levels."
        fn = -self.sol['phi_n']
        fp = -self.sol['phi_p']
        return (fn*units.E, fp*units.E) if self.is_dimensionless else (fn, fp)

    @staticmethod
    def save_csv(fname: str, arrays: Iterable[np.ndarray],
                 labels: Iterable[str], sep: str = ',') -> NoReturn:
        with open(fname, 'w') as outfile:
            outfile.write(sep.join(labels))  # header
            for vals in zip(*arrays):        # 1D, same length
                outfile.write('\n')
                outfile.write(sep.join(map('{:e}'.format, vals)))

    def save_results(self, fname: str, sep: str = ',',
                     x_to_um: bool = True) -> NoReturn:
        x = self.get_nodes()
        if x_to_um:
            x *= 1e4
        jn = np.pad(self.vb['jn'], 1, mode='edge')
        jp = np.pad(self.vb['jp'], 1, mode='edge')
        if self.is_dimensionless:
            jn *= units.j
            jp *= units.j
        labels = ['x', 'Ec', 'Ev', 'fn', 'fp', 'n', 'p', 'psi', 'jn', 'jp']
        arrays = [x, *(self.get_Ec_Ev()), *(self.get_fermi_levels()),
                  *(self.get_carrier_densities()), self.get_potential(),
                  jn, jp]
        return self.save_csv(fname, arrays, labels, sep)


class LaserDiodeModel2d(LaserDiodeModel1d):
    def __init__(self, *args,  num_slices=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.zn = None  # longitudinal (z) axis nodes
        self.zb = None  # z axis volume boundaries
        self.Sf = None  # forward propagating photon densities at `zb`
        self.Sb = None  # backward -/-
        self.set_number_of_slices(num_slices)

    def set_number_of_slices(self, m: int):
        """
        Set number of cross-sections (along x, perpendicular to z).
        """
        self.mz = m
        self.alpha_fca = np.zeros(m)
        self.Gain = np.zeros(m)
        self.Sf = np.ones(m + 1) * 1e-12
        self.Sb = np.ones(m + 1) * 1e-12
        self.S = np.ones(m) * 1e-12

    # scale all parameters
    def make_dimensionless(self):
        "Make every parameter dimensionless."
        self.Sf /= units.dct['S']
        self.Sb /= units.dct['S']
        return super().make_dimensionless()

    def original_units(self):
        "Convert all values back to original units."
        self.Sf *= units.dct['S']
        self.Sb *= units.dct['S']
        return super().original_units()

    def generate_nonuniform_mesh(self, method='first', step_uni=5e-8,
                                 step_min=1e-7, step_max=20e-7, sigma=100e-7,
                                 y_ext=[None, None]):
        """
        Generate mesh that is nonuniform along the vertical (x) axis and
        uniform along the longitudinal (z) axis. Uses local nonuniformity in
        bandgap (Eg) to choose mesh spacing.
        See `ldsim.mesh.generate_nonuniform_mesh` for detailed description.

        Parameter `method` decides the way this mesh is generated.
        * 'first' -- use the first slice along the z axis (z = 0). Default,
        useful when bandgap does not depend on z, as this option avoids
        repetative computations.
        * 'mean` -- average bandgap over all slices, then generate mesh using
        local change in this "mean" bandgap.
        * 'finest' -- generate mesh for every slice, then use the one with the
        most points.
        """
        assert not self.is_dimensionless
        thickness = self.get_thickness()
        num_points = int(round(thickness // step_uni))
        x = np.linspace(0, thickness, num_points)
        z = np.linspace(0, self.L, self.mz + 1)

        # generate 1D array of mesh nodes along x (`xn`)
        if method == 'finest':
            xn = []
            for i in range(self.mz):
                Eg = self.calculate('Eg', x, z[i])
                xn_i, _ = generate_nonuniform_mesh(
                    x, Eg, step_min=step_min, step_max=step_max,
                    sigma=sigma, y_ext=y_ext
                )
                if len(xn_i) > len(xn):
                    xn = xn_i
        else:
            if method == 'first':
                Eg = self.calculate('Eg', x,  0)
            else:
                assert method == 'mean'
                Eg = np.zeros((self.mz, num_points))
                for i in range(self.mz):
                    Eg[i] = self.calculate('Eg', x, z[i])
                Eg = np.mean(Eg, axis=0)

            xn, _ = generate_nonuniform_mesh(
                x, Eg, step_min=step_min, step_max=step_max,
                sigma=sigma, y_ext=y_ext
            )

        # save 2D arrays as mesh
        xb = (xn[1:] + xn[:-1]) / 2
        self.xn = np.tile(xn, (self.mz, 1))
        self.xb = np.tile(xb, (self.mz, 1))
        zn = (z[1:] + z[:-1]) / 2
        self.zn = np.tile(zn[:, np.newaxis], (1, len(xn)))
        self.zb = np.tile(z[:, np.newaxis], (1, len(xn)))
        self.ar_ix = self._get_ar_mask(self.xn)
        self._calculate_all_params()

    def solve_waveguide(self, step=1e-7, n_modes=3, remove_layers=(0, 0)):
        """
        Calculate vertical mode profile for every vertical slice. Finds
        `n_modes` solutions of the eigenvalue problem with the highest
        eigenvalues (effective indices) and picks the one with the highest
        optical confinement factor (active region overlap). This
        implementation asserts that the found mode profiles of every slice
        correspond to the same mode.

        Parameters
        ----------
        step : float, optional
            Uniform mesh step (cm).
        n_modes : int, optional
            Number of calculated eigenproblem solutions.
        remove_layers : (int, int), optional
            Number of layers to exclude from calculated refractive index
            profile at each side of the device. Useful to exclude contact
            layers.

        """
        self.gamma = np.zeros(self.mz)
        self.n_eff = np.zeros_like(self.gamma)
        self.waveguide_function = []
        if self.zn is None:
            zn = np.linspace(0, self.L, self.mz + 1)
            zn = (zn[1:] + zn[:-1]) / 2
        else:
            zn = self.zn[:, 0]

        for i in range(self.mz):
            rv = LaserDiode.solve_waveguide(
                self, z=zn[i], step=step, n_modes=n_modes,
                remove_layers=remove_layers
            )
            j = np.argmax(rv['gammas'])
            self.gamma[i] = rv['gammas'][j]
            self.n_eff[i] = rv['n_eff'][j]
            self.waveguide_function.append(rv['waveguide_function'])

        if self.xn is not None:
            self._update_waveguide_mode()

    def solve_lcn(self, maxiter=100, fluct=1e-8, omega=1.0):
        """
        Find potential distribution at zero external bial assuming local charge
        neutrality via Newtons's method.

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
        # calculate LCN potential for every slice
        psi_lcn = np.zeros_like(self.xn)
        for i in range(self.mz):
            solver = newton.LCNSolver1d(self.vn, i)
            solver.solve(maxiter, fluct, omega)
            if solver.fluct[-1] > fluct:
                warnings.warn('LCN solver: fluctuation'
                              + f'{solver.fluct[-1]:.3e} exceeds {fluct:.3e}.')
            psi_lcn[i] = solver.x

        # store results
        self.vn['psi_lcn'] = psi_lcn
        self.vn['n0'] = semicond.n(psi=psi_lcn, phi_n=0, Nc=self.vn['Nc'],
                                   Ec=self.vn['Ec'], Vt=self.vn['Vt'])
        self.vn['p0'] = semicond.p(psi=psi_lcn, phi_p=0, Nv=self.vn['Nv'],
                                   Ev=self.vn['Ev'], Vt=self.vn['Vt'])

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
        if self.vn['psi_lcn'] is None:
            self.solve_lcn()

        # calculate built-in potential for every slice
        psi_bi = np.zeros_like(self.xn)
        for i in range(self.mz):
            solver = newton.EquilibriumSolver1d(self.vn, self.xn, self.xb,
                                                self.q, self.eps_0, i)
            solver.solve(maxiter, fluct, omega)
            if solver.fluct[-1] > fluct:
                warnings.warn('Equilibrium solver: fluctuation'
                              + f'{solver.fluct[-1]:.3e} exceeds {fluct:.3e}.')
            psi_bi[i] = solver.x

        # save results
        self.vn['psi_bi'] = psi_bi
        self.sol['psi'] = psi_bi.copy()
        self.sol['phi_n'] = np.zeros_like(psi_bi)
        self.sol['phi_p'] = np.zeros_like(psi_bi)
        self._update_densities()

    def _update_waveguide_mode(self):
        assert self.waveguide_function
        wg_mode = np.zeros((self.mz, self.xn.shape[1]))
        for i in range(self.mz):
            f = self.waveguide_function[i]
            if self.is_dimensionless:
                wg_mode[i] = f(self.xn[i] * units.x) * units.x
            else:
                wg_mode[i] = f(self.xn[i])
        self.vn['wg_mode'] = wg_mode

    def _update_gain_fca_Rst(self):
        self._update_gain()
        mxa = self.ar_ix.sum(axis=1)
        Gain = (self.vn['gain'] * self.vn['wg_mode'][self.ar_ix]
                * (self.xb[:, 1:] - self.xb[:, :-1])[self.ar_ix[:, 1:-1]])
        self.Gain = np.zeros(self.mz)
        i1 = 0
        for k in range(self.mz):
            self.Gain[k] = np.sum(Gain[i1:i1+mxa[k]])
            i1 += mxa[k]
        self._update_fca()
        self._update_Rst(S=np.array([[self.S[k]] * np.sum(self.ar_ix[k])
                                     for k in range(self.mz)]).ravel())

    def _solve_transport_system(self):
        # initialize the system
        mx = self.xn.shape[1] - 2  # number of inner nodes along x axis
        data = np.zeros((11, mx * 3 * self.mz))
        residuals = np.zeros(mx * 3 * self.mz)

        # fill arrays
        for i in range(self.mz):
            i1 = i * mx * 3
            i2 = i1 + mx * 3
            data[:, i1:i2], diags, residuals[i1:i2] = newton.transport_system(
                xn=self.xn, vn=self.vn, xb=self.xb, vb=self.vb, sol=self.sol,
                q=self.q, eps_0=self.eps_0, index=i
            )

        # solve the system
        J = sparse.spdiags(data=data, diags=diags, m=mx*3*self.mz,
                           n=mx*3*self.mz, format='csc')
        dx = sparse.linalg.spsolve(J, -residuals)
        return dx.reshape((self.mz, 3 * mx))

    def _lasing_system(self):
        mx = self.xn.shape[1] - 2
        mxa = np.array([self.ar_ix[k].sum() for k in range(self.mz)])
        mz = self.mz

        # initialize Jacobian matrix
        # drift-diffusion system (F1, F2, F3)
        data = np.zeros((11, 3*mx*mz))     # J11-J33 (transport)
        J24 = np.zeros(sum(mxa))           # J24 = -J34
        # photon density (Sb and Sf) rate equations (F4, F5)
        J44 = np.zeros((2*mz, 2*mz))       # derivatives w.r.t. Sf/Sb

        # initialize vector of residuals
        residuals = np.zeros(3*mx*mz + 2*mz)

        # stuff
        w = self.xb[:, 1:] - self.xb[:, :-1]
        wM = w[self.ar_ix[:, 1:-1]] * self.vn['wg_mode'][self.ar_ix]
        G = self.Gain - self.alpha_fca - self.alpha_i
        dz = self.L / self.mz
        Sb_n = (self.Sb[:-1] + self.Sb[1:]) / 2
        Sf_n = (self.Sf[:-1] + self.Sf[1:]) / 2

        # iterate over slices
        for k in range(mz):
            i1 = 3 * mx * k
            i2 = i1 + 3 * mx
            data[:, i1:i2], diags, residuals[i1:i2] = newton.transport_system(
                xn=self.xn, vn=self.vn, xb=self.xb, vb=self.vb, sol=self.sol,
                q=self.q, eps_0=self.eps_0, index=k)

            # Jacobian, bottom right corner (dF_4,5 / dS_b,f)
            J44[k, k] = self.vg * (-1 / dz + G[k] / 2)
            if k < mz - 1:
                J44[k, k+1] = self.vg * (1 / dz + G[k] / 2)
            else:
                J44[mz-1, -1] = self.R2 * self.vg * (1 / dz + G[k] / 2)
            J44[mz+k, mz+k] = self.vg * (-1 / dz + G[k] / 2)
            if k > 0:
                J44[mz+k, mz+k-1] = self.vg * (1 / dz + G[k] / 2)
            else:
                J44[mz, 0] = self.R1 * self.vg * (1 / dz + G[k] / 2)

        # update vector of residuals to take into account stimulated emission
        # drift-diffusion equations
        inds = np.flatnonzero(self.ar_ix[:, 1:-1])
        i = 0
        for k, mxa_k in enumerate(mxa):
            inds[i:i+mxa_k] += k * 2 * mx
            i += mxa_k
        residuals[mx+inds] += self.q * self.vn['R_st']     # R_st is 1D
        residuals[2*mx+inds] += -self.q * self.vn['R_st']
        # photon density rate equations
        Rrad_dx = self.vn['R_rad'][self.ar_ix] * wM
        Rrad_integral = np.zeros(mz)
        i1 = 0
        for k in range(mz):
            Rrad_integral[k] = Rrad_dx[i1:i1+mxa[k]].sum()
            i1 += mxa[k]
        # Rrad_integral = (self.vn['R_rad'][self.ar_ix] * wM).sum()
        residuals[3*mx*mz:(3*mx+1)*mz] = \
            (self.vg * (self.Sb[1:] - self.Sb[:-1]) / dz
             + self.vg * (self.Gain - self.alpha_fca - self.alpha_i) * Sb_n
             + self.beta_sp * Rrad_integral / 2)
        residuals[(3*mx+1)*mz:] = \
            (-self.vg * (self.Sf[1:] - self.Sf[:-1]) / dz
             + self.vg * (self.Gain - self.alpha_fca - self.alpha_i) * Sf_n
             + self.beta_sp * Rrad_integral / 2)

        # update Jacobian values
        data[6, inds] += self.q * self.vn['dRst_dpsi']         # J21
        data[3, mx+inds] += self.q * self.vn['dRst_dphin']     # J22
        data[1, 2*mx+inds] += self.q * self.vn['dRst_dphip']   # J23
        data[9, inds] += -self.q * self.vn['dRst_dpsi']        # J31
        data[6, mx+inds] += -self.q * self.vn['dRst_dphin']    # J32
        data[3, 2*mx+inds] += -self.q * self.vn['dRst_dphip']  # J33
        J24[:] = self.q * self.vn['dRst_dS'] / 2

        # assemble Jacobian
        # drift-diffusion system diagonals
        J = sparse.spdiags(data, diags, format='lil',
                           m=3*mx*mz + 2*mz, n=3*mx*mz + 2*mz)

        # rightmost columns
        offsets = [[k] * mxa[k] for k in range(mz)]
        offsets = np.array(list(itertools.chain(*offsets))) + 3 * mx * mz
        for j in [0, 1, mz - 1, mz]:
            col = np.ones_like(offsets)
            if j == 0:
                col[:mxa[0]] = 1 + self.R1
            elif j == 1:
                col[-mxa[-1]:] = 0.0
            elif j == mz - 1:
                col[:mxa[0]] = 0.0
            else:
                col[-mxa[-1]:] = 1 + self.R2
            J[mx+inds, j+offsets] = J24 * col
            J[2*mx+inds, j+offsets] = -J24 * col

        # bottom rows
        _Sb = [[self.Sb[k]] * mxa[k] for k in range(mz)]
        _Sb = np.array(list(itertools.chain(*_Sb)))
        _Sf = [[self.Sf[k]] * mxa[k] for k in range(mz)]
        _Sf = np.array(list(itertools.chain(*_Sf)))
        ixa = self.ar_ix
        J[offsets, inds] = (self.beta_sp * self.vn['dRrad_dpsi'][ixa] * wM
                            + self.vg * self.vn['dg_dpsi'] * wM * _Sb)
        J[offsets+mz, inds] = (self.beta_sp * self.vn['dRrad_dpsi'][ixa] * wM
                               + self.vg * self.vn['dg_dpsi'] * wM * _Sf)
        J[offsets, mx+inds] = (self.beta_sp * self.vn['dRrad_dphin'][ixa] * wM
                               + self.vg * self.vn['dg_dphin'] * wM * _Sb)
        J[offsets+mz, mx+inds] = (self.beta_sp*self.vn['dRrad_dphin'][ixa] * wM
                                  + self.vg * self.vn['dg_dphin'] * wM * _Sf)
        J[offsets, 2*mx+inds] = (self.beta_sp*self.vn['dRrad_dphip'][ixa] * wM
                                 + self.vg * self.vn['dg_dphip'] * wM * _Sb)
        J[offsets+mz, 2*mx+inds] = (self.beta_sp*self.vn['dRrad_dphip'][ixa]*wM
                                    + self.vg * self.vn['dg_dphip'] * wM * _Sf)

        # bottom right corner
        J[3*mx*mz:, 3*mx*mz:] = J44

        return J.tocsc(), residuals

    def lasing_step(self, omega=0.1, omega_S=(1.0, 0.1)):
        """
        Perform a single Newton step for the lasing problem -- combination
        of the transport problem with the photon density rate equation.
        As a result,  solution vector `x` (all potentials and photon
        densities) is updated by `dx` with damping parameter `omega`,
        i.e. `x += omega * dx`.

        Parameters
        ----------
        omega : float
            Damping parameter for potentials.
        omega_S : (float, float)
            Damping parameter for photon density `S`. First value is used
            for increasing `S`, second -- for decreasing `S`.
            Separate values are needed to prevent `S` converging to a
            negative value near threshold.

        Returns
        -------
        fluct : number
            Solution fluctuation, i.e. ratio of `dx` and `x` L2 norms.

        """
        # `apply_voltage` does not calculate gain
        if len(self.fluct) == 0:
            self._update_gain_fca_Rst()

        # solve the system
        J, residuals = self._lasing_system()
        dx = sparse.linalg.spsolve(J, -residuals)

        # calculate and save fluctuation
        fluct = np.sqrt(
            np.sum(dx**2) / (
                np.sum(self.sol['psi'][:, 1:-1]**2)
                + np.sum(self.sol['phi_n'][:, 1:-1]**2)
                + np.sum(self.sol['phi_p'][:, 1:-1]**2)
                + np.sum(self.Sb[:-1]**2)
                + np.sum(self.Sf[1:]**2)
            )
        )
        self.fluct.append(fluct)

        # update current solution
        # potentials
        m = self.xn.shape[1] - 2
        for k in range(self.mz):
            i1 = 3*m*k
            self.sol['psi'][k, 1:-1] += dx[i1:i1+m] * omega
            self.sol['phi_n'][k, 1:-1] += dx[i1+m:i1+2*m] * omega
            self.sol['phi_p'][k, 1:-1] += dx[i1+2*m:i1+3*m] * omega

        # photon densities
        dx_S = dx[-2*self.mz:]
        ix = (dx_S[:self.mz] > 0)
        self.Sb[:-1][ix] += dx_S[:self.mz][ix] * omega_S[0]
        self.Sb[:-1][~ix] += dx_S[:self.mz][~ix] * omega_S[1]
        ix = (dx_S[self.mz:] > 0)
        self.Sf[1:][ix] += dx_S[self.mz:][ix] * omega_S[0]
        self.Sf[1:][~ix] += dx_S[self.mz:][~ix] * omega_S[1]
        self.Sb[-1] = self.Sf[-1] * self.R2  # BC
        self.Sf[0] = self.Sb[0] * self.R1
        self.S = (self.Sf[:-1] + self.Sf[1:] + self.Sb[:-1] + self.Sb[1:]) / 2

        self._update_densities()
        self._update_current_densities()
        self._update_recombination_rates()
        self._update_gain_fca_Rst()
        self.iterations += 1

        return fluct
