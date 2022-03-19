import warnings
import numpy as np
from scipy.linalg import solve_banded
from scipy import sparse
from ldsim.preprocessing.design import LaserDiode
from ldsim.mesh import generate_nonuniform_mesh
from ldsim import units, newton, semicond
import ldsim.semicond.equilibrium as eq
import ldsim.semicond.recombination as rec
from ldsim.transport import flux, vrs


class LaserDiodeModel1d(LaserDiode):
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
        'R_aug', 'dRrad_dpsi', 'dRaug_dphin', 'dRaug_dphip']
    calculated_params_boundaries = [
        'Vt',
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
        self.vb['Vt'] = self.vb['T'] * self.kb

    def _update_current_densities(self):
        psi = self.sol['psi']
        B_plus = flux.bernoulli((psi[1:] - psi[:-1]) / self.vb['Vt'])
        B_minus = flux.bernoulli(-(psi[1:] - psi[:-1]) / self.vb['Vt'])
        Bdot_plus = flux.bernoulli_dot((psi[1:] - psi[:-1]) / self.vb['Vt'])
        Bdot_minus = flux.bernoulli_dot(-(psi[1:] - psi[:-1]) / self.vb['Vt'])
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
        Vt = self.vb['Vt']
        mu_n = self.vb['mu_n']
        q = self.q

        # electron densities at finite volume boundaries
        n1 = semicond.n(psi[:-1], phi_n[:-1], Nc, Ec, Vt)
        n2 = semicond.n(psi[1:], phi_n[1:], Nc, Ec, Vt)

        # forward (2-2) and backward (1-1) derivatives w.r.t. potentials
        dn1_dpsi1 = semicond.dn_dpsi(psi[:-1], phi_n[:-1], Nc, Ec, Vt)
        dn2_dpsi2 = semicond.dn_dpsi(psi[1:], phi_n[1:], Nc, Ec, Vt)
        dn1_dphin1 = semicond.dn_dphin(psi[:-1], phi_n[:-1], Nc, Ec, Vt)
        dn2_dphin2 = semicond.dn_dphin(psi[1:], phi_n[1:], Nc, Ec, Vt)

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
        Vt = self.vb['Vt']
        mu_p = self.vb['mu_p']
        q = self.q

        # hole densities at finite bolume boundaries
        p1 = semicond.p(psi[:-1], phi_p[:-1], Nv, Ev, Vt)
        p2 = semicond.p(psi[1:], phi_p[1:], Nv, Ev, Vt)

        # forward (2-2) and backward derivatives w.r.t. potentials
        dp1_dpsi1 = semicond.dp_dpsi(psi[:-1], phi_p[:-1], Nv, Ev, Vt)
        dp2_dpsi2 = semicond.dp_dpsi(psi[1:], phi_p[1:], Nv, Ev, Vt)
        dp1_dphip1 = semicond.dp_dphip(psi[:-1], phi_p[:-1], Nv, Ev, Vt)
        dp2_dphip2 = semicond.dp_dphip(psi[1:], phi_p[1:], Nv, Ev, Vt)

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
        eta_n1 = (psi[:-1] - phi_n[:-1] - Ec) / self.vn['Vt'][:-1]
        eta_n2 = (psi[1:] - phi_n[1:] - Ec) / self.vn['Vt'][1:]
        exp_eta_n1 = np.exp(eta_n1)
        exp_eta_n2 = np.exp(eta_n2)
        gn = flux.g(eta_n1, eta_n2, F)
        jn_SG = flux.oSG_jn(exp_eta_n1, exp_eta_n2, B_plus, B_minus, h, Nc,
                            self.vb['Vt'], q, mu_n)
        jn = jn_SG * gn

        # derivatives
        gdot_n1 = flux.gdot(gn, eta_n1, F, F_dot) / self.vn['Vt'][:-1]
        gdot_n2 = flux.gdot(gn, eta_n2, F, F_dot) / self.vn['Vt'][1:]
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
        eta_1 = (-psi[:-1] + phi_p[:-1] + Ev) / self.vn['Vt'][:-1]
        eta_2 = (-psi[1:] + phi_p[1:] + Ev) / self.vn['Vt'][1:]
        exp_eta_p1 = np.exp(eta_1)
        exp_eta_p2 = np.exp(eta_2)
        gp = flux.g(eta_1, eta_2, F)
        jp_SG = flux.oSG_jp(exp_eta_p1, exp_eta_p2, B_plus, B_minus,
                            h, Nv, self.vb['Vt'], q, mu_p)
        jp = jp_SG * gp

        # derivatives
        gdot_p1 = flux.gdot(gp, eta_1, F, F_dot) / self.vn['Vt'][:-1]
        gdot_p2 = flux.gdot(gp, eta_2, F, F_dot) / self.vn['Vt'][1:]
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

    def _transport_system(self):
        m = len(self.xn) - 2            # number of inner nodes
        h = self.xn[1:] - self.xn[:-1]  # mesh steps
        w = self.xb[1:] - self.xb[:-1]  # 1D volumes
        R = (self.vn['R_srh'] + self.vn['R_rad'] + self.vn['R_aug'])[1:-1]
        dR_dpsi = sum(self.vn[f'd{r}_dpsi']
                      for r in ['Rsrh', 'Rrad', 'Raug'])[1:-1]
        dR_dphin = sum(self.vn[f'd{r}_dphin']
                       for r in ['Rsrh', 'Rrad', 'Raug'])[1:-1]
        dR_dphip = sum(self.vn[f'd{r}_dphip']
                       for r in ['Rsrh', 'Rrad', 'Raug'])[1:-1]

        # aliases
        jn = self.vb['jn']
        jp = self.vb['jp']

        # vector of residuals for van Roosbroeck-Shockley system
        # [poisson, electron current density continuity, hole -/-]
        residuals = np.zeros(m * 3)
        residuals[:m] = vrs.poisson_res(
            self.sol['psi'], self.sol['n'], self.sol['p'], h, w,
            self.vn['eps'], self.eps_0, self.q, self.vn['C_dop']
        )
        residuals[m:2*m] = self.q * R * w - (jn[1:] - jn[:-1])
        residuals[2*m:] = -self.q * R * w - (jp[1:] - jp[:-1])

        # Jacobian
        # 1. Poisson's equation
        j11 = vrs.poisson_dF_dpsi(self.sol['dn_dpsi'], self.sol['dp_dpsi'], h,
                                  w, self.vn['eps'], self.eps_0, self.q)
        j12 = vrs.poisson_dF_dphin(self.sol['dn_dphin'], w, self.eps_0, self.q)
        j13 = vrs.poisson_dF_dphip(self.sol['dp_dphip'], w, self.eps_0, self.q)

        # 2. Electron current density continuity equation
        j21 = vrs.jn_dF_dpsi(self.vb['djn_dpsi1'], self.vb['djn_dpsi2'],
                             dR_dpsi, w, self.q, m)
        j22 = vrs.jn_dF_dphin(self.vb['djn_dphin1'], self.vb['djn_dphin2'],
                              dR_dphin, w, self.q, m)
        j23 = vrs.jn_dF_dphip(dR_dphip, w, self.q, m)

        # 3. Hole current continiuty equation
        j31 = vrs.jp_dF_dpsi(self.vb['djp_dpsi1'], self.vb['djp_dpsi2'],
                             dR_dpsi, w, self.q, m)
        j32 = vrs.jp_dF_dphin(dR_dphin, w, self.q, m)
        j33 = vrs.jp_dF_dphip(self.vb['djp_dphip1'], self.vb['djp_dphip2'],
                              dR_dphip, w, self.q, m)

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

        # assemble sparse matrix
        diags = [2*m, m, 1, 0, -1, -m+1, -m, -m-1, -2*m+1, -2*m, -2*m-1]

        return data, diags, residuals

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
        self.sol['n'] = semicond.n(psi, phi_n, Nc, Ec, Vt)
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
        for param in self.input_params_boundaries:
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
                if d[param] is not None and param in units.dct:
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
            warnings.warn('LaserDiode1D.solve_lcn(): fluctuation '
                          + f'{solver.fluct[-1]:.3e} exceeds {fluct:.3e}.')

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
            warnings.warn('LaserDiode1D.solve_equilibrium(): fluctuation '
                          + f'{sol.fluct[-1]:.3e} exceeds {fluct:.3e}.')
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

        # update calculated arrays
        self._update_current_densities()
        self._update_recombination_rates()

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
        m = len(self.xn) - 2
        data, diags, residuals = self._transport_system()
        J = sparse.spdiags(data=data, diags=diags, m=m*3, n=m*3, format='csc')
        dx = sparse.linalg.spsolve(J, -residuals)

        # calculate and save fluctuation
        fluct = np.sqrt(
            np.sum(dx**2) / (
                np.sum(self.sol['psi'][1:-1]**2)
                + np.sum(self.sol['phi_n'][1:-1]**2)
                + np.sum(self.sol['phi_p'][1:-1]**2)
            )
        )
        self.fluct.append(fluct)

        # update current solution
        self.sol['psi'][1:-1] += dx[:m] * omega
        self.sol['phi_n'][1:-1] += dx[m:2*m] * omega
        self.sol['phi_p'][1:-1] += dx[2*m:] * omega
        self._update_densities()
        self._update_current_densities()
        self._update_recombination_rates()

        return fluct

    def get_current_density(self):
        return (self.vb['jn'] + self.vb['jp']).mean()
