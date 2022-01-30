# -*- coding: utf-8 -*-
"""
1D/2D laser diode model. Based on simulating free carrier transport along
the vertical (x) axis using van Roosbroeck (drift-diffusion) system.
Can also solve 2D vertical-longitudinal (x-z) problem, in which a set of
identical vertical slices are connected via stimulated emission.
"""

import warnings
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import solve_banded
from scipy import sparse
import design
import constants as const
import units
import carrier_concentrations as cc
import equilibrium as eq
import newton
import waveguide
import vrs
import flux
import recombination as rec
import sdf

yin_params = ['Ev', 'Ec', 'Eg', 'Nd', 'Na', 'C_dop', 'Nc', 'Nv', 'mu_n',
              'mu_p', 'tau_n', 'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr',
              'g0', 'N_tr', 'fca_e', 'fca_h']
ybn_params = ['Ev', 'Ec', 'Nc', 'Nv', 'mu_n', 'mu_p']


class LaserDiode(object):

    def __init__(self, epi, L, w, R1, R2, lam, ng, alpha_i, beta_sp):
        """
        Class for storing all the model parameters of a 1D/2D laser diode.
        Initialized as a 1D model, as there is no difference between
        implemented 1D and 2D models below lasing threshold.

        Parameters
        ----------
        epi : design.EpiDesign
            Epitaxial design.
        L : number
            Resonator length (cm).
        w : number
            Stripe width (cm).
        R1 : float
            Back (x=0) mirror reflectivity (0<`R1`<=1).
        R2 : float
            Front (x=L) mirror reflectivity (0<`R2`<=1).
        lam : number
            Operating wavelength.
        ng : number
            Group refractive index.
        alpha_i : number
            Internal optical loss (cm-1). Should not include free carrier
            absorption.
        beta_sp : number
            Spontaneous emission factor, i.e. the fraction of spontaneous
            emission that is coupled with the lasing mode.

        """
        # checking if all the necessary parameters were specified
        # and if there is an active region
        assert isinstance(epi, design.EpiDesign)
        has_active_region = False
        self.ar_inds = list()
        for i, layer in enumerate(epi):
            assert True not in [list(yi) == [np.nan] for yi in layer.d.values()]
            if layer.active:
                has_active_region = True
                self.ar_inds.append(i)
        assert has_active_region
        self.epi = epi

        # constants
        self.Vt = const.kb*const.T
        self.q = const.q
        self.eps_0 = const.eps_0

        # device parameters
        self.L = L
        self.w = w
        self.R1 = R1
        self.R2 = R2
        assert all([r > 0 and r < 1 for r in (R1, R2)])
        self.alpha_m = 1/(2*L) * np.log(1/(R1*R2))
        self.lam = lam
        self.photon_energy = const.h * const.c / lam
        self.ng = ng
        self.vg = const.c / ng
        self.n_eff = None
        self.gamma = None
        self.alpha_i = alpha_i
        self.beta_sp = beta_sp

        # parameters at mesh nodes
        self.yin = dict()  # values at interior nodes
        self.ybn = dict()  # values at boundary nodes
        self.sol = dict()  # current solution (potentials and concentrations)
        self.sol['S'] = 1e-12  # essentially 0

        # 2D laser parameters
        self.mz = 1                  # number of z grid nodes
        self.dz = L                  # longitudinal grid step
        self.zin = np.array(L/2)     # z grid nodes
        self.zbn = np.array([0, L])  # z grid volume boundaries
        self.sol2d = list()          # solution at every slice
        self.ndim = 1                # initialize as 1D
        # densities of forward-and backward-propagating photons
        self.Sf = np.zeros(2)
        self.Sb = np.zeros(2)
        self._update_Sb_Sf_1D()

        self.gen_uniform_mesh(calc_params=False)
        self.is_dimensionless = False

    # grid generation and parameter calculation / scaling
    def gen_uniform_mesh(self, step=1e-7, calc_params=False):
        """
        Generate a uniform mesh with a specified `step`. Should result with at
        least 50 nodes, otherwise raises an exception.
        """
        d = self.epi.get_thickness()
        if d/step < 50:
            raise Exception("Mesh step (%f) is too large." % step)
        self.xin = np.arange(0, d, step)
        self.xbn = (self.xin[1:]+self.xin[:-1]) / 2
        self.nx = len(self.xin)  # number of grid nodes
        self.ar_ix = self._get_ar_ix()
        if calc_params:
            self.calc_all_params()

    def gen_nonuniform_mesh(self, step_min=1e-7, step_max=20e-7, step_uni=5e-8,
                            param='Eg', sigma=100e-7, y_ext=[0, 0]):
        """
        Generate nonuniform mesh with step inversely proportional to change
        in `param` value. Uses gaussian function with standard deviation
        `sigma` for smoothing.

        Parameters
        ----------
        step_min : float, optional
            Minimum step (cm). The default is 1e-7.
        step_max : float, optional
            Maximum step (cm). The default is 20e-7.
        step_uni : float, optional
            Step of the uniform mesh that is used to calculate `param` values.
            The default is `5e-8`.
        param : str, optional
            Name of the parameter that is used to decide local step.
            The default is `'Eg'`.
        sigma : float, optional
            Gaussian function standard deviation. The default is 100e-7.
        y_ext : number or NoneType, optional
            `param` values outside the laser boundaries. These are used for
            obtaining finer mesh at contacts. Pass [None, None] to disable
            this feature. The default is [0, 0].

        """
        def gauss(x, mu, sigma):
            return np.exp(-(x-mu)**2 / (2*sigma**2))

        self.gen_uniform_mesh(step=step_uni, calc_params=False)
        x = self.xin.copy()
        self._calculate_param(param, nodes='i')
        y = self.yin[param]

        # adding external values to y
        # same value as inside if y0 or yn is not a number
        if not isinstance(y_ext[0], (float, int)):
            y_ext[0] = y[0]
        if not isinstance(y_ext[-1], (float, int)):
            y_ext[1] = y[-1]
        y_ext = np.concatenate([np.array([y_ext[0]]),
                                y,
                                np.array([y_ext[1]])])

        # function for choosing local step size
        f = np.abs(y_ext[2:]-y_ext[:-2])  # change of y at every point
        fg = np.zeros_like(f)  # convolution for smoothing
        for i, xi in enumerate(x):
            g = gauss(x, xi, sigma)
            fg[i] = np.sum(f*g)
        fg_fun = interp1d(x, fg/fg.max())

        # generating new grid
        k = step_max - step_min
        new_grid = list()
        xi = 0
        while xi <= x[-1]:
            new_grid.append(xi)
            xi += step_min + k*(1 - fg_fun(xi))
        self.xin = np.array(new_grid)
        self.xbn = (self.xin[1:] + self.xin[:-1]) / 2
        self.nx = len(self.xin)
        self.ar_ix = self._get_ar_ix()
        self.calc_all_params()

        # return grid nodes, values of `param` at grid nodes
        # and function values used to determine local step
        return x, y, step_min + k*(1 - fg_fun(x))

    def calc_all_params(self):
        "Calculate all parameters' values at mesh nodes."
        inds, dx = self.epi._inds_dx(self.xin)
        for p in yin_params:
            if p in design.params_active:
                continue
            self._calculate_param(p, 'i', inds, dx)
        inds, dx = inds[self.ar_ix], dx[self.ar_ix]
        for p in design.params_active:
            self._calculate_param(p, 'i', inds, dx)
        inds, dx = self.epi._inds_dx(self.xbn)
        for p in ybn_params:
            inds, dx = self.epi._inds_dx(self.xbn)
            self._calculate_param(p, 'b', inds, dx)
        if self.n_eff is not None:  # waveguide problem has been solved
            self._calc_wg_mode()

    def _calculate_param(self, p, nodes='internal', inds=None, dx=None):
        """
        Calculate values of parameter `p` at all mesh nodes.

        Parameters
        ----------
        p : str
            Parameter name.
        nodes : str, optional
            Which mesh nodes -- `internal` (`i`) or `boundary` (`b`) -- should
            the values be calculated at. The default is 'internal'.
        inds : numpy.ndarray or None
            Layer index for each node.
        dx : numpy.ndarray or None
            Distance from layer left boundary for each node.

        Raises
        ------
        Exception
            If `p` is an unknown parameter name.

        """
        assert not self.is_dimensionless
        # picking nodes
        if p in design.params_active:
            assert nodes in ('i', 'internal')
            x = self.xin[self.ar_ix]
            d = self.yin
        elif p not in design.params:
            raise Exception('Error: unknown parameter %s' % p)
        else:
            if nodes == 'internal' or nodes == 'i':
                x = self.xin
                d = self.yin
            elif nodes == 'boundary' or nodes == 'b':
                x = self.xbn
                d = self.ybn

        # calculating values
        y = self.epi.calculate(p, x, inds, dx)
        d[p] = y  # modifies self.yin or self.ybn

    def _get_ar_ix(self, x=None, epi=None):
        "Mask for x, where elements belong to active region."
        if x is None:
            x = self.xin
        if epi is None:
            epi = self.epi
        bnds = epi.boundaries()
        ar_ix = np.zeros_like(x, dtype=bool)
        ar_inds = [i for i, lr in enumerate(epi) if lr.active]
        for ind in ar_inds:
            ar_ix |= np.logical_and(x > bnds[ind], x <= bnds[ind+1])
        return ar_ix

    def make_dimensionless(self):
        "Make every parameter dimensionless."
        if self.is_dimensionless:
            return

        # constants
        self.Vt /= units.V
        self.q /= units.q
        self.eps_0 /= units.q/(units.x*units.V)

        # device parameters
        self.L /= units.x
        self.w /= units.x
        self.alpha_m /= 1/units.x
        self.lam /= units.x
        self.photon_energy /= units.E
        self.vg /= units.x/units.t
        self.alpha_i /= 1/units.x

        # mesh
        self.xin /= units.x
        self.xbn /= units.x
        self.zin /= units.x
        self.zbn /= units.x
        self.dz /= units.x

        # arrays
        for key in self.yin:
            self.yin[key] /= units.dct[key]
        for key in self.ybn:
            self.ybn[key] /= units.dct[key]
        if self.ndim == 1:
            solutions = [self.sol]
        else:
            solutions = self.sol2d
        for sol in solutions:
            for key in sol:
                sol[key] /= units.dct[key]
        self.Sf /= units.n * units.x
        self.Sb /= units.n * units.x

        self.is_dimensionless = True

    def original_units(self):
        "Convert all values back to original units."
        if not self.is_dimensionless:
            return

        # constants
        self.Vt *= units.V
        self.q *= units.q
        self.eps_0 *= units.q/(units.x*units.V)

        # device parameters
        self.L *= units.x
        self.w *= units.x
        self.alpha_m *= 1/units.x
        self.lam *= units.x
        self.photon_energy *= units.E
        self.vg *= units.x/units.t
        self.alpha_i *= 1/units.x

        # mesh
        self.xin *= units.x
        self.xbn *= units.x
        self.zin *= units.x
        self.zbn *= units.x
        self.dz *= units.x

        # arrays
        for key in self.yin:
            self.yin[key] *= units.dct[key]
        for key in self.ybn:
            self.ybn[key] *= units.dct[key]
        if self.ndim == 1:
            solutions = [self.sol]
        else:
            solutions = self.sol2d
        for sol in solutions:
            for key in sol:
                sol[key] *= units.dct[key]
        self.Sb *= units.n * units.x
        self.Sf *= units.n * units.x

        self.is_dimensionless = False

    def to_2D(self, m):
        "Convert 1D problem to 2D with `m` slices along z axis."
        assert self.ndim == 1

        # z grid
        self.mz = m
        self.dz = self.L / m
        self.zin = np.arange(self.dz / 2, self.L, self.dz)
        self.zbn = np.linspace(0, self.L, self.mz + 1)

        # densities of forward- and backward-propagating photons
        S = self.sol['S']
        self.Sf = np.zeros(m + 1)
        self.Sf[0] = S * self.R1 / (1 + self.R1)
        self.Sf[1:-1] = S / 2
        self.Sf[-1] = S / (1 + self.R2)
        self.Sb = np.zeros(m + 1)
        self.Sb[0] = S / (1 + self.R1)
        self.Sb[1:-1] = S / 2
        self.Sb[-1] = S * self.R2 / (1 + self.R2)

        # self.sol -> self.sol2d
        self.sol2d = [dict() for _ in range(m)]
        for i in range(m):
            for key in ('psi', 'phi_n', 'phi_p', 'n', 'p',
                        'dn_dpsi', 'dn_dphin', 'dp_dpsi', 'dp_dphip'):
                self.sol2d[i][key] = self.sol[key].copy()
        self.iterations = 0
        self.fluct = list()
        self.ndim = 2

    def to_1D(self):
        "Convert 2D problem back to 1D."
        assert self.ndim == 2
        self.mz = 1
        self.dz = self.L
        self.zin = np.array([self.L / 2])
        self.zbn = np.array([0.0, self.L])
        self.sol = self.sol2d[0].copy()
        self.sol2d = list()
        self.Sb = np.array([self.Sb[0], self.Sb[-1]])
        self.Sf = np.array([self.Sf[0], self.Sf[-1]])
        self.iterations = 0
        self.fluct = list()
        self.ndim = 1

    def _update_Sb_Sf_1D(self):
        "Calculate `Sf` and `Sb` in a 1D model (S != f(z))."
        assert self.ndim == 1
        S = self.sol['S']
        R1, R2 = self.R1, self.R2
        self.Sb[0] = S * np.sqrt(R2) * np.log(1 / np.sqrt(R1*R2)) \
            / ((1 - np.sqrt(R1*R2)) * (np.sqrt(R1) + np.sqrt(R2)))
        self.Sf[0] = self.Sb[0] * R1
        self.Sf[1] = self.Sf[0] / np.sqrt(R1*R2)
        self.Sb[1] = self.Sf[1] * R2

    # local charge neutrality and equilibrium problems
    def gen_lcn_solver(self):
        """
        Generate solver for electrostatic potential distribution along
        vertical axis at equilibrium assuming local charge neutrality.
        """
        def f(psi):
            n = cc.n(psi, 0, self.yin['Nc'], self.yin['Ec'], self.Vt)
            p = cc.p(psi, 0, self.yin['Nv'], self.yin['Ev'], self.Vt)
            return self.yin['C_dop']-n+p
        def fdot(psi):
            ndot = cc.dn_dpsi(psi, 0, self.yin['Nc'], self.yin['Ec'], self.Vt)
            pdot = cc.dp_dpsi(psi, 0, self.yin['Nv'], self.yin['Ev'], self.Vt)
            return -ndot+pdot

        # initial guess with Boltzmann statistics
        ni = eq.intrinsic_concentration(self.yin['Nc'], self.yin['Nv'],
                                        self.yin['Ec'], self.yin['Ev'], self.Vt)
        Ei = eq.intrinsic_level(self.yin['Nc'], self.yin['Nv'], self.yin['Ec'],
                                self.yin['Ev'], self.Vt)
        Ef_i = eq.Ef_lcn_boltzmann(self.yin['C_dop'], ni, Ei, self.Vt)

        # Jacobian is a vector -> element-wise division
        la_fun = lambda A, b: b/A

        sol = newton.NewtonSolver(f, fdot, Ef_i, la_fun)
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
        assert self.ndim == 1
        sol = self.gen_lcn_solver()
        sol.solve(maxiter, fluct, omega)
        if sol.fluct[-1]>fluct:
            warnings.warn('LaserDiode1D.solve_lcn(): fluctuation '+
                         ('%e exceeds %e.' % (sol.fluct[-1], fluct)))

        self.yin['psi_lcn'] = sol.x.copy()
        self.yin['n0'] = cc.n(psi=self.yin['psi_lcn'], phi_n=0,
                              Nc=self.yin['Nc'], Ec=self.yin['Ec'],
                              Vt=self.Vt)
        self.yin['p0'] = cc.p(psi=self.yin['psi_lcn'], phi_p=0,
                              Nv=self.yin['Nv'], Ev=self.yin['Ev'],
                              Vt=self.Vt)

    def gen_equilibrium_solver(self):
        """
        Generate solver for electrostatic potential distribution along
        vertical axis at equilibrium.
        """
        if 'psi_lcn' not in self.yin:
            self.solve_lcn()
        h = self.xin[1:]-self.xin[:-1]
        w = self.xbn[1:]-self.xbn[:-1]

        def res(psi):
            n = cc.n(psi, 0, self.yin['Nc'], self.yin['Ec'], self.Vt)
            p = cc.p(psi, 0, self.yin['Nv'], self.yin['Ev'], self.Vt)
            r = eq.poisson_res(psi, n, p, h, w, self.yin['eps'], self.eps_0,
                               self.q, self.yin['C_dop'])
            return r

        def jac(psi):
            n = cc.n(psi, 0, self.yin['Nc'], self.yin['Ec'], self.Vt)
            ndot = cc.dn_dpsi(psi, 0, self.yin['Nc'], self.yin['Ec'], self.Vt)
            p = cc.p(psi, 0, self.yin['Nv'], self.yin['Ev'], self.Vt)
            pdot = cc.dp_dpsi(psi, 0, self.yin['Nv'], self.yin['Ev'], self.Vt)
            j = eq.poisson_jac(psi, n, ndot, p, pdot, h, w, self.yin['eps'],
                                self.eps_0, self.q, self.yin['C_dop'])
            return j

        la_fun = lambda A, b: solve_banded((1,1), A, b)
        psi_init = self.yin['psi_lcn']
        sol = newton.NewtonSolver(res, jac, psi_init, la_fun,
                                  inds=np.arange(1, len(psi_init)-1))
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
        assert self.ndim == 1
        sol = self.gen_equilibrium_solver()
        sol.solve(maxiter, fluct, omega)
        if sol.fluct[-1]>fluct:
            warnings.warn('LaserDiode1D.solve_equilibrium(): fluctuation '+
                         ('%e exceeds %e.' % (sol.fluct[-1], fluct)))
        self.yin['psi_bi'] = sol.x.copy()
        self.sol['psi'] = sol.x.copy()
        self.sol['phi_n'] = np.zeros_like(sol.x)
        self.sol['phi_p'] = np.zeros_like(sol.x)
        self._update_densities()

    def _update_densities(self):
        """
        Update electron and hole densities using currently stored potentials.
        """
        if self.ndim == 1:
            solutions = [self.sol]
        else:  # self.ndim == 2
            solutions = self.sol2d
        for sol in solutions:
            # aliases
            psi = sol['psi']
            phi_n = sol['phi_n']
            phi_p = sol['phi_p']
            Nc = self.yin['Nc']
            Nv = self.yin['Nv']
            Ec = self.yin['Ec']
            Ev = self.yin['Ev']
            Vt = self.Vt

            # densities
            sol['n'] = cc.n(psi, phi_n, Nc, Ec, Vt)
            sol['p'] = cc.p(psi, phi_p, Nv, Ev, Vt)
            # derivatives
            sol['dn_dpsi'] = cc.dn_dpsi(psi, phi_n, Nc, Ec, Vt)
            sol['dn_dphin'] = cc.dn_dphin(psi, phi_n, Nc, Ec, Vt)
            sol['dp_dpsi'] = cc.dp_dpsi(psi, phi_p, Nv, Ev, Vt)
            sol['dp_dphip'] = cc.dp_dphip(psi, phi_p, Nv, Ev, Vt)

    # waveguide problem
    def solve_waveguide(self, step=1e-7, n_modes=3, remove_layers=(0, 0)):
        """
        Calculate vertical mode profile. Finds `n_modes` solutions of the
        eigenvalue problem with the highest eigenvalues (effective
        indices) and picks the one with the highest optical confinement
        factor (active region overlap).

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
        # remove outer layers if needed
        i1, i2 = remove_layers
        if i2 == 0:
            epi = design.EpiDesign(self.epi[i1:])
        else:
            assert i2 > 0
            epi = design.EpiDesign(self.epi[i1:-i2])
        x0 = self.epi.boundaries()[i1]

        # generating refractive index profile
        x = x0 + np.arange(0, epi.get_thickness(), step)[1:]
        n = self.epi.calculate('n_refr', x)
        ar_ix = self._get_ar_ix(x, self.epi)

        # solving the eigenvalue problem
        if self.is_dimensionless:
            lam = self.lam * units.x
        else:
            lam = self.lam
        n_eff_values, modes = waveguide.solve_wg(x, n, lam, n_modes)
        # and picking one mode with the largest confinement factor (Gamma)
        gammas = np.zeros(n_modes)
        for i in range(n_modes):
            mode = modes[:, i]
            gammas[i] = (mode*step)[ar_ix].sum()  # modes are normalized
        i = np.argmax(gammas)
        mode = modes[:, i]

        # storing results
        self.n_eff = n_eff_values[i]
        self.gamma = gammas[i]
        self.wgm_fun = interp1d(x, mode, bounds_error=False,
                                fill_value=0)
        self.wgm_fun_dls = interp1d(x/units.x, mode*units.x,
                                    bounds_error=False, fill_value=0)
        self._calc_wg_mode()

        # return calculated mode profiles
        return x, modes, gammas

    def _calc_wg_mode(self):
        "Calculate normalized laser mode profile and store in `self.yin`."
        assert self.n_eff is not None
        if self.is_dimensionless:
            self.yin['wg_mode'] = self.wgm_fun_dls(self.xin)
        else:
            self.yin['wg_mode'] = self.wgm_fun(self.xin)

    # nonequilibrium drift-diffusion
    def apply_voltage(self, V):
        """
        Modify current solution so that boundary conditions at external
        voltage `V` are satisfied.
        """
        # scale voltage
        if self.is_dimensionless:
            V /= units.V

        # 1D: update `self.sol`, 2D: update `self.sol2d`
        if self.ndim == 1:
            solutions = [self.sol]
        else:
            solutions = self.sol2d

        # apply boundary conditions
        for sol in solutions:
            sol['psi'][0] = self.yin['psi_bi'][0] - V / 2
            sol['psi'][-1] = self.yin['psi_bi'][-1] + V / 2
            for phi in ('phi_n', 'phi_p'):
                sol[phi][0] = -V / 2
                sol[phi][-1] = V / 2
            # carrier densities at boundaries should not change

        # track solution convergence
        self.iterations = 0
        self.fluct = []  # fluctuation for every Newton iteration

    def transport_step(self, omega=0.1, discr='mSG'):
        """
        Perform a single Newton step for the transport problem.
        As a result,  solution vector `x` (concatenated `psi`, `phi_n` and
        `phi_p' vectors) is updated by `dx` with damping parameter `omega`,
        i.e. `x += omega * dx`.

        Parameters
        ----------
        omega : float
            Damping parameter.
        discr : str
            Current density discretiztion scheme.

        Returns
        -------
        fluct : float
            Solution fluctuation, i.e. ratio of `dx` and `x` L2 norms.

        """
        # solve the system
        assert self.ndim == 1
        m = self.mx - 2
        data, diags, rvec = self._transport_system(discr)
        J = sparse.spdiags(data=data, diags=diags, m=m*3, n=m*3,
                           format='csc')
        dx = sparse.linalg.spsolve(J, -rvec)

        # calculate and save fluctuation
        x = np.concatenate((self.sol['psi'][1:-1],
                            self.sol['phi_n'][1:-1],
                            self.sol['phi_p'][1:-1]))
        fluct = newton.l2_norm(dx) / newton.l2_norm(x)
        self.fluct.append(fluct)

        # update current solution (potentials and densities)
        self.sol['psi'][1:-1] += dx[:m] * omega
        self.sol['phi_n'][1:-1] += dx[m:2*m] * omega
        self.sol['phi_p'][1:-1] += dx[2*m:] * omega
        self._update_densities()

        return fluct

    def lasing_step(self, omega=0.1, omega_S=(1.0, 0.1), discr='mSG'):
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
        discr : str
            Current density discretiztion scheme.

        Returns
        -------
        fluct : number
            Solution fluctuation, i.e. ratio of `dx` and `x` L2 norms.

        """
        # residual vector and Jacobian for transport problem
        if self.ndim == 1:
            J, rvec = self._lasing_system_1D(discr)
        else:
            J, rvec = self._lasing_system_2D(discr)

        # solve the system
        dx = sparse.linalg.spsolve(J, -rvec)
        mx = self.nx - 2
        mz = self.mz
        if self.ndim == 1:
            x = np.concatenate((self.sol['psi'][1:-1],
                                self.sol['phi_n'][1:-1],
                                self.sol['phi_p'][1:-1],
                                np.array([self.sol['S']])))
        else:  # 2D
            x = np.zeros(3*mx*mz + 2*mz)
            for k in range(mz):
                xk = x[3*mx*k:3*mx*(k+1)]
                sol = self.sol2d[k]
                xk[:mx] = sol['psi'][1:-1]
                xk[mx:2*mx] = sol['phi_n'][1:-1]
                xk[2*mx:3*mx] = sol['phi_p'][1:-1]
            x[-2*mz:-mz] = self.Sb[:-1]
            x[-mz:] = self.Sf[1:]
        fluct = newton.l2_norm(dx) / newton.l2_norm(x)
        self.fluct.append(fluct)
        self.iterations += 1

        # update solution
        if self.ndim == 1:
            self.sol['psi'][1:-1] += dx[:mx]*omega
            self.sol['phi_n'][1:-1] += dx[mx:2*mx]*omega
            self.sol['phi_p'][1:-1] += dx[2*mx:3*mx]*omega
            self._update_densities()
            if dx[-1] > 0:
                self.sol['S'] += dx[-1] * omega_S[0]
            else:
                self.sol['S'] += dx[-1] * omega_S[1]
            self._update_Sb_Sf_1D()
        else:
            for k in range(self.mz):
                sol = self.sol2d[k]
                dxk = dx[3*mx*k:3*mx*(k+1)]
                sol['psi'][1:-1] += dxk[:mx] * omega
                sol['phi_n'][1:-1] += dxk[mx:2*mx] * omega
                sol['phi_p'][1:-1] += dxk[2*mx:3*mx] * omega
            self._update_densities()
            dx_S = dx[-2*mz:]
            ix = (dx_S[:mz] > 0)
            self.Sb[:-1][ix] += dx_S[:mz][ix] * omega_S[0]
            self.Sb[:-1][~ix] += dx_S[:mz][~ix] * omega_S[1]
            self.Sf[0] = self.Sb[0] * self.R1
            self.Sf[1:][ix] += dx_S[mz:][ix] * omega_S[0]
            self.Sf[1:][~ix] += dx_S[mz:][~ix] * omega_S[1]
            self.Sb[-1] = self.Sf[-1] * self.R2

        return fluct

    def _transport_system(self, discr):
        """
        Calculate Jacobian and residual for the transport problem.
        Uses solution currently stored in `self.sol`.
        Saves calculated values of current densities in `self.sol`.

        Parameters
        ----------
        discr : str
            Current density discretiztion scheme.

        Returns
        -------
        data : numpy.ndarray
            Jacobian diagonals.
        diags : list
            Jacobian diagonals indices.
        r : numpy.ndarray
            Vector of residuals.

        """
        # mesh parameters
        m = self.nx - 2  # number of inner nodes
                         # used in comments to show array shape
        h = self.xin[1:] - self.xin[:-1]  # mesh steps (m+1)
        w = self.xbn[1:] - self.xbn[:-1]  # volumes (m)

        # potentials, carrier densities and their derivatives at nodes
        psi = self.sol['psi']
        n = self.sol['n']
        p = self.sol['p']
        dn_dpsi = self.sol['dn_dpsi']
        dn_dphin = self.sol['dn_dphin']
        dp_dpsi = self.sol['dp_dpsi']
        dp_dphip = self.sol['dp_dphip']

        # Bernoulli function for current density calculation (m+1)
        B_plus = flux.bernoulli(+(psi[1:]-psi[:-1])/self.Vt)
        B_minus = flux.bernoulli(-(psi[1:]-psi[:-1])/self.Vt)
        Bdot_plus = flux.bernoulli_dot(+(psi[1:]-psi[:-1])/self.Vt)
        Bdot_minus = flux.bernoulli_dot(-(psi[1:]-psi[:-1])/self.Vt)

        # calculating current densities and their derivatives
        if discr == 'SG':  # Scharfetter-Gummel discretization
            jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2 = \
                self._jn_SG(B_plus, B_minus, Bdot_plus, Bdot_minus, h,
                            derivatives=True)
            jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2 = \
                self._jp_SG(B_plus, B_minus, Bdot_plus, Bdot_minus, h,
                            derivatives=True)

        elif discr == 'mSG':  # modified SG discretization
            jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2 = \
                self._jn_mSG(B_plus, B_minus, Bdot_plus, Bdot_minus, h,
                             derivatives=True)
            jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2 = \
                self._jp_mSG(B_plus, B_minus, Bdot_plus, Bdot_minus, h,
                             derivatives=True)

        else:
            raise Exception('Error: unknown current density '
                            + 'discretization scheme %s.' % discr)

        # store calculated current densities
        self.sol['jn'] = jn
        self.sol['jp'] = jp

        # spontaneous recombination rates (m)
        n0 = self.yin['n0'][1:-1]
        p0 = self.yin['p0'][1:-1]
        tau_n = self.yin['tau_n'][1:-1]
        tau_p = self.yin['tau_p'][1:-1]
        B_rad = self.yin['B'][1:-1]
        Cn = self.yin['Cn'][1:-1]
        Cp = self.yin['Cp'][1:-1]
        R_srh = rec.srh_R(n[1:-1], p[1:-1], n0, p0, tau_n, tau_p)
        R_rad = rec.rad_R(n[1:-1], p[1:-1], n0, p0, B_rad)
        R_aug = rec.auger_R(n[1:-1], p[1:-1], n0, p0, Cn, Cp)
        R = (R_srh + R_rad + R_aug)

        # recombination rates' derivatives
        dRsrh_dpsi = rec.srh_Rdot(n[1:-1], dn_dpsi[1:-1],
                                  p[1:-1], dp_dpsi[1:-1],
                                  n0, p0, tau_n, tau_p)
        dRrad_dpsi = rec.rad_Rdot(n[1:-1], dn_dpsi[1:-1],
                                  p[1:-1], dp_dpsi[1:-1], B_rad)
        dRaug_dpsi = rec.auger_Rdot(n[1:-1], dn_dpsi[1:-1],
                                    p[1:-1], dp_dpsi[1:-1],
                                    n0, p0, Cn, Cp)
        dR_dpsi = dRsrh_dpsi + dRrad_dpsi + dRaug_dpsi
        dRsrh_dphin = rec.srh_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                   n0, p0, tau_n, tau_p)
        dRrad_dphin = rec.rad_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                   B_rad)
        dRaug_dphin = rec.auger_Rdot(n[1:-1], dn_dphin[1:-1], p[1:-1], 0,
                                     n0, p0, Cn, Cp)
        dR_dphin = dRsrh_dphin + dRrad_dphin + dRaug_dphin
        dRsrh_dphip = rec.srh_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                   n0, p0, tau_n, tau_p)
        dRrad_dphip = rec.rad_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                   B_rad)
        dRaug_dphip = rec.auger_Rdot(n[1:-1], 0, p[1:-1], dp_dphip[1:-1],
                                     n0, p0, Cn, Cp)
        dR_dphip = dRsrh_dphip + dRrad_dphip + dRaug_dphip

        # residual of the system
        rvec = np.zeros(m*3)
        rvec[:m] = vrs.poisson_res(psi, n, p, h, w, self.yin['eps'],
                                   self.eps_0, self.q, self.yin['C_dop'])
        rvec[m:2*m] = self.q*R*w - (jn[1:]-jn[:-1])
        rvec[2*m:3*m] = -self.q*R*w - (jp[1:]-jp[:-1])

        # Jacobian
        # 1. Poisson's equation
        j11 = vrs.poisson_dF_dpsi(dn_dpsi, dp_dpsi, h, w, self.yin['eps'],
                                  self.eps_0, self.q)
        j12 = vrs.poisson_dF_dphin(dn_dphin, w, self.eps_0, self.q)
        j13 = vrs.poisson_dF_dphip(dp_dphip, w, self.eps_0, self.q)

        # 2. Electron current continuity equation
        j21 = vrs.jn_dF_dpsi(djn_dpsi1, djn_dpsi2, dR_dpsi, w, self.q, m)
        j22 = vrs.jn_dF_dphin(djn_dphin1, djn_dphin2, dR_dphin,
                              w, self.q, m)
        j23 = vrs.jn_dF_dphip(dR_dphip, w, self.q, m)

        # 3. Hole current continuity equation
        j31 = vrs.jp_dF_dpsi(djp_dpsi1, djp_dpsi2, dR_dpsi, w, self.q, m)
        j32 = vrs.jp_dF_dphin(dR_dphin, w, self.q, m)
        j33 = vrs.jp_dF_dphip(djp_dphip1, djp_dphip2, dR_dphip,
                              w, self.q, m)

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

        return data, diags, rvec

    def _lasing_system_1D(self, discr):
        m = self.nx - 2
        rvec = np.empty(m * 3 + 1)
        data, diags, rvec[:-1] = self._transport_system(discr)

        ixa = self.ar_ix
        inds = np.where(ixa)[0] - 1
        w_ar = (self.xbn[1:] - self.xbn[:-1])[inds-1]
        T = self.yin['wg_mode'][ixa]
        S = self.sol['S']

        # solution in the active region
        n = self.sol['n'][ixa]
        p = self.sol['p'][ixa]
        dn_dpsi = self.sol['dn_dpsi'][ixa]
        dn_dphin = self.sol['dn_dphin'][ixa]
        dp_dpsi = self.sol['dp_dpsi'][ixa]
        dp_dphip = self.sol['dp_dphip'][ixa]

        gain, dg_dpsi, dg_dphin, dg_dphip = \
            self._calculate_gain()

        # loss and net gain (Gamma*g - alpha)
        alpha_fca = self._calculate_fca()
        alpha = self.alpha_i + self.alpha_m + alpha_fca
        net_gain = np.sum(gain * w_ar * T) - alpha

        # stimulated emission rate and its derivatives
        R_st = self.vg * gain * w_ar * T * S
        dRst_dS = self.vg * gain * w_ar * T
        dRst_dpsi = self.vg * dg_dpsi * w_ar * T * S
        dRst_dphin = self.vg * dg_dphin * w_ar * T * S
        dRst_dphip = self.vg * dg_dphip * w_ar * T * S

        # radiative recombination rate and its derivatives
        n0 = self.yin['n0'][ixa]
        p0 = self.yin['p0'][ixa]
        B = self.yin['B'][ixa]
        R = rec.rad_R(n, p, n0, p0, B)
        dR_dpsi = rec.rad_Rdot(n, dn_dpsi, p, dp_dpsi, B)
        dR_dphin = rec.rad_Rdot(n, dn_dphin, p, 0, B)
        dR_dphip = rec.rad_Rdot(n, 0, p, dp_dphip, B)

        # update vector of residuals
        rvec[m + inds] += self.q * R_st
        rvec[2*m + inds] += -self.q * R_st
        rvec[-1] = (self.vg * net_gain * S
                    + self.beta_sp * np.sum(R * w_ar * T))

        # update Jacobian diagonals
        data[6, inds] += self.q * dRst_dpsi        # j21
        data[3, m+inds] += self.q * dRst_dphin     # j22
        data[1, 2*m+inds] += self.q * dRst_dphip   # j23
        data[9, inds] += -self.q * dRst_dpsi       # j31
        data[6, m+inds] += -self.q * dRst_dphin    # j32
        data[3, 2*m+inds] += -self.q * dRst_dphip  # j33
        J = sparse.spdiags(data, diags, m=3*m+1, n=3*m+1, format='lil')

        # fill rightmost column
        J[m + inds, -1] = self.q * dRst_dS
        J[2*m + inds, -1] = -self.q * dRst_dS

        # fill bottom row
        J[-1, inds] = (self.vg * dg_dpsi * S
                       + self.beta_sp * dR_dpsi) * w_ar * T
        J[-1, m+inds] = (self.vg * dg_dphin * S
                         + self.beta_sp * dR_dphin) * w_ar * T
        J[-1, 2*m+inds] = (self.vg * dg_dphip * S
                           + self.beta_sp * dR_dphip) * w_ar * T
        J[-1, -1] = self.vg * net_gain

        J = J.tocsc()
        return J, rvec

    def _lasing_system_2D(self, discr):
        # number of grid nodes
        mx = self.nx - 2          # x grid w/o boundaries
        mxa = np.sum(self.ar_ix)  # x grid, only active region
        mz = self.mz              # z grid

        # initialize Jacobian matrix
        # drift-diffusion system (F1, F2, F3)
        data = np.zeros((11, 3*mx*mz))   # J11-J33 (transport)
        diags = [2*mx, mx, 1, 0, -1, -mx+1, -mx, -mx-1, -2*mx+1, -2*mx, -2*mx-1]
        J24 = np.zeros(mxa*mz)           # J24 = -J34
        # photon density (Sb and Sf) rate equations (F4, F5)
        J4_13 = np.zeros((2, 3*mxa*mz))  # derivatives w.r.t. potentials
        J44 = np.zeros((2*mz, 2*mz))     # derivatives w.r.t. Sf/Sb

        # initialize vector of residuals
        rvec = np.zeros(3*mx*mz + 2*mz)

        # photon densities
        Sf_in = (self.Sf[1:] + self.Sf[:-1]) / 2
        Sb_in = (self.Sb[1:] + self.Sb[:-1]) / 2
        S = Sf_in + Sb_in

        # parameters shared by all slices
        w_ar = (self.xbn[1:] - self.xbn[:-1])[self.ar_ix[1:-1]]
        T = self.yin['wg_mode'][self.ar_ix]
        inds = np.where(self.ar_ix[1:-1])[0]
        n0 = self.yin['n0'][self.ar_ix]
        p0 = self.yin['p0'][self.ar_ix]
        B = self.yin['B'][self.ar_ix]

        # iterate over slices
        for k in range(mz):
            self.sol = self.sol2d[k]
            self.sol['S'] = S[k]

            # system without stimulated emission
            data[:, 3*mx*k:3*mx*(k+1)], _, rvec[3*mx*k:3*mx*(k+1)] = \
                self._transport_system(discr)

            # gain and stimulated emission rate
            gain, dg_dpsi, dg_dphin, dg_dphip = self._calculate_gain()
            R_st = self.vg * gain * w_ar * T * S[k]
            dRst_dSb = self.vg * gain * w_ar * T / 2
            dRst_dpsi = self.vg * dg_dpsi * w_ar * T * S[k]
            dRst_dphin = self.vg * dg_dphin * w_ar * T * S[k]
            dRst_dphip = self.vg * dg_dphip * w_ar * T * S[k]

            # loss and net gain
            alpha_fca = self._calculate_fca()
            alpha = self.alpha_i + alpha_fca
            net_gain = np.sum(gain * w_ar * T) - alpha

            # radiative recombination rate in the active region
            n = self.sol['n'][self.ar_ix]
            p = self.sol['p'][self.ar_ix]
            R = rec.rad_R(n, p, n0, p0, B)
            dR_dpsi = rec.rad_Rdot(n, self.sol['dn_dpsi'][self.ar_ix],
                                   p, self.sol['dp_dpsi'][self.ar_ix], B)
            dR_dphin = rec.rad_Rdot(n, self.sol['dn_dphin'][self.ar_ix],
                                    p, 0, B)
            dR_dphip = rec.rad_Rdot(n, 0,
                                    p, self.sol['dp_dphip'][self.ar_ix], B)
            R_modal = np.sum(R * w_ar * T)

            # update vector of residuals
            if k > 0:
                inds += 3*mx
            rvec[mx+inds] += self.q * R_st
            rvec[2*mx+inds] += -self.q * R_st
            rvec[3*mx*mz + k] = \
                (self.vg*(self.Sb[k+1] - self.Sb[k]) / self.dz
                 + self.vg * net_gain * Sb_in[k]
                 + self.beta_sp * R_modal/2)
            rvec[3*mx*mz + mz + k] = \
                (-self.vg*(self.Sf[k+1] - self.Sf[k]) / self.dz
                 + self.vg * net_gain * Sf_in[k]
                 + self.beta_sp * R_modal/2)

            # update Jacobian
            # drift-diffusion system
            data[6, inds] += self.q * dRst_dpsi         # j21
            data[3, mx+inds] += self.q * dRst_dphin     # j22
            data[1, 2*mx+inds] += self.q * dRst_dphip   # j23
            data[9, inds] += -self.q * dRst_dpsi        # j31
            data[6, mx+inds] += -self.q * dRst_dphin    # j32
            data[3, 2*mx+inds] += -self.q * dRst_dphip  # j33
            J24[mxa*k:mxa*(k+1)] = self.q * dRst_dSb

            # photon density rate equations
            J413_k = J4_13[:, 3*mxa*k:3*mxa*(k+1)]
            J413_k[0, :mxa] = (self.beta_sp * dR_dpsi * w_ar * T
                               + self.vg * dg_dpsi * w_ar * T * Sb_in[k])
            J413_k[1, :mxa] = (self.beta_sp * dR_dpsi * w_ar * T
                               + self.vg * dg_dpsi * w_ar * T * Sf_in[k])
            J413_k[0, mxa:2*mxa] = \
                (self.beta_sp * dR_dphin * w_ar * T
                 + self.vg * dg_dphin * w_ar * T * Sb_in[k])
            J413_k[1, mxa:2*mxa] = \
                (self.beta_sp * dR_dphin * w_ar * T
                 + self.vg * dg_dphin * w_ar * T * Sf_in[k])
            J413_k[0, 2*mxa:3*mxa] = \
                (self.beta_sp * dR_dphip * w_ar * T
                 + self.vg * dg_dphip * w_ar * T * Sb_in[k])
            J413_k[1, 2*mxa:3*mxa] = \
                (self.beta_sp * dR_dphip * w_ar * T
                 + self.vg * dg_dphip * w_ar * T * Sf_in[k])

            J44[k, k] = self.vg * (-1/self.dz + net_gain/2)
            if k < mz - 1:
                J44[k, k+1] = self.vg * (1/self.dz + net_gain/2)
            else:
                J44[mz-1, -1] = self.R2 * self.vg * (1/self.dz + net_gain/2)
            J44[mz+k, mz+k] = self.vg * (-1 / self.dz + net_gain/2)
            if k > 0:
                J44[mz+k, mz+k-1] = self.vg * (1/self.dz + net_gain/2)
            else:
                J44[mz, 0] = self.R1 * self.vg * (1/self.dz + net_gain/2)

        # assemble Jacobian
        # drift-diffusion system diagonals
        J = sparse.spdiags(data, diags, format='lil',
                           m=3*mx*mz + 2*mz, n=3*mx*mz + 2*mz)

        # rightmost colums
        inds = np.where(self.ar_ix[1:-1])[0]
        for k in range(mz):
            q_Rstdot = J24[mxa*k:mxa*(k+1)]  # q*dRst/dSb
            for i in [0, 1, mz-1, mz]:
                if k == 0 and i == 0:
                    b = 1 + self.R1
                elif k == mz-1 and i == mz:
                    b = 1 + self.R2
                else:
                    b = 1
                J[3*mx*k + mx + inds, 3*mx*mz + k+i] = q_Rstdot * b
                J[3*mx*k + 2*mx + inds, 3*mx*mz + k+i] = -q_Rstdot * b

        # bottom rows
        for k in range(mz):
            J413_k = J4_13[:, 3*mxa*k:3*mxa*(k+1)]
            for i in range(3):
                J[3*mx*mz + k, 3*mx*k + i*mx + inds] = \
                    J413_k[0, mxa*i:mxa*(i+1)]
                J[3*mx*mz + mz + k, 3*mx*k + i*mx + inds] = \
                    J413_k[1, mxa*i:mxa*(i+1)]

        # bottom right corner
        J[3*mx*mz:, 3*mx*mz:] = J44

        self.sol = dict()
        J = J.tocsc()
        return J, rvec

    def _jn_SG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h,
               derivatives=True):
        "Electron current density, Scharfetter-Gummel scheme."
        psi = self.sol['psi']
        phi_n = self.sol['phi_n']

        # electron densities at finite volume boundaries
        n1 = cc.n(psi[:-1], phi_n[:-1], self.ybn['Nc'], self.ybn['Ec'],
                  self.Vt)
        n2 = cc.n(psi[1:], phi_n[1:], self.ybn['Nc'], self.ybn['Ec'],
                  self.Vt)

        # forward (2-2) and backward (1-1) derivatives
        # w.r.t. potentials at volume boundaries
        dn1_dpsi1 = cc.dn_dpsi(psi[:-1], phi_n[:-1], self.ybn['Nc'],
                               self.ybn['Ec'], self.Vt)
        dn2_dpsi2 = cc.dn_dpsi(psi[1:], phi_n[1:], self.ybn['Nc'],
                               self.ybn['Ec'], self.Vt)
        dn1_dphin1 = cc.dn_dphin(psi[:-1], phi_n[:-1], self.ybn['Nc'],
                                 self.ybn['Ec'], self.Vt)
        dn2_dphin2 = cc.dn_dphin(psi[1:], phi_n[1:], self.ybn['Nc'],
                                 self.ybn['Ec'], self.Vt)

        # electron current density and its derivatives
        jn = flux.SG_jn(n1, n2, B_plus, B_minus, h,
                        self.Vt, self.q, self.ybn['mu_n'])
        if not derivatives:
            return jn
        djn_dpsi1 = flux.SG_djn_dpsi1(n1, n2, dn1_dpsi1, B_minus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_n'])
        djn_dpsi2 = flux.SG_djn_dpsi2(n1, n2, dn2_dpsi2, B_plus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_n'])
        djn_dphin1 = flux.SG_djn_dphin1(dn1_dphin1, B_minus, h, self.Vt,
                                        self.q, self.ybn['mu_n'])
        djn_dphin2 = flux.SG_djn_dphin2(dn2_dphin2, B_plus, h, self.Vt,
                                        self.q, self.ybn['mu_n'])

        return jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2

    def _jp_SG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h,
               derivatives=True):
        "Hole current density, Scharfetter-Gummel scheme."
        psi = self.sol['psi']
        phi_p = self.sol['phi_p']

        # hole densities at finite volume boundaries
        p1 = cc.p(psi[:-1], phi_p[:-1], self.ybn['Nv'], self.ybn['Ev'],
                  self.Vt)
        p2 = cc.p(psi[1:], phi_p[1:], self.ybn['Nv'], self.ybn['Ev'],
                  self.Vt)

        # forward (2-2) and backward (1-1) derivatives
        # w.r.t. to potentials at volume boundaries
        dp1_dpsi1 = cc.dp_dpsi(psi[:-1], phi_p[:-1], self.ybn['Nv'],
                               self.ybn['Ev'], self.Vt)
        dp2_dpsi2 = cc.dp_dpsi(psi[1:], phi_p[1:], self.ybn['Nv'],
                               self.ybn['Ev'], self.Vt)
        dp1_dphip1 = cc.dp_dphip(psi[:-1], phi_p[:-1], self.ybn['Nv'],
                                 self.ybn['Ev'], self.Vt)
        dp2_dphip2 = cc.dp_dphip(psi[1:], phi_p[1:], self.ybn['Nv'],
                                 self.ybn['Ev'], self.Vt)

        # hole current density and its derivatives
        jp = flux.SG_jp(p1, p2, B_plus, B_minus, h,
                        self.Vt, self.q, self.ybn['mu_p'])
        if not derivatives:
            return jp
        djp_dpsi1 = flux.SG_djp_dpsi1(p1, p2, dp1_dpsi1, B_plus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_p'])
        djp_dpsi2 = flux.SG_djp_dpsi2(p1, p2, dp2_dpsi2, B_minus,
                                      Bdot_plus, Bdot_minus, h, self.Vt,
                                      self.q, self.ybn['mu_p'])
        djp_dphip1 = flux.SG_djp_dphip1(dp1_dphip1, B_plus, h, self.Vt,
                                        self.q, self.ybn['mu_p'])
        djp_dphip2 = flux.SG_djp_dphip2(dp2_dphip2, B_minus, h, self.Vt,
                                        self.q, self.ybn['mu_p'])

        return jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2

    def _jn_mSG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h,
                derivatives=True):
        "Electron current density, modified Scharfetter-Gummel scheme."
        psi = self.sol['psi']
        phi_n = self.sol['phi_n']

        # n = Nc * F(nu_n)
        F = sdf.fermi_approx
        nu_n1 = (psi[:-1]-phi_n[:-1]-self.ybn['Ec']) / self.Vt  # (m+1)
        nu_n2 = (psi[1:]-phi_n[1:]-self.ybn['Ec']) / self.Vt
        exp_nu_n1 = np.exp(nu_n1)
        exp_nu_n2 = np.exp(nu_n2)

        # electron current density
        gn = flux.g(nu_n1, nu_n2, F)
        jn_SG = flux.oSG_jn(exp_nu_n1, exp_nu_n2, B_plus, B_minus,
                            h, self.ybn['Nc'], self.Vt, self.q,
                            self.ybn['mu_n'])
        jn = jn_SG * gn
        if not derivatives:
            return jn

        # electron current density derivatives
        Fdot = sdf.fermi_dot_approx
        gdot_n1 = flux.gdot(gn, nu_n1, F, Fdot) / self.Vt
        gdot_n2 = flux.gdot(gn, nu_n2, F, Fdot) / self.Vt
        djn_dpsi1_SG = flux.oSG_djn_dpsi1(exp_nu_n1, exp_nu_n2,
                                          B_minus, Bdot_plus, Bdot_minus,
                                          h, self.ybn['Nc'], self.q,
                                          self.ybn['mu_n'])
        djn_dpsi1 = flux.mSG_jdot(jn_SG, djn_dpsi1_SG, gn, gdot_n1)
        djn_dpsi2_SG = flux.oSG_djn_dpsi2(exp_nu_n1, exp_nu_n2,
                                          B_plus, Bdot_plus, Bdot_minus,
                                          h, self.ybn['Nc'], self.q,
                                          self.ybn['mu_n'])
        djn_dpsi2 = flux.mSG_jdot(jn_SG, djn_dpsi2_SG, gn, gdot_n2)
        djn_dphin1_SG = flux.oSG_djn_dphin1(exp_nu_n1, B_minus, h,
                                            self.ybn['Nc'], self.q,
                                            self.ybn['mu_n'])
        djn_dphin1 = flux.mSG_jdot(jn_SG, djn_dphin1_SG, gn, -gdot_n1)
        djn_dphin2_SG = flux.oSG_djn_dphin2(exp_nu_n2, B_plus, h,
                                            self.ybn['Nc'], self.q,
                                            self.ybn['mu_n'])
        djn_dphin2 = flux.mSG_jdot(jn_SG, djn_dphin2_SG, gn, -gdot_n2)

        return jn, djn_dpsi1, djn_dpsi2, djn_dphin1, djn_dphin2

    def _jp_mSG(self, B_plus, B_minus, Bdot_plus, Bdot_minus, h,
                derivatives=True):
        "Hole current density, modified Scharfetter-Gummel scheme."
        psi = self.sol['psi']
        phi_p = self.sol['phi_p']

        #  p = Nv * F(nu_p)
        F = sdf.fermi_approx
        nu_p1 = (-psi[:-1]+phi_p[:-1]+self.ybn['Ev']) / self.Vt
        nu_p2 = (-psi[1:]+phi_p[1:]+self.ybn['Ev']) / self.Vt
        exp_nu_p1 = np.exp(nu_p1)
        exp_nu_p2 = np.exp(nu_p2)

        # hole current density
        gp = flux.g(nu_p1, nu_p2, F)
        jp_SG = flux.oSG_jp(exp_nu_p1, exp_nu_p2, B_plus, B_minus,
                            h, self.ybn['Nv'], self.Vt, self.q,
                            self.ybn['mu_p'])
        jp = jp_SG * gp
        if not derivatives:
            return jp

        # hole current density derivatives
        Fdot = sdf.fermi_dot_approx
        gdot_p1 = flux.gdot(gp, nu_p1, F, Fdot) / self.Vt
        gdot_p2 = flux.gdot(gp, nu_p2, F, Fdot) / self.Vt
        djp_dpsi1_SG = flux.oSG_djp_dpsi1(exp_nu_p1, exp_nu_p2,
                                          B_plus, Bdot_plus, Bdot_minus,
                                          h, self.ybn['Nv'], self.q,
                                          self.ybn['mu_p'])
        djp_dpsi1 = flux.mSG_jdot(jp_SG, djp_dpsi1_SG, gp, -gdot_p1)
        djp_dpsi2_SG = flux.oSG_djp_dpsi2(exp_nu_p1, exp_nu_p2,
                                          B_minus, Bdot_plus, Bdot_minus,
                                          h, self.ybn['Nv'], self.q,
                                          self.ybn['mu_p'])
        djp_dpsi2 = flux.mSG_jdot(jp_SG, djp_dpsi2_SG, gp, -gdot_p2)
        djp_dphip1_SG = flux.oSG_djp_dphip1(exp_nu_p1, B_plus, h,
                                            self.ybn['Nv'], self.q,
                                            self.ybn['mu_p'])
        djp_dphip1 = flux.mSG_jdot(jp_SG, djp_dphip1_SG, gp, gdot_p1)
        djp_dphip2_SG = flux.oSG_djp_dphip2(exp_nu_p2, B_minus, h,
                                            self.ybn['Nv'], self.q,
                                            self.ybn['mu_p'])
        djp_dphip2 = flux.mSG_jdot(jp_SG, djp_dphip2_SG, gp, gdot_p2)

        return jp, djp_dpsi1, djp_dpsi2, djp_dphip1, djp_dphip2

    def _calculate_gain(self):
        "Calculate material gain and its derivatives at active region nodes."
        # aliases
        n = self.sol['n'][self.ar_ix]
        p = self.sol['p'][self.ar_ix]
        g0 = self.yin['g0']
        N_tr = self.yin['N_tr']

        # g = g0 * ln(min(n,p) / N_tr)
        N = p.copy()
        ixn = (n < p)
        ixp = ~ixn
        N[ixn] = n[ixn]
        gain = g0 * np.log(N / N_tr)
        ix_abs = (gain < 0)  # ignore absorption
        if ix_abs.all():
            m = self.ar_ix.sum()
            return [np.zeros(m)] * 4  # g, dg/dpsi, dg/dphin, dg/dphip
        gain[ix_abs] = 0.0
        ixn = np.logical_and(ixn, ~ix_abs)
        ixp = np.logical_and(ixp, ~ix_abs)

        # calculate gain derivatives
        dn_dpsi = self.sol['dn_dpsi'][self.ar_ix]
        dn_dphin = self.sol['dn_dphin'][self.ar_ix]
        dp_dpsi = self.sol['dp_dpsi'][self.ar_ix]
        dp_dphip = self.sol['dp_dphip'][self.ar_ix]
        dg_dpsi = np.zeros_like(gain)
        dg_dpsi[ixn] = g0[ixn] * dn_dpsi[ixn] / n[ixn]
        dg_dpsi[ixp] = g0[ixp] * dp_dpsi[ixp] / p[ixp]
        dg_dphin = np.zeros_like(gain)
        dg_dphin[ixn] = g0[ixn] * dn_dphin[ixn] / n[ixn]
        dg_dphip = np.zeros_like(gain)
        dg_dphip[ixp] = g0[ixp] * dp_dphip[ixp] / p[ixp]

        return gain, dg_dpsi, dg_dphin, dg_dphip

    def _calculate_fca(self, n=None, p=None):
        "Calculate free-carrier absorption."
        T = self.yin['wg_mode'][1:-1]
        w = self.xbn[1:] - self.xbn[:-1]
        fca_e = self.yin['fca_e'][1:-1]
        fca_h = self.yin['fca_h'][1:-1]
        if n is None:
            n = self.sol['n'][1:-1]
        else:
            n = n[1:-1]
        if p is None:
            p = self.sol['p'][1:-1]
        else:
            p = p[1:-1]
        arr = T * w * (n * fca_e + p * fca_h)
        return np.sum(arr)

    # extract useful data from simulation results
    def get_J(self, discr='mSG'):
        "Get current density through device (A/cm2)."
        # 1D -- return float, 2D -- return numpy.ndarray
        if self.ndim == 1:
            J = (self.sol['jn'] + self.sol['jp']).mean()
        else:  # calculate J for every slice
            J = np.zeros(self.mz)
            for k in range(self.mz):
                sol = self.sol2d[k]
                J[k] = (sol['jn'] + sol['jp']).mean()
            self.sol = dict()

        if self.is_dimensionless:
            J *= units.j  # convert to A/cm2
        return J

    def get_I(self, discr='mSG'):
        "Get current through device (A)."
        if self.is_dimensionless:
            w = self.w * units.x
            dz = self.dz * units.x
        else:
            w = self.w
            dz = self.dz
        return self.get_J(discr) * w * dz

    def get_P(self):
        "Get output power from each facet (W)."
        P = np.zeros(2)
        E_ph = self.photon_energy
        P[0] = E_ph * self.vg * self.Sb[0] * (1 - self.R1) * self.w
        P[1] = E_ph * self.vg * self.Sf[-1] * (1 - self.R2) * self.w
        if self.is_dimensionless:
            P *= units.P
        return P

    def get_n_active(self):
        "Get average electron and hole density in the active region."
        def avg_density(n, dx, D):
            # integrate using trapezoid formula
            return np.sum((n[:-1] + n[1:]) / 2 * dx) / D

        ix = self.ar_ix
        x = self.xin[ix]
        dx = x[1:] - x[:-1]
        D = x[-1] - x[0]  # active region thickness
        if self.ndim == 1:
            n_avg = avg_density(self.sol['n'][ix], dx, D)
            p_avg = avg_density(self.sol['p'][ix], dx, D)
        else:
            n_avg = np.zeros(self.mz)
            p_avg = np.zeros(self.mz)
            for k in range(self.mz):
                n_avg[k] = avg_density(self.sol2d[k]['n'][ix], dx, D)
                p_avg[k] = avg_density(self.sol2d[k]['p'][ix], dx, D)
        if self.is_dimensionless:
            n_avg *= units.n
            p_avg *= units.n
        return n_avg, p_avg

    def get_FCA(self):
        "Get free-carrier absorption (cm-1)."
        if self.ndim == 1:
            alpha_fca = self._calculate_fca()
        else:
            alpha_fca = np.zeros(self.mz)
            for k in range(self.mz):
                self.sol = self.sol2d[k]
                alpha_fca[k] = self._calculate_fca()
            self.sol = dict()
        if self.is_dimensionless:
            alpha_fca *= (1 / units.x)
        return alpha_fca

    def get_Jsp(self, which='all'):
        """
        Get current density through device corresponding to spontaneous
        recombination (A/cm2).

        Parameters
        ----------
        which : str
            Recombination mechanism. One of:
                * `'SRH'`,
                * `'radiative'`,
                * `'Auger'`,
                * `'all'`.

        """
        # functions for calculating recombination rate
        def R_srh(n, p):
            return rec.srh_R(n, p, n0=self.yin['n0'][1:-1],
                             p0=self.yin['p0'][1:-1],
                             tau_n=self.yin['tau_n'][1:-1],
                             tau_p=self.yin['tau_p'][1:-1])

        def R_rad(n, p):
            return rec.rad_R(n, p, n0=self.yin['n0'][1:-1],
                             p0=self.yin['p0'][1:-1], B=self.yin['B'][1:-1])

        def R_aug(n, p):
            return rec.auger_R(n, p, n0=self.yin['n0'][1:-1],
                               p0=self.yin['p0'][1:-1],
                               Cn=self.yin['Cn'][1:-1],
                               Cp=self.yin['Cp'][1:-1])

        def R(n, p):
            return R_srh(n, p) + R_rad(n, p) + R_aug(n, p)

        # choose function
        if which == 'all':
            R_fun = R
        elif which.lower() == 'srh':
            R_fun = R_srh
        elif which.lower() in ('rad', 'radiative'):
            R_fun = R_rad
        elif which.lower() in ('aug', 'auger'):
            R_fun = R_aug
        else:
            raise Exception('Error: unknown recombination mechanism %s.'
                            % which)

        # calculate spontaneous recombination current density
        J_sp = self._J_sp(R_fun)
        if self.is_dimensionless:
            J_sp *= units.j
        return J_sp

    def get_Isp(self, which='all'):
        """
        Get current through device corresponding to spontaneous
        recombination (A).

        Parameters
        ----------
        which : str
            Recombination mechanism. One of:
                * `'SRH'`,
                * `'radiative'`,
                * `'Auger'`,
                * `'all'`.

        """
        J_sp = self.get_Jsp(which)
        W = self.w
        dz = self.dz
        if self.is_dimensionless:
            W *= units.x
            dz *= units.x
        return J_sp * W * dz

    def _J_sp(self, R_fun):
        "J_sp for arbitrary recombination mechanism."
        w = self.xbn[1:] - self.xbn[:-1]
        if self.ndim == 1:
            n = self.sol['n'][1:-1]
            p = self.sol['p'][1:-1]
            J = self.q * np.sum(R_fun(n, p) * w)
        else:
            J = np.zeros(self.mz)
            for k in range(self.mz):
                n = self.sol2d[k]['n'][1:-1]
                p = self.sol2d[k]['p'][1:-1]
                J[k] = self.q * np.sum(R_fun(n, p) * w)
        return J

    # export simulation results
    def export_results(self, folder=None, vp=2, delimiter=',', x_to_um=True):
        """
        Export current solution in the band diagram form to one or several
        csv files (depending on the number of dimensions).

        Parameters
        ----------
        folder : str or None
            Directory to save the file.
        vp : int
            Voltage precision, i.e. number of digits after `.`, to use in
            the file name. The default is `2`.
        delimiter : str
            Column separator to use. The default is `','`.
        x_to_um : bool
            Whether to convert `x` from centimeters to micrometers.

        """
        # pick directory for export
        if folder is not None and isinstance(folder, str):
            if not os.path.isdir(folder):
                os.mkdir(folder)
        else:
            folder = ''

        # make file name without '1D' or '2D' prefix
        if self.ndim == 1:
            sol = self.sol
        else:
            sol = self.sol2d[0]
        voltage = sol['phi_n'][-1] - sol['phi_n'][0]
        if self.is_dimensionless:
            voltage *= units.V
        assert isinstance(vp, int) and vp >= 0
        s = '{:.' + str(vp) + 'f}'  # format string
        fname = s.format(voltage) + 'V.csv'

        # export results
        if self.ndim == 1:  # 1D -> single csv file
            file = os.path.join(folder, '1D_' + fname)
            self._export_solution(file, delimiter, x_to_um)

        else:  # 2D -> one csv file per slice
            for k in range(self.mz):
                self.sol = self.sol2d[k]
                file = os.path.join(folder, f'2D_{k+1}_'+fname)
                self._export_solution(file, delimiter, x_to_um)
            self.sol = dict()

    def _export_solution(self, file, delimiter, x_to_um):
        # create arrays for export
        x = self.xin.copy()
        Ev = self.yin['Ev'] - self.sol['psi']
        Ec = self.yin['Ec'] - self.sol['psi']
        fn = -self.sol['phi_n']
        fp = -self.sol['phi_p']
        n = self.sol['n'].copy()
        p = self.sol['p'].copy()
        psi = self.sol['psi'].copy()
        F_spl = InterpolatedUnivariateSpline(self.xin, psi).derivative()
        F = -F_spl(self.xin)
        jn = np.zeros_like(self.xin)
        jn[1:-1] = (self.sol['jn'][1:] + self.sol['jn'][:-1]) / 2
        jp = np.zeros_like(self.xin)
        jp[1:-1] = (self.sol['jp'][1:] + self.sol['jp'][:-1]) / 2

        # convert dimensionless values
        if self.is_dimensionless:
            x *= units.x
            Ev *= units.E
            Ec *= units.E
            fn *= units.E
            fp *= units.E
            n *= units.n
            p *= units.n
            psi *= units.V
            F *= units.V / units.x
            jn *= units.j
            jp *= units.j

        # convert x to micrometers
        if x_to_um:
            x *= 1e4

        # write to file
        with open(file, 'w') as f:
            # header
            f.write(delimiter.join(('x', 'Ev', 'Ec', 'fn', 'fp', 'n', 'p',
                                    'psi', 'F', 'jn', 'jp')))
            # values
            for i in range(self.nx):
                f.write('\n')
                vals = map('{:e}'.format, (x[i], Ev[i], Ec[i], fn[i],
                                          fp[i], n[i], p[i], psi[i], F[i],
                                          jn[i], jp[i]))
                line = delimiter.join(vals)
                f.write(line)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sample_design import epi

    plt.rc('lines', linewidth=0.7)
    plt.rc('figure.subplot', left=0.15, right=0.85)

    print('Creating an instance of LaserDiode1D...', end=' ')
    ld = LaserDiode(epi=epi, L=3000e-4, w=100e-4, R1=0.95, R2=0.05,
                    lam=0.87e-4, ng=3.9, alpha_i=0.5, beta_sp=1e-4)
    print('Complete.')

    # 1. nonuniform mesh
    print('Generating a nonuniform mesh...', end=' ')
    ld.gen_nonuniform_mesh(param='Eg', step_min=1e-7, step_max=20e-7,
                           sigma=1e-5, y_ext=[0.3, 0.3])
    print('Complete.')
    x = ld.xin*1e4
    plt.figure('Flat bands')
    plt.plot(x, ld.yin['Ec'], color='b')
    plt.plot(x, ld.yin['Ev'], color='b')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Energy (eV)', color='b')
    plt.twinx()
    plt.plot(x, ld.yin['n_refr'], ls=':', color='g', marker='x',
             ms=3)
    plt.ylabel('Refractive index', color='g')

    # 2. equilibrium
    ld.make_dimensionless()
    print('Calculating built-in potential assuming local charge neutrality...',
          end=' ')
    ld.solve_lcn()
    print('Complete.')
    print('Calculating built-in potential by solving Poisson\'s equation...',
          end=' ')
    ld.solve_equilibrium()
    print('Complete.')
    ld.original_units()
    psi = ld.yin['psi_bi']
    plt.figure('Equilibrium')
    plt.plot(x, ld.yin['Ec']-psi, color='b')
    plt.plot(x, ld.yin['Ev']-psi, color='b')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Energy (eV)', color='b')
    plt.twinx()
    plt.plot(x, ld.sol['n'], color='g')
    plt.plot(x, ld.sol['p'], color='g', ls='--')
    plt.yscale('log')
    plt.ylabel('Carrier densities (cm$^{-3}$)', color='g')

    # 3. waveguide
    print('Calculating vertical mode profile...', end=' ')
    ld.solve_waveguide(remove_layers=[1, 1])
    print('Complete.')
    plt.figure('Waveguide')
    plt.plot(x, ld.yin['wg_mode']/1e4, color='b')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Mode intensity', color='b')
    plt.twinx()
    plt.plot(x, ld.yin['n_refr'], color='g')
    plt.ylabel('Refractive index', color='g')

    # 4. forward bias
    nsteps = 500
    ld.make_dimensionless()
    print('Solving drift-diffusion system at small forward bias...',
          end=' ')
    ld.apply_voltage(0.1)
    for _ in range(nsteps):
        ld.lasing_step(0.1, [1.0, 0.1], 'mSG')
    print('Complete.')
    ld.original_units()
    plt.figure('Small forward bias')
    plt.plot(x, ld.yin['Ec']-ld.sol['psi'], color='b')
    plt.plot(x, ld.yin['Ev']-ld.sol['psi'], color='b')
    plt.plot(x, -ld.sol['phi_n'], color='g', ls=':')
    plt.plot(x, -ld.sol['phi_p'], color='r', ls=':')
    plt.xlabel(r'$x$ ($\mu$m)')
    plt.ylabel('Energy (eV)')

    plt.figure('Convergence')
    plt.plot(ld.fluct)
    plt.xlabel('Iteration number')
    plt.ylabel('Fluctuation')
    plt.yscale('log')

    # 5. test 2D model
    # 5.1. reach threshold
    ld.make_dimensionless()
    V, dV = 0.1, 0.1
    print('Increasing bias until reaching threshold...')
    while ld.sol['S'] < 1:
        V += dV
        ld.apply_voltage(V)
        fluct = 1
        while fluct > 1e-8:
            fluct = ld.lasing_step(0.1, [1.0, 0.1], 'mSG')
        print(V, ld.iterations, ld.sol['S'])
    ld.original_units()
    plt.figure('Photon density distribution')
    plt.plot(ld.zbn*1e4, ld.Sb, 'bx')
    plt.plot(ld.zbn*1e4, ld.Sf, 'rx')
    plt.xlabel(r'$z$ ($\mu$m)')
    plt.ylabel('$S$ (cm$^{-2}$)')

    I_1D = ld.get_I()
    P = ld.get_P()
    n_1D, p_1D = ld.get_n_active()
    print('I_1D =', I_1D)
    print(f'P_1D = {P[0]:.3f} + {P[1]:.3f} W')

    # 5.2. move from 1D to 2D model
    print('2D problem')
    ld.make_dimensionless()
    ld.to_2D(10)
    for i in range(30):
        fluct = ld.lasing_step(1.0, [1.0, 1.0], 'mSG')
        print(i, fluct, ld.Sb[0], ld.Sf[-1])
    ld.original_units()
    plt.plot(ld.zbn*1e4, ld.Sb, 'b.--')
    plt.plot(ld.zbn*1e4, ld.Sf, 'r.--')

    I_2D = ld.get_I()
    P = ld.get_P()
    print('I_2D = ', I_2D.sum())
    print(f'P_2D = {P[0]:.3f} + {P[1]:.3f} W')

    n_2D, p_2D = ld.get_n_active()
    plt.figure('Carrier density distribution')
    plt.plot(ld.zin*1e4, n_2D, 'b.-')
    plt.plot([0, ld.L*1e4], [n_1D]*2, 'b:')
    plt.plot(ld.zin*1e4, p_2D, 'r.-')
    plt.plot([0, ld.L*1e4], [p_1D]*2, 'r:')
    plt.xlabel(r'$z$, $\mu$m')
    plt.ylabel(r'$n_{a}$, $p_{a}$ (cm$^{-3}$)')
