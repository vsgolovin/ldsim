# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:10:42 2022

@author: Leonid_PC
"""

"""
Temparature dependensies of model parameters
Parameters was taken from:
    NSM Archive - Physical Properties of Semiconductors (http://matprop.ru/)
    Vurgaftman, I. et al - Band parameters for III-V compound semiconductors and their alloys
"""

import numpy as np
from ldsim import constants as const
from ldsim.units import dct as units
from ldsim.preprocessing.design import Layer


#%% Lists of parameters
material_params = ('Eg', 'm_e', 'm_h', 'mu_n', 'mu_p', 'tau_n','tau_p', 
                   'B', 'Cn', 'Cp', 'fca_e', 'fca_h')
params_active = ('g0', 'N_tr')
temperature_params = ('Eg_A', 'Eg_B', 'mu_n_P', 'mu_p_P', 'B_P', 'C_Ea',
                      'fca_e_P', 'fca_h_P', 'd_g0', 'd_Ntr', 'Rt')


#%% Some dependencies with bowing parameters
def _Eg_AlGaAs(x):
    return 1.519 + 1.155*x + 0.37*x**2  # for x < 0.41

def _mu_n_AlGaAs(x):
    return 8000 - 22000*x + 10000*x**2  # for x < 0.45

def _mu_p_AlGaAs(x):
    return 370 - 970*x + 740*x**2  # for x < 0.45

def _n_refr_AlGaAs(x, l=1e-4):
    """
    Function for calculating refractive index of Al(x)Ga(1-x)As depending on x
    and wavelength
    Default wavelength 1 micron
    Source: https://www.batop.de/information/n_AlGaAs.html
    """
    l *= 1e-2 # to meters

    a0 = 6.3 + 19*x  # fitting coefficients
    b0 = 9.4 - 10.2*x

    e0 = _Eg_AlGaAs(x)
    de = 0.34 - 0.04*x  # spin-orbit splitting energy

    y = const.h*const.c / (const.q*l*e0*1e2)
    y0 = const.h*const.c / (const.q*l*(e0+de)*1e2)
    f = (2 - np.sqrt(1-y) - np.sqrt(1+y))/y**2
    f0 = (2 - np.sqrt(1-y0) - np.sqrt(1+y0))/(y0**2)
    return np.sqrt(a0*(f + (f0/2)*((e0 / (e0+de))**(3/2))) + b0)


#%% Main class for updating material parameters

class material_AlGaAs:

    def __init__(self, T_HS, AC_params, BC_params, ar_params, dT_coeffs):
        """
        Initialize all material params temeperature coefficients for 
        Al(x)Ga(1-x)As
        
        AC_params = parameters for AlAs
        BC_params - parameters for GaAs
        dT_coeffs - temperature coefficients
        """
        self.T_HS = T_HS  # heatsink temperature
        self.Vt0 = const.kb*T_HS
        self.AC_params = AC_params
        self.BC_params = BC_params
        self.ar_params = ar_params
        self.dT_coeffs = dT_coeffs

        # check if all parameters specified
        assert all(p in material_params for p in self.AC_params.keys())
        assert all(p in material_params for p in self.BC_params.keys())
        assert all(ar_p in params_active for ar_p in self.ar_params.keys())
        assert all(t_p in temperature_params for t_p in self.dT_coeffs.keys())

        self.x_profile = None  # structure composition profile
        self.is_dimensionless = False

    def _set_T(self, T):
        return self.T_HS if T is None else T 

    def _dimension(self, value, unit):
        return value/units[unit] if self.is_dimensionless else value

    def setup_layers(self, layers_design):
        """
        Function for setting up all structure layers for given parameters.
        layers_design - dictionary with all parameters:
            names - layers names
            thicknesses of layers
            x - part of Al in AlGaAs
            Nd - doping profile for donors
            Na - doping profile for acceptors
        """
        layers = layers_design['names']
        d = layers_design['thickness']
        x = layers_design['x']
        Nd = layers_design['Nd']
        Na = layers_design['Na']

        self.structure = dict()

        grad_layers = []
        for i, l in enumerate(layers):
            if 'grad' not in l:  # skip gardient layers  
                is_active = True if l == 'active' else False
                self.structure[l] = Layer(l, d[i], active=is_active)
                self.structure[l].update(self.set_initial_params(x=x[i]))
                self.structure[l].update({'Nd': Nd[i], 'Na': Na[i]})
                if l == 'active':
                    self.structure[l].update(self.ar_params)
            else:
                layers[i] = 'grad' + str(len(grad_layers) + 1)                
                grad_layers.append(layers[i])

        for i, l in enumerate(layers):
            if l in grad_layers:  # add gradient layers
                self.structure[l] = self.structure[layers[i-1]].\
                make_gradient_layer(self.structure[layers[i+1]], 'gradient', d[i])

        self.layers_list = [self.structure[l] for l in layers]
        return self.layers_list


    def Eg_AlGaAs(self, x=0, T=None):
        """Band gap energy"""
        T = self._set_T(T)
        T_dim = T*units['T'] if self.is_dimensionless else T

        Eg = _Eg_AlGaAs(x)
        Eg_T = Eg + (self.dT_coeffs['Eg_A'] * T_dim**2) / \
            (self.dT_coeffs['Eg_B'] + T_dim)

        # !!! fix calculations of Ec and Ev
        Ec = self.BC_params['Eg'] + (Eg_T - self.BC_params['Eg'])*2/3
        Ev = 0 - (Eg_T - self.BC_params['Eg'])*1/3
        return self._dimension(Eg_T, 'Eg'), self._dimension(Ec, 'Ec'), \
            self._dimension(Ev, 'Ev') 

    def mu_n(self, x=0, T=None):
        """Electron mobility"""
        T = self._set_T(T)

        # weakly doped AlGaAs
        mu_n = _mu_n_AlGaAs(x)
        mu_n_T = mu_n * (T/self.T_HS)**self.dT_coeffs['mu_n_P']

        return self._dimension(mu_n_T, 'mu_n')

    def mu_p(self, x=0, T=None):
        """Hole mobility"""
        T = self._set_T(T)

        # weakly doped AlGaAs
        mu_p = _mu_p_AlGaAs(x)
        mu_p_T = mu_p * (T/self.T_HS)**self.dT_coeffs['mu_p_P']

        return self._dimension(mu_p_T, 'mu_p')

    def Nc(self, x=0, T=None):
        """Density of states in conductive band"""
        T = self._set_T(T)
        T_dim = T*units['T'] if self.is_dimensionless else T
        self.Vt = const.kb*T_dim

        # 1e-6 for converting to cm
        Nc_AlAs = 2*((2*np.pi*self.AC_params['m_e']*self.Vt*const.m0*const.q)/\
                     (const.h**2))**(3/2)*1e-6
        Nc_GaAs = 2*((2*np.pi*self.BC_params['m_e']*self.Vt*const.m0*const.q)/\
                     (const.h**2))**(3/2)*1e-6

        # !!! linear approximation - without bowing parameter
        Nc_AlGaAs = Nc_AlAs*x + Nc_GaAs*(1-x)

        return self._dimension(Nc_AlGaAs, 'Nc')

    def Nv(self, x=0, T=None):
        """Density of states in valence band"""
        T = self._set_T(T)
        T_dim = T*units['T'] if self.is_dimensionless else T
        self.Vt = const.kb*T_dim

        # 1e-6 for converting to cm
        Nv_AlAs = 2*((2*np.pi*self.AC_params['m_h']*self.Vt*const.m0*const.q)/\
                     (const.h**2))**(3/2)*1e-6
        Nv_GaAs = 2*((2*np.pi*self.BC_params['m_h']*self.Vt*const.m0*const.q)/\
                     (const.h**2))**(3/2)*1e-6

        # without bowing parameter
        Nv_AlGaAs = Nv_AlAs*x + Nv_GaAs*(1-x)

        return self._dimension(Nv_AlGaAs, 'Nv')

    def tau_n(self, x=0, T=None):
        """Nonradiative recombination coefficient"""
        tau_n = self.AC_params['tau_n']
        return self._dimension(tau_n, 'tau_n')

    def tau_p(self, x=0, T=None):
        """Nonradiative recombination coefficient"""
        tau_p = self.AC_params['tau_p']
        return self._dimension(tau_p, 'tau_n')

    def B(self, x=0, T=None):
        """Radiative recombination coefficient"""
        T = self._set_T(T)

        B = self.AC_params['B']*x + self.BC_params['B']*(1-x)
        B *= (T/self.T_HS)**self.dT_coeffs['B_P']
        return self._dimension(B, 'B')

    def Cn(self, x=0, T=None):
        """Auger recombination"""
        T = self._set_T(T)
        T_dim = T*units['T'] if self.is_dimensionless else T
        self.Vt = const.kb*T_dim
        dV_inv = 1/(self.Vt * self.T_HS/T) - 1/self.Vt

        Cn = self.AC_params['Cn']*x + self.BC_params['Cn']*(1-x)
        Cn *= np.exp(self.dT_coeffs['C_Ea']*dV_inv)
        return self._dimension(Cn, 'Cn')

    def Cp(self, x=0, T=None):
        """Auger recombination"""
        T = self._set_T(T)
        T_dim = T*units['T'] if self.is_dimensionless else T
        self.Vt = const.kb*T_dim
        dV_inv = 1/(self.Vt * self.T_HS/T) - 1/self.Vt

        Cp = self.AC_params['Cp']*x + self.BC_params['Cp']*(1-x)
        Cp *= np.exp(self.dT_coeffs['C_Ea']*dV_inv)
        return self._dimension(Cp, 'Cp')

    def n_refr(self, x=0, T=None):
        """Refractive index and dielectric permittivity"""
        T = self._set_T(T)

        # !!! dont take in account temperature and other effects
        n_refr = _n_refr_AlGaAs(x)
        # only real part of dielectric permittivity
        eps = n_refr**2
        return self._dimension(n_refr, 'n_refr'), self._dimension(eps, 'eps')

    def fca_e(self, x=0, T=None):
        """Elecrons free carrier adsorbtion coefficient"""
        T = self._set_T(T)

        fca_e = self.AC_params['fca_e']*x + self.BC_params['fca_e']*(1-x)
        fca_e *= (T/self.T_HS)**self.dT_coeffs['fca_e_P']
        return self._dimension(fca_e, 'fca_e')

    def fca_h(self, x=0, T=None):
        """Elecrons free carrier adsorbtion coefficient"""
        T = self._set_T(T)

        fca_h = self.AC_params['fca_h']*x + self.BC_params['fca_h']*(1-x)
        fca_h *= (T/self.T_HS)**self.dT_coeffs['fca_h_P']
        return self._dimension(fca_h, 'fca_h')

    def g0(self, T=None):
        """Material gain constant"""
        T = self._set_T(T)
        dT = (T - self.T_HS)*units['T'] if self.is_dimensionless else \
            T - self.T_HS

        g0 = self.ar_params['g0'] + self.dT_coeffs['d_g0']*dT
        return self._dimension(g0, 'g0')

    def Ntr(self, T=None):
        """Material gain constant"""
        T = self._set_T(T)
        dT = (T - self.T_HS)*units['T'] if self.is_dimensionless else \
            T - self.T_HS

        Ntr = self.ar_params['N_tr'] + self.dT_coeffs['d_Ntr']*dT
        return self._dimension(Ntr, 'N_tr')

    def set_initial_params(self, x=0, T=None):
        """Function for setuping initial structure parameters"""
        T = self._set_T(T)

        _, Ec, Ev = self.Eg_AlGaAs(x=x)
        n_refr, eps = self.n_refr(x=x)

        params = dict(Ev=Ev, Ec=Ec, Nc=self.Nc(x=x), Nv=self.Nv(x=x), 
                      mu_n=self.mu_n(x=x), mu_p=self.mu_p(x=x),
                      tau_n=self.tau_n(x=x), tau_p=self.tau_p(x=x),
                      B=self.B(x=x), Cn=self.Cn(x=x), Cp=self.Cp(x=x),
                      eps=eps, n_refr=n_refr, 
                      fca_e=self.fca_e(x=x), fca_h=self.fca_h(x=x), x=x)
        return params

    def update_all_params(self, is_dimensionless, T=None):
        """Temperature update of all params"""

        assert self.x_profile is not None