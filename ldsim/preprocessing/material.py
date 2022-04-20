"""
Temparature dependensies of model parameters

Parameters was taken from:
    NSM Archive - Physical Properties of Semiconductors (http://matprop.ru/)
    Vurgaftman, I. et al - Band parameters for III-V compound semiconductors and their alloys
"""

import numpy as np
from ldsim.constants import kb, m0, q, h, c
from ldsim.preprocessing.design import Layer, EpiDesign


#%% Lists of parameters
material_params = ['Eg', 'm_e', 'm_h', 'tau_n','tau_p', 'B', 'Cn', 'Cp', 
                   'fca_e', 'fca_h']

ar_params = ['g0', 'N_tr']

temperature_params = ['Eg_A', 'Eg_B', 'mu_n_P', 'mu_p_P', 'B_P', 'C_Ea',
                      'fca_e_P', 'fca_h_P', 'd_g0', 'd_Ntr', 'Rt']


#%% Some dependencies with bowing parameters
def _Eg_AlGaAs(x):
    return 1.519 + 1.155*x + 0.37*x**2  # for x < 0.41

def _mu_n_AlGaAs(x):
    return 8000 - 22000*x + 10000*x**2  # for x < 0.45

def _mu_p_AlGaAs(x):
    return 370 - 970*x + 740*x**2  # for x < 0.45

def _n_refr_AlGaAs(x, l=1e-4):
    # default lambda = 1 micron
    # https://www.batop.de/information/n_AlGaAs.html
    l *= 1e-2 # to meters
    
    a0 = 6.3 + 19*x
    b0 = 9.4 - 10.2*x
    e0 = _Eg_AlGaAs(x)
    de = 0.34 - 0.04*x  # spin-orbit splitting energy
    
    y = h*c / (q*l*e0*1e2)
    y0 = h*c / (q*l*(e0+de)*1e2)
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
        self.Vt0 = kb*T_HS
        self.AC_params = AC_params
        self.BC_params = BC_params
        self.ar_params = ar_params
        self.dT_coeffs = dT_coeffs
        
        # check if all parameters specified
        for param in self.AC_params.keys():
            assert param in material_params
        for param in self.BC_params.keys():
            assert param in material_params
        for ar_param in self.ar_params.keys():
            assert ar_param in ar_params
        for t_param in self.dT_coeffs.keys():
            assert t_param in temperature_params 
            
    def setup_layers(self, layers, d, x, Nd, Na):
        """
        Function for setting up all structure layers for given parameters:
            layers - list of names
            d - thicknesses
            x - part of Al in AlGaAs
            Nd - doping profile for donors
            Na - doping profile for acceptors
        """
        self.structure = dict()
        
        for i, l in enumerate(layers):
            n_grad_layers = 0
            if l == 'grad':  # skip gardient layers
                n_grad_layers += 1
                layer_name = 'grad' + str(n_grad_layers)
                self.structure[layer_name] = None
                continue
            
            if l == 'active':
                self.structure[l] = Layer(name=l, dx=d[i], active=True)
                self.structure[l].update(self.ar_params)
            else:
                self.structure[l] = Layer(name=l, dx=d[i], active=False)
            self.structure[l].update(self.set_initial_params(x=x[i]))
            self.structure[l].update({'Nd': Nd[i], 'Na': Na[i]})
        
        n_grad_layers = 0
        for i, l in enumerate(layers):
            if l == 'grad':  # add gradient layers
                n_grad_layers += 1
                layer_name = 'grad' + str(n_grad_layers)
                self.structure[layer_name] = self.structure[layers[i-1]].\
                make_gradient_layer(self.structure[layers[i+1]], 'gradient', d[i])
                layers[i] = layer_name
                
        self.layers_list = [self.structure[l] for l in layers]
        return self.layers_list
	
            
    def Eg_AlGaAs(self, x=0, T=None):
        "Band gap energy"
        T = self.T_HS if T is None else T
        
        Eg = _Eg_AlGaAs(x)
        Eg_T = Eg + (self.dT_coeffs['Eg_A'] * T**2) / \
            (self.dT_coeffs['Eg_B'] + T)
            
        # !!! fix calculations of Ec and Ev
        Ec = self.BC_params['Eg'] + (Eg_T - self.BC_params['Eg'])*2/3
        Ev = 0 - (Eg_T - self.BC_params['Eg'])*1/3
        return Eg_T, Ec, Ev 
    
    def mu_n_AlGaAs(self, x=0, T=None):
        "Electron mobility"
        T = self.T_HS if T is None else T
        
        # weakly doped AlGaAs
        mu_n = _mu_n_AlGaAs(x)
        mu_n_T = mu_n * (T/self.T_HS)**self.dT_coeffs['mu_n_P']
        
        return mu_n_T
    
    def mu_p_AlGaAs(self, x=0, T=None):
        "Hole mobility"
        T = self.T_HS if T is None else T
        
        # weakly doped AlGaAs
        mu_p = _mu_p_AlGaAs(x)
        mu_p_T = mu_p * (T/self.T_HS)**self.dT_coeffs['mu_p_P']
        
        return mu_p_T
    
    def Nc(self, x=0, T=None):
        "Density of states in conductive band"
        T = self.T_HS if T is None else T
        self.Vt = kb*T
        
        # 1e-6 for converting to cm
        Nc_AlAs = 2*((2*np.pi*self.AC_params['m_e']*self.Vt*m0*q)/(h**2)) \
            **(3/2)*1e-6
        Nc_GaAs = 2*((2*np.pi*self.BC_params['m_e']*self.Vt*m0*q)/(h**2)) \
            **(3/2)*1e-6
        
        # !!! linear approximation - without bowing parameter
        Nc_AlGaAs = Nc_AlAs*x + Nc_GaAs*(1-x)
        
        return Nc_AlGaAs
    
    def Nv(self, x=0, T=None):
        "Density of states in valence band"
        T = self.T_HS if T is None else T
        self.Vt = kb*T
        
        # 1e-6 for converting to cm
        Nv_AlAs = 2*((2*np.pi*self.AC_params['m_h']*self.Vt*m0*q)/(h**2)) \
            **(3/2)*1e-6
        Nv_GaAs = 2*((2*np.pi*self.BC_params['m_h']*self.Vt*m0*q)/(h**2)) \
            **(3/2)*1e-6
        
        # without bowing parameter
        Nv_AlGaAs = Nv_AlAs*x + Nv_GaAs*(1-x)
        
        return Nv_AlGaAs
    
    def tau_n(self, x=0, T=None):
        "Nonradiative recombination coefficient"
        tau_n = self.AC_params['tau_n']
        return tau_n
    
    def tau_p(self, x=0, T=None):
        "Nonradiative recombination coefficient"
        tau_p = self.AC_params['tau_p']
        return tau_p
    
    def B(self, x=0, T=None):
        "Radiative recombination coefficient"
        T = self.T_HS if T is None else T
        
        B = self.AC_params['B']*x + self.BC_params['B']*(1-x)
        B *= (T/self.T_HS)**self.dT_coeffs['B_P']
        return B
    
    def Cn(self, x=0, T=None):
        "Auger recombination"
        T = self.T_HS if T is None else T
        self.Vt = kb*T
        dV_inv = 1/(self.Vt * self.T_HS/T) - 1/self.Vt
        
        Cn = self.AC_params['Cn']*x + self.BC_params['Cn']*(1-x)
        Cn *= np.exp(self.dT_coeffs['C_Ea']*dV_inv)
        return Cn
    
    def Cp(self, x=0, T=None):
        "Auger recombination"
        T = self.T_HS if T is None else T
        self.Vt = kb*T
        dV_inv = 1/(self.Vt * self.T_HS/T) - 1/self.Vt
        
        Cp = self.AC_params['Cp']*x + self.BC_params['Cp']*(1-x)
        Cp *= np.exp(self.dT_coeffs['C_Ea']*dV_inv)
        return Cp
    
    def n_refr(self, x=0, T=None):
        "Refractive index and dielectric permittivity"
        T = self.T_HS if T is None else T
        
        # !!! dont take in account temperature and other effects
        n_refr = _n_refr_AlGaAs(x)
        # only real part of dielectric permittivity
        eps = n_refr**2
        return n_refr, eps
    
    def fca_e(self, x=0, T=None):
        "Elecrons free carrier adsorbtion coefficient"
        T = self.T_HS if T is None else T
        
        fca_e = self.AC_params['fca_e']*x + self.BC_params['fca_e']*(1-x)
        fca_e *= (T/self.T_HS)**self.dT_coeffs['fca_e_P']
        return fca_e
    
    def fca_h(self, x=0, T=None):
        "Elecrons free carrier adsorbtion coefficient"
        T = self.T_HS if T is None else T
        
        fca_h = self.AC_params['fca_h']*x + self.BC_params['fca_h']*(1-x)
        fca_h *= (T/self.T_HS)**self.dT_coeffs['fca_h_P']
        return fca_h
    
    def g0(self, T=None):
        "Material gain constant"
        T = self.T_HS if T is None else T
        
        g0 = self.ar_params['g0'] + self.dT_coeffs['d_g0']*(T - self.T_HS)
        return g0
    
    def Ntr(self, T=None):
        "Material gain constant"
        T = self.T_HS if T is None else T
        
        Ntr = self.ar_params['N_tr'] + self.dT_coeffs['d_Ntr']*(T - self.T_HS)
        return Ntr
    
    def set_initial_params(self, x=0, T=None):
        "Maybe it usefull to store all params in separate class"
        T = self.T_HS if T is None else T
        
        _, Ec, Ev = self.Eg_AlGaAs(x=x)
        n_refr, eps = self.n_refr(x=x)
        
        params = dict(Ev=Ev, Ec=Ec, Nc=self.Nc(x=x), Nv=self.Nv(x=x), 
                      mu_n=self.mu_n_AlGaAs(x=x), mu_p=self.mu_p_AlGaAs(x=x),
                      tau_n=self.tau_n(x=x), tau_p=self.tau_p(x=x),
                      B=self.B(x=x), Cn=self.Cn(x=x), Cp=self.Cp(x=x),
                      eps=eps, n_refr=n_refr, 
                      fca_e=self.fca_e(x=x), fca_h=self.fca_h(x=x))
        return params
 
    


