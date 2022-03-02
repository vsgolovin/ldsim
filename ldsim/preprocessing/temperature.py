"""
Temparature dependensies of model parameters
"""

import numpy as np
from ldsim.constants import kb

params = ['Eg_A', 'Eg_B', 'mu_n_p', 'mu_p_p']
alloy_params = ['Nc', 'Nv', 'B_p', 'C_Ea', 'fca_e_p', 'fca_h_p', 
                'd_g0', 'd_Ntr', 'Rt']

class temperature_dependensies:
    
    def __init__(self, T0, T0_AlGaAs, dT_AlGaAs):
        "Initialize all model params temeperature coefficients"
        self.T0 = T0  # heatsink temperature
        # alloy params
        self.T0_AlGaAs = T0_AlGaAs
        self.dT_AlGaAs = dT_AlGaAs
        self.Rt = dT_AlGaAs['Rt']  # temperature resistance
        
    def Eg_AlGaAs(self, T, x=None):
        "Band gap energy"
        if x is not None:  # need to specify x for each layer (Al(x)Ga(1-x)As)
            Eg_AlGaAs = 1.424 + 1.155*x + 0.37*x**2
        else:
            Eg_AlGaAs = self.T0_AlGaAs['Eg']
            
        Eg_T = Eg_AlGaAs + (self.dT_AlGaAs['Eg_A'] * T**2) / \
            (self.dT_AlGaAs['Eg_B'] + T)
        Ec = self.T0_AlGaAs['Ec'] + (Eg_T - Eg_AlGaAs)*2/3
        Ev = self.T0_AlGaAs['Ev'] - (Eg_T - Eg_AlGaAs)*1/3
        return Eg_T, Ec, Ev
    
    def mu_n(self, T):
        "Electron mobility"
        return self.T0_AlGaAs['mu_n'] * (T/self.T0)**self.dT_AlGaAs['mu_n_p']
    
    def mu_p(self, T):
        "Electron mobility"
        return self.T0_AlGaAs['mu_p'] * (T/self.T0)**self.dT_AlGaAs['mu_p_p']
    
    def Nc(self, T):
        "Conductive density of states"
        return self.T0_AlGaAs['Nc'] * (T/self.T0)**(3/2)
    
    def Nv(self, T):
        "Conductive density of states"
        return self.T0_AlGaAs['Nv'] * (T/self.T0)**(3/2)
    
    def B(self, T):
        "Radiative recombination"
        return self.T0_AlGaAs['B'] * (T/self.T0)**self.dT_AlGaAs['B_p']
    
    def Cn(self, T, Vt):
        "Auger recombination"
        dV_inv = 1/(Vt*self.T0/T) - 1/Vt
        return self.T0_AlGaAs['Cn'] * np.exp(self.dT_AlGaAs['C_Ea']*dV_inv)
    
    def Cp(self, T, Vt):
        "Auger recombination"
        dV_inv = 1/(Vt*self.T0/T) - 1/Vt
        return self.T0_AlGaAs['Cp'] * np.exp(self.dT_AlGaAs['C_Ea']*dV_inv)
    
    def g0(self, T):
        "Material gain constant"
        return self.T0_AlGaAs['g0'] + self.dT_AlGaAs['d_g0'] * (T - self.T0)
    
    def Ntr(self, T):
        "Material gain constant"
        return self.T0_AlGaAs['N_tr'] + self.dT_AlGaAs['d_Ntr'] * (T - self.T0)
    
    def fca_e(self, T):
        "Elecrons free carrier adsorbtion coefficient"
        return self.T0_AlGaAs['fca_e'] * (T/self.T0)**self.dT_AlGaAs['fca_e_p']
    
    def fca_h(self, T):
        "Elecrons free carrier adsorbtion coefficient"
        return self.T0_AlGaAs['fca_h'] * (T/self.T0)**self.dT_AlGaAs['fca_h_p']
    
    def update_all(self, T, Vt):
        return self.Nc(T), self.Nv(T), self.B(T), self.Cn(T, Vt), \
               self.Cp(T, Vt), self.g0(T), self.Ntr(T), self.fca_e(T), \
               self.fca_h(T), self.mu_n(T), self.mu_p(T)


