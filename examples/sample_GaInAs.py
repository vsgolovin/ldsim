"""
Sample laser design.
"""

from ldsim.preprocessing.design import LaserDiode, DEFAULT_TEMPERATURE
from ldsim.preprocessing.material import material_AlGaAs

#%% Design with one QW
# 'grad' - means that property changing (linearly) from previous layer to next

# layers names
layers = ['n_contact', 'n_cladding', 'grad', 'n_waveguide', 'spac1', 'active', 
          'spac2', 'p_waveguide', 'grad', 
          'p_cladding', 'p_contact']

# layers thickness in microns
d = [0.25, 1.4, 0.1, 0.45, 0.02, 0.01, 0.02, 0.45, 0.1, 1.4, 0.25]
d = [i*1e-4 for i in d]

# material system for each layer
composition = [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
composition_keys = {'AlGaAs': 1, 'GaInAs': 2}
# composition: X for Al(x)Ga(1-x)As
x = [0, 0.3, 'grad', 0.1, 0, 'GaInAs', 0, 0.1, 'grad', 0.3, 0]

# doping profiles
Nd = [1e18, 5e17, 'grad', 1e17, 2e16, 2e16,  0, 0, 'grad', 0, 0]
Na = [0, 0, 'grad', 0, 0, 0, 2e16, 1e17, 'grad', 1e18, 2e18]

"""
#%% Design with two QWs

# layers names
layers = ['n_contact', 'n_cladding', 'grad', 'n_waveguide', 'spac1', 'active', 
          'spac2', 'active', 'spac3', 'p_waveguide', 'grad', 
          'p_cladding', 'p_contact']

# layers thickness in microns
d = [0.25, 1.4, 0.1, 0.45, 0.02, 0.01, 0.03, 0.01, 0.02, 0.45, 0.1, 1.4, 0.25]
d = [i*1e-4 for i in d]

# material system for each layer
composition = [1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1]
composition_keys = {'AlGaAs': 1, 'GaInAs': 2}
# composition: X for Al(x)Ga(1-x)As
x = [0, 0.3, 'grad', 0.1, 0, 'GaInAs', 0, 'GaInAs', 0, 0.1, 'grad', 0.3, 0]

# doping profiles
Nd = [1e18, 5e17, 'grad', 1e17, 2e16, 2e16, 2e16, 0, 0, 0, 'grad', 0, 0]
Na = [0, 0, 'grad', 0, 0, 0, 0, 2e16, 2e16, 1e17, 'grad', 1e18, 2e18]
"""

assert len(d) == len(x)== len(composition) == len(Nd) == len(Na) == len(layers)
layers_design = dict(names=layers, thickness=d, x=x, composition=composition, 
                     composition_keys=composition_keys, Nd=Nd, Na=Na)

#%% Define material parameters 

# material parameters for binary compounds: AlAs and GaAs
AlAs = dict(Eg=3.1, m_e=0.15, m_h=0.7, tau_n=5e-9, tau_p=5e-9, B=1e-10, 
            Cn=2e-30, Cp=2e-30, fca_e=4e-18, fca_h=12e-18)
GaAs = dict(Eg=1.424, m_e=0.063, m_h=0.51, tau_n=5e-9, tau_p=5e-9, B=1e-10, 
            Cn=2e-30, Cp=2e-30, fca_e=4e-18, fca_h=12e-18)
# active region GaInAs params
ar_params = dict(Eg=1.09, Ec=1.09*0.68, Ev=1.09*0.32, m_e=0.0505, m_h=0.486, 
                 tau_n=5e-9, tau_p=5e-9, B=1e-10, Cn=2e-30, Cp=2e-30, 
                 fca_e=4e-18, fca_h=12e-18, mu_n=16000, mu_p=424,
                 n_refr=3.58, eps=3.39**2, g0=1500, N_tr=1.85e18)  # X = 76%

# temperature dependence coefficients of model parameters (for GaAs)
# _P - power dependence, _Ea - activation energy,
# Rt - temperature resistance
dT_AlGaAs = dict(Eg_A=-5.41e-4, Eg_B=204, mu_n_P=-1, mu_p_P=-2, B_P=-1, 
                 C_Ea=0.1, fca_e_P=1, fca_h_P=2, d_g0=-2, d_Ntr=2e15, Rt=2.0)

#%% Setup layers
AlGaAs = material_AlGaAs(DEFAULT_TEMPERATURE, AlAs, GaAs, ar_params, dT_AlGaAs)

laser = LaserDiode(layers_design, AlGaAs, L=0.3, w=0.01, R1=0.95, R2=0.05, 
                   lam=1.06e-4, ng=3.9, alpha_i=0.5, beta_sp=1e-5, 
                   T_HS=DEFAULT_TEMPERATURE, T_dependent=True)




