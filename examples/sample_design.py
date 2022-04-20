"""
Sample epitaxial design.
"""

from ldsim.preprocessing.design import Layer, EpiDesign
from ldsim.preprocessing.material import material_AlGaAs



#%% Specify layers properties

# layers thickness in microns
d = [0.25, 0.1, 1.4, 0.1, 0.45, 0.03, 0.45, 0.1, 1.4, 0.1, 0.25]
d = [i*1e-4 for i in d]

# X for Al(x)Ga(1-x)As
x = [0, 'grad', 0.4, 'grad', 0.25, 0, 0.25, 'grad', 0.4, 'grad', 0]

# doping profiles
Nd = [1e18, 'grad', 5e17, 'grad', 1e17, 2e16, 0, 'grad', 0, 'grad', 0]
Na = [0, 'grad', 0, 'grad', 0, 0, 1e17, 'grad', 1e18, 'grad', 2e18]

# layers names
layers = ['n_contact', 'grad', 'n_cladding', 'grad', 'n_waveguide',
          'active', 'p_waveguide', 'grad', 'p_cladding', 'grad', 'p_contact']

assert len(d) == len(x) == len(Nd) == len(Na) == len(layers), \
'All lengths should be equal!'

#%% Define material parameters

T_HS = 300  

# material parameters for binary compounds: AlAs and GaAs
AlAs = dict(Eg=3.1, m_e=0.15, m_h=0.7, tau_n=5e-9, tau_p=5e-9, B=1e-10, 
            Cn=2e-30, Cp=2e-30, fca_e=4e-18, fca_h=12e-18)
GaAs = dict(Eg=1.424, m_e=0.063, m_h=0.51, tau_n=5e-9, tau_p=5e-9, B=1e-10, 
            Cn=2e-30, Cp=2e-30, fca_e=4e-18, fca_h=12e-18)
ar_params = dict(g0=1500, N_tr=1.85e18)  # active region params

# temperature dependence coefficients of model parameters (for GaAs)
# _p - power dependence, _Ea - activation energy,
# Rt - temperature resistance
dT_AlGaAs = dict(Eg_A=-5.41e-4, Eg_B=204, mu_n_P=-1, mu_p_P=-2, B_P=-1, 
                 C_Ea=0.1, fca_e_P=1, fca_h_P=2, d_g0=-2, d_Ntr=2e15, Rt=2)


#%% Setup layers
AlGaAs = material_AlGaAs(T_HS, AlAs, GaAs, ar_params, dT_AlGaAs)  # material

layers_list = AlGaAs.setup_layers(layers, d, x, Nd, Na)

epi = EpiDesign(layers_list)


#%% Create Layer objects
ncont = Layer(name='n-contact', dx=d[0])
ncont.update(AlGaAs.set_initial_params(x=x[0]))
ncont.update({'Nd': Nd[0], 'Na': Na[0]})

ncl = Layer(name='n-cladding', dx=d[2])
ncl.update(AlGaAs.set_initial_params(x=x[2]))
ncl.update({'Nd': Nd[2], 'Na': Na[2]})

ngrad2 = ncont.make_gradient_layer(ncl, 'gradient', d[1])

nwg = Layer(name='n-waveguide', dx=d[4])
nwg.update(AlGaAs.set_initial_params(x=x[4]))
nwg.update({'Nd': Nd[4], 'Na': Na[4]})

ngrad = ncl.make_gradient_layer(nwg, 'gradient', d[3])

act = Layer(name='active', dx=d[5], active=True)
act.update(AlGaAs.set_initial_params(x=x[5]))
act.update({'Nd': Nd[5], 'Na': Na[5]})
act.update(ar_params)

pwg = Layer(name='p-waveguide', dx=d[6])
pwg.update(AlGaAs.set_initial_params(x=x[6]))
pwg.update({'Nd': Nd[6], 'Na': Na[6]})

pcl = Layer(name='p-cladding', dx=d[8])
pcl.update(AlGaAs.set_initial_params(x=x[8]))
pcl.update({'Nd': Nd[8], 'Na': Na[8]})

pgrad = pwg.make_gradient_layer(pcl, 'gradient', d[7])

pcont = Layer(name='p-contact', dx=d[10])
pcont.update(AlGaAs.set_initial_params(x=x[10]))
pcont.update({'Nd': Nd[10], 'Na': Na[10]})

pgrad2 = pcl.make_gradient_layer(pcont, 'gradient', d[9])

# create design as a list of layers
epi1 = EpiDesign([ncont, ngrad2, ncl, ngrad, nwg, act, pwg, pgrad, pcl,
                 pgrad2, pcont])




