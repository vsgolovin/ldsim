"""
Separate file for generating grid and viewing the operating mode profile
of a laser diode with design described in `sample_design.py`.
"""

import matplotlib.pyplot as plt
from ldsim.models import LaserDiodeModel1d
from sample_laser import layers_design, AlGaAs

# change default Matplotlib settings
plt.rc('lines', linewidth=0.7)
plt.rc('figure.subplot', left=0.15, right=0.85)

# create an instance of `LaserDiode` class
# all parameters except for `ar_inds` (active region layer indices)
# are actually irrelevant in this case
ld = LaserDiodeModel1d(layers_design, AlGaAs, L=0.3, w=0.01, R1=0.95, R2=0.05, 
                       lam=0.87e-4, ng=3.9, alpha_i=0.5, beta_sp=1e-5)

# generate nonuniform mesh
# see method docstring for detailed description
ld.generate_nonuniform_mesh(y_ext=[0., 0.])

# x -- initial (uniform) mesh nodes used for generating nonuniform mesh
# y -- bandgap values at x
# step -- nonuniform mesh nodes' spacing, inversely proportional to change in y

# plotting
plt.figure('Grid generation')
plt.ylabel('Grid step (nm)')
plt.xlabel(r'$x$ ($\mu$m)')
plt.twinx()
plt.plot(ld.xn*1e4, ld.vn['Eg'], 'k:')
plt.ylabel('$E_g$ (eV)')

# calculate waveguide mode profile
n_modes = 3  # number of modes to find
mode = ld.solve_waveguide(step=1e-8, n_modes=n_modes, remove_layers=(1, 1))
plt.figure('Waveguide mode')
for i in range(n_modes):
    gamma = mode['gammas'][i] * 1e2
    plt.plot(mode['x']*1e4, mode['modes'][:, i],
             label=r'$\Gamma$ = ' + f'{gamma:.2f}%')
plt.legend()
plt.ylabel('Mode intensity')
plt.xlabel(r'$x$ ($\mu$m)')
plt.twinx()
plt.plot(ld.xn*1e4, ld.vn['n_refr'], 'k:', lw=0.5)
plt.ylabel('Refractive index')
plt.show()
