# -*- coding: utf-8 -*-
"""
Separate file for generating grid and viewing the operating mode profile
of a laser diode with design described in `sample_design.py`.
"""

import matplotlib.pyplot as plt
from sample_design import epi
from laser_diode_xz import LaserDiode

# change default Matplotlib settings
plt.rc('lines', linewidth=0.7)
plt.rc('figure.subplot', left=0.15, right=0.85)

# create an instance of `LaserDiode` class
# all parameters except for `ar_inds` (active region layer indices)
# are actually irrelevant in this case
ld = LaserDiode(epi=epi, L=3000e-4, w=100e-4, R1=0.95, R2=0.05,
                lam=0.87e-4, ng=3.9, alpha_i=0.5, beta_sp=1e-4)

# generate nonuniform mesh
# see method docstring for detailed description
x, y, step = \
    ld.gen_nonuniform_mesh(step_min=1e-7, step_max=20e-7, step_uni=5e-8,
                           sigma=1e-5, y_ext=[0.3, 0.3])
# x -- initial (uniform) mesh nodes used for generating nonuniform mesh
# y -- bandgap values at x
# step -- nonuniform mesh nodes' spacing, inversely proportional to change in y

# plotting
x *= 1e4     # convert to micrometers
step *= 1e7  # convert to nanometers
plt.figure('Grid generation')
plt.plot(x, step)
plt.ylabel('Grid step (nm)')
plt.xlabel(r'$x$ ($\mu$m)')
plt.twinx()
plt.plot(x, y, 'k:')
plt.ylabel('$E_g$ (eV)')

# calculate waveguide mode profile
n_modes = 3  # number of modes to find
x_wg, modes, gammas = ld.solve_waveguide(step=1e-8, n_modes=n_modes,
                                         remove_layers=(1, 1))
plt.figure('Waveguide mode')
for i in range(n_modes):
    gamma = gammas[i] * 1e2
    plt.plot(x_wg*1e4, modes[:, i],
             label=r'$\Gamma$ = ' + f'{gamma:.2f}%')
plt.legend()
plt.ylabel('Mode intensity')
plt.xlabel(r'$x$ ($\mu$m)')
plt.twinx()
plt.plot(ld.xin*1e4, ld.yin['n_refr'], 'k:', lw=0.5)
plt.ylabel('Refractive index')
