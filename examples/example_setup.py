"""
Separate file for generating grid and viewing the operating mode profile
of a laser diode with design described in `sample_design.py`.
"""

import pandas as pd
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
ld.generate_nonuniform_mesh(step_min=1e-7, step_max=20e-7, step_uni=5e-8,
                            sigma=1e-5, y_ext=[1., 1.])

# x -- initial (uniform) mesh nodes used for generating nonuniform mesh
# y -- bandgap values at x
# step -- nonuniform mesh nodes' spacing, inversely proportional to change in y
x = ld.xn
y = ld.vn['Eg']
step = x[1:] - x[:-1]

# plotting
x *= 1e4     # convert to micrometers
step *= 1e7  # convert to nanometers
plt.figure('Grid generation')
plt.plot(x[1:], step)
plt.ylabel('Grid step (nm)')
plt.xlabel(r'$x$ ($\mu$m)')
plt.twinx()
plt.plot(x, y, 'k:')
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
plt.plot(x, ld.vn['n_refr'], 'k:', lw=0.5)
plt.ylabel('Refractive index')
plt.show()


def export_design(ld, params_list=['Ec', 'C_dop', 'wg_mode'], plot=True):
    """
    Parameters
    ----------
    ld : LaserDiode
        model class with all params
    params_list : list
        list with names of all params

    Write necessary parameters to csv file
    """
    
    from os import path, mkdir
    EXPORT_FOLDER = 'results'
    if not path.isdir(EXPORT_FOLDER):
        mkdir(EXPORT_FOLDER)
    
    design = pd.DataFrame()
    design['x'] = ld.xn    
    for i in params_list:
        if i == 'C_dop':
            design[i] = abs(ld.vn[i])
        else:
            design[i] = ld.vn[i]
    
    design.to_csv(EXPORT_FOLDER + '/design.csv', index=False)    
    return design

export_design(ld)