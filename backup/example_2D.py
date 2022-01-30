# -*- coding: utf-8 -*-
"""
Calculate P-I and J-V curves for a laser diode with design described in
`sample_design.py` by using a 2D (vertical-longitudinal / x-z)
drift-diffusion model.
"""

import numpy as np
import matplotlib.pyplot as plt
from sample_design import epi
from laser_diode_xz import LaserDiode
import units

# export settings
export = True
export_folder = 'results'

# set up the problem
# initialize as 1D, because below threshold 1D and 2D models are identical
ld = LaserDiode(epi=epi, L=3000e-4, w=100e-4, R1=0.95, R2=0.05,
                lam=0.87e-4, ng=3.9, alpha_i=0.5, beta_sp=1e-4)
ld.gen_nonuniform_mesh(step_min=1e-7, step_max=20e-7, sigma=1e-5,
                       y_ext=[0.3, 0.3])
ld.make_dimensionless()
ld.solve_waveguide(remove_layers=(1, 1))  # ignore contact layers
ld.solve_equilibrium()

mz = 10  # number of 1D slices along the longitudinal (z) axis

# arrays for storing results
voltages = np.arange(0, 2.51, 0.1)
mv = len(voltages)
J_values = np.zeros((mz, mv))
Jsrh_values = np.zeros((mz, mv))  # Shockley-Read-Hall
Jrad_values = np.zeros((mz, mv))  # radiative
Jaug_values = np.zeros((mz, mv))  # Auger
fca_values = np.zeros((mz, mv))   # free-carrier-absorption
P_values = np.zeros((2, mv))      # power from both facets
n_values = np.zeros((mz, mv))
p_values = np.zeros((mz, mv))

# solve the transport system for every voltage
# (twice at threshold voltage: 1D, then 2D)
# use previous solution as initial guess
print('Voltage Iterations Power')
i = 0
while i < mv:
    v = voltages[i]
    print('  %.2f' % v, end='  ')
    ld.apply_voltage(v)  # apply boundary conditions
    # track convergence with fluctuation
    # i.e., ratio of update vector and solution L2 norms
    fluct = 1  # initial value
    if ld.ndim == 1:
        fluct_max = 1e-8
    else:  # 2D
        fluct_max = 1e-11
    while fluct > fluct_max:  # perform Newton's method iterations
        # choose value of damping parameter `omega`
        # depending on fluctuation and number of dimensions
        if ld.ndim == 1:
            if fluct > 1e-3:
                omega = 0.05
                omega_S = (1.0, 0.05)  # see example_1D.py
            else:
                omega = 0.25
                omega_S = (1.0, 0.25)
        else:
            if fluct > 1e-1:
                omega = 0.05
                omega_S = (0.05, 0.05)
            else:
                omega = 1.0
                omega_S = (1.0, 1.0)
        # perform single Newton iteration
        fluct = ld.lasing_step(omega=omega, omega_S=omega_S, discr='mSG')

    P1, P2 = ld.get_P()
    print('  %5d    %.1e' % (ld.iterations, P1 + P2))
    if export:
        ld.export_results(folder=export_folder, x_to_um=True)

    if ld.ndim == 1 and P1 + P2 > 1e-2:  # reached threshold
        print('Reached threshold. Transitioning to a 2D model.')
        ld.to_2D(mz)
        continue  # find 2D solution at same voltage

    # save results to corresponding arrays
    J_values[:, i] = -1 * ld.get_J()
    P_values[:, i] = ld.get_P()
    Jsrh_values[:, i] = ld.get_Jsp('SRH')
    Jrad_values[:, i] = ld.get_Jsp('radiative')
    Jaug_values[:, i] = ld.get_Jsp('Auger')
    fca_values[:, i] = ld.get_FCA()
    n_values[:, i], p_values[:, i] = ld.get_n_active()

    # export longitudinal dependencies
    if ld.ndim == 2 and export:
        fname = '2D_f(z)_%.2fV.csv' % v
        with open(export_folder + '/' + fname, 'w') as f:
            f.write(','.join(('z', 'J', 'J_srh', 'J_rad', 'J_aug',
                              'FCA', 'S', 'n', 'p')))
            S = (ld.Sb + ld.Sf) * units.n * units.x
            S = (S[:-1] + S[1:]) / 2
            for k in range(len(ld.zin)):
                f.write('\n')
                zk = ld.zin[k] * units.x
                vals = map(str, (zk, J_values[k, i], Jsrh_values[k, i],
                                 Jrad_values[k, i], Jaug_values[k, i],
                                 fca_values[k, i], S[k],
                                 n_values[k, i], p_values[k, i]))
                f.write(','.join(vals))

    i += 1  # increase voltage

# prepare results for plotting / export
ld.original_units()
x = ld.xin * 1e4
J_avg = J_values.mean(axis=0)
I_values = J_avg * ld.w * ld.L
I_srh = Jsrh_values.mean(axis=0) * ld.w * ld.L
I_rad = Jrad_values.mean(axis=0) * ld.w * ld.L
I_aug = Jaug_values.mean(axis=0) * ld.w * ld.L

# change default Matplotlib settings
plt.rc('lines', linewidth=0.7)
plt.rc('figure.subplot', left=0.15, right=0.85)

# plot results
plt.figure('Waveguide mode')
plt.plot(x, ld.yin['wg_mode'], 'b-')
plt.ylabel('Mode intensity', color='b')
plt.xlabel(r'$x$ ($\mu$m)')
plt.twinx()
plt.plot(x, ld.yin['n_refr'], 'k-', lw=0.5)
plt.ylabel('Refractive index')

plt.figure('J-V curve')
plt.plot(voltages, J_avg*1e-3, 'b.-')
plt.xlabel('$V$')
plt.ylabel('$j$ (kA / cm$^2$)')

plt.figure('P-I curve')
plt.plot(I_values, P_values[0, :], 'r.-', label='$P_1$')
plt.plot(I_values, P_values[1, :], 'b.-', label='$P_2$')
plt.legend()
plt.xlabel('$I$ (A)')
plt.ylabel('$P$ (W)')

plt.figure('S(z) and n(z)')
plt.plot(ld.zbn*1e4, ld.Sf + ld.Sb, 'rx-')
plt.ylabel('Photon density $S$ (cm$^{-2}$)', color='r')  # 1 / (y*z)
plt.xlabel(r'$z$ ($\mu$m)')
plt.twinx()
plt.plot(ld.zin*1e4, n_values[:, -1], 'b.-')
plt.ylabel('$n_{a}$ (cm$^{-3}$)', color='b')

# export arrays with simulation results
if export:
    with open(export_folder + '/' + 'LIV_2D.csv', 'w') as f:
        f.write(','.join(('V', 'J', 'I', 'P1', 'P2', 'I_srh', 'I_rad',
                          'I_aug', 'FCA')))
        for i in range(len(voltages)):
            f.write('\n')
            fca = fca_values[:, i].mean()  # avg FCA over z
            vals = map(str, (voltages[i], J_avg[i], I_values[i],
                             P_values[0, i], P_values[1, i],
                             I_srh[i], I_rad[i], I_aug[i], fca))
            f.write(','.join(vals))
