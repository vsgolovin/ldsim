from os import path
import numpy as np
import matplotlib.pyplot as plt
from ldsim.models import LaserDiodeModel1d
from sample_laser import layers


EXPORT_FOLDER = 'results'

# initialize model
ld = LaserDiodeModel1d(layers, L=0.3, w=0.01, R1=0.95, R2=0.05, lam=0.87e-4,
                       ng=3.9, alpha_i=0.5, beta_sp=1e-5)
ld.generate_nonuniform_mesh(y_ext=[0., 0.])
mode = ld.solve_waveguide()

# find electrostatic potential and carrier densities corresponding
# to local charge neutrality (LCN)
ld.make_dimensionless()
ld.solve_equilibrium()

# applied voltage values
voltages = np.arange(0.0, 1.601, 0.05)

# simulation results
current_densities = np.zeros_like(voltages)
output_power = np.zeros_like(voltages)
# currents, associated with different spontaneous recombination mechanisms
I_srh = np.zeros_like(voltages)  # Shockley-Read-Hall
I_rad = np.zeros_like(voltages)  # radiative
I_aug = np.zeros_like(voltages)  # Auger
# total internal absorption
alpha_i = np.zeros_like(voltages)

# solve the drift-diffusion problem for every voltage
for i, v in enumerate(voltages):
    print(f'{v:.3f}V', end=' / ')
    ld.apply_voltage(v)
    # store previous solution fluctuation
    fluct = 1  # any value so that the first iteration is performed
    while fluct > 5e-8:
        if fluct > 1e-3:
            omega = 0.1
        else:
            omega = 0.25
        fluct = ld.lasing_step(omega, (1.0, omega))
    print(f'{ld.iterations} iterations')

    # save results
    current_densities[i] = ld.get_current_density() * (-1)
    output_power[i] = ld.get_output_power()
    I_srh[i] = ld.get_recombination_current('srh')
    I_rad[i] = ld.get_recombination_current('rad')
    I_aug[i] = ld.get_recombination_current('aug')
    alpha_i[i] = ld.get_loss()

    # export data for given voltage
    fname = f'{v:.3f}V.csv'
    fname = path.join(EXPORT_FOLDER, fname)
    ld.save_results(fname)

ld.original_units()
currents = current_densities * ld.w * ld.L  # can also use `.get_current()`

# export results for different voltages (current, power, etc)
ld.save_csv(
    fname=path.join(EXPORT_FOLDER, 'LIV.csv'),
    arrays=(voltages, currents, current_densities, output_power,
            I_srh, I_rad, I_aug, alpha_i),
    labels=('V', 'I', 'J', 'P', 'I_srh', 'I_rad', 'I_aug', 'alpha_i'),
    sep=','
)

# plot results
plt.figure('J-V curve')
plt.plot(voltages, current_densities * 1e-3, 'b.-')
plt.xlabel('Voltage (V)')
plt.ylabel('Current density (kA/cm$^2$)')

plt.figure('P-I curve')
plt.plot(currents, output_power, 'b.-')
plt.xlabel('Current (A)')
plt.ylabel('Output power (W)')

plt.figure('Band diagram')
plt.plot(ld.xn * 1e4, ld.vn['Ec']-ld.sol['psi'], color='k')
plt.plot(ld.xn * 1e4, ld.vn['Ev']-ld.sol['psi'], color='k')
plt.plot(ld.xn * 1e4, -ld.sol['phi_n'], 'b:', label=r'$\varphi_n$')
plt.plot(ld.xn * 1e4, -ld.sol['phi_p'], 'r:', label=r'$\varphi_p$')
plt.legend()
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel('$E$ (eV)')
plt.show()
