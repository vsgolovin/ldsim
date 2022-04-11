import numpy as np
import matplotlib.pyplot as plt
from ldsim.models import LaserDiodeModel1d
from sample_laser import layers


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
current_densities = np.zeros_like(voltages)
output_power = np.zeros_like(voltages)

# solve the drift-diffusion problem for every voltage
for i, v in enumerate(voltages):
    print(f'{v:.3f}V', end=' / ')
    ld.apply_voltage(v)
    # store previous solution fluctuation
    fluct = 1  # any value so that the first iteration is performed
    while fluct > 5e-8:
        fluct = ld.lasing_step(0.1, (1.0, 0.1))
    print(f'{ld.iterations} iterations')
    current_densities[i] = ld.get_current_density() * (-1)
    output_power[i] = ld.get_output_power()

ld.original_units()

plt.figure('J-V curve')
plt.plot(voltages, current_densities * 1e-3, 'b.-')
plt.xlabel('Voltage (V)')
plt.ylabel('Current density (kA/cm$^2$)')

currents = current_densities * ld.w * ld.L
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
