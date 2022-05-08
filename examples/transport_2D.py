import numpy as np
import matplotlib.pyplot as plt
from ldsim.models import LaserDiodeModel2d
from sample_laser import layers


# initialize model
ld = LaserDiodeModel2d(layers, L=0.3, w=0.01, R1=0.95, R2=0.05, lam=0.87e-4,
                       ng=3.9, alpha_i=0.5, beta_sp=1e-5, num_slices=10)
ld.generate_nonuniform_mesh(y_ext=[0., 0.])
mode = ld.solve_waveguide()

# find electrostatic potential and carrier densities corresponding
# to local charge neutrality (LCN)
ld.make_dimensionless()
ld.solve_equilibrium()

# applied voltage values
voltages = np.arange(0.0, 1.501, 0.05)
current_densities = np.zeros_like(voltages)

# solve the drift-diffusion problem for every voltage
for i, v in enumerate(voltages):
    print(f'{v:.3f}V', end=' / ')
    ld.apply_voltage(v)
    # store previous solution fluctuation
    fluct = 1  # any value so that the first iteration is performed
    while fluct > 1e-8:
        fluct = ld.transport_step(0.1)
    print(f'{len(ld.fluct)} iterations')
    current_densities[i] = ld.get_current_density().mean() * (-1)

ld.original_units()

plt.figure('J-V curve')
plt.plot(voltages, current_densities, 'b.-')
plt.show()
