import numpy as np
import matplotlib.pyplot as plt
from sample_laser import laser
from ldsim import units


# find waveguide mode profiles
results = laser.solve_waveguide(step=1e-7, n_modes=3, remove_layers=(2, 2))

# plot mode profiles
plt.figure()
x = results['x'] * 1e4  # cm -> um
for i in range(len(results['gammas'])):
    n = results['n_eff'][i]
    gamma = results['gammas'][i] * 100
    label = r'$n_{eff}$' + f' = {n:.2f}'
    label = label + r', $\Gamma$' + f' = {gamma:.2f}%'
    mode = results['modes'][:, i] * 1e-4
    plt.plot(x, mode, label=label)
plt.ylabel('Mode profile')
plt.legend()

# plot refractive index profile
plt.twinx()
plt.plot(x, results['n'], 'k:')
plt.ylabel('Refractive index')
plt.xlabel(r'Coordinate ($\mu$m)')
# plt.show()


xmax = laser.get_thickness()
x = np.linspace(0, xmax, 200) / units.x
y = laser.get_waveguide_mode(x, dimensionless=True)
plt.figure()
plt.plot(x, y)
plt.show()
