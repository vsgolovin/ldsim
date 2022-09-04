import matplotlib.pyplot as plt
from ldsim.models import LaserDiodeModel2d
from sample_laser import layers


MZ = 3        # number of vertical cross-sections / slices
DN_DZ = -0.2  # refractive index z derivative (1 / cm)

# make band edges functions of z in waveguide and active layers
for layer in layers:
    name = layer.name
    if 'waveguide' in name or 'active' in name:
        layer.update(n_refr=DN_DZ, axis='z')

# construct laser
laser = LaserDiodeModel2d(layers, L=0.2, w=0.01, R1=0.95, R2=0.05, lam=0.87e-4,
                          ng=3.9, alpha_i=0.5, beta_sp=1e-5, num_slices=MZ)

# find waveguide mode profile for every slice
results = laser.solve_waveguide(step=1e-7, n_modes=3, remove_layers=(2, 2))

# plot results
fig = plt.figure()
ax = fig.gca()    # waveguide mode
ax2 = ax.twinx()  # refractive index
for i, r in enumerate(results):
    x = r['x'] * 1e4
    n_eff = r['n_eff']
    label = '#{}, {:.3f} / {:.1f}%'.format(
        i + 1, r['n_eff'][0], r['gammas'][0] * 1e2)
    ax.semilogy(x, r['modes'][:, 0], label=label)
    ax2.plot(x, r['n'], ls=':')
ax.legend(loc='upper right')
ax.set_ylabel('Mode profile [solid lines]')
ax.set_xlabel(r'Coordinate ($\mu$m)')
ax2.set_ylabel('Refractive index [dotted lines]')
plt.show()
