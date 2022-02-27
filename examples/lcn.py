import matplotlib.pyplot as plt
from ldsim.models import LaserDiodeModel1d
from sample_laser import layers


ld = LaserDiodeModel1d(layers, L=0.3, w=0.01, R1=0.95, R2=0.05, lam=0.87e-4,
                       ng=3.9, alpha_i=0.5, beta_sp=1e-5)
ld.generate_nonuniform_mesh(y_ext=[0., 0.])
ld.make_dimensionless()
ld.solve_lcn()
ld.original_units()

plt.rc('lines', linewidth=1.0)
x = ld.xn * 1e4
plt.figure('Local charge neutrality')
plt.plot(x, ld.vn['Ec'] - ld.vn['psi_lcn'], 'k-')
plt.plot(x, ld.vn['Ev'] - ld.vn['psi_lcn'], 'k-')
plt.plot([x[0], x[-1]], [0.0, 0.0], 'k:')
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel('$E$ (eV)')
plt.twinx()
plt.plot(x, ld.vn['n0'], 'b:')
plt.plot(x, ld.vn['p0'], 'r:')
plt.ylabel('$n$, $p$ (cm$^{-3}$)')
plt.show()
