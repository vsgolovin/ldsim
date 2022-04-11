import matplotlib.pyplot as plt
from ldsim.models import LaserDiodeModel2d
from sample_laser import layers


# initialize model
ld = LaserDiodeModel2d(layers, L=0.3, w=0.01, R1=0.95, R2=0.05, lam=0.87e-4,
                       ng=3.9, alpha_i=0.5, beta_sp=1e-5)
ld.generate_nonuniform_mesh(method='mean', y_ext=[0., 0.])

Eg = ld.calculate('Eg', ld.xn, ld.zn)
plt.figure()
cn = plt.contourf(ld.zn * 1e4, ld.xn * 1e4, ld.vn['Ec'], cmap=plt.cm.coolwarm)
plt.colorbar(cn, label='$E_c$ (eV)')
plt.xlabel(r'$z$ ($\mu$m)')
plt.ylabel(r'$x$ ($\mu$m)')
plt.show()
