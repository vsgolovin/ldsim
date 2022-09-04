import numpy as np
import ldsim.constants as const
from ldsim.materials import Material


def bandgap(x, T=300.):
    Eg_gamma = 1.519 + 1.155 * x + 0.37 * x**2 - 5.41e-4 * T**2 / (T + 204.)
    Eg_x = 1.981 + 0.124 * x + 0.144 * x**2 - 4.6e-4 * T**2 / (T + 204.)
    return np.minimum(Eg_gamma, Eg_x)

def dEc_dEg(x):
    return np.piecewise(x, [x <= 0.45, x > 0.45],
                        [0.6, lambda t: 0.6 - (t - 0.45) * 0.4])

def conduction_band_bottom(x, T=300.):
    Eg_base = bandgap(0., 300.)
    return Eg_base + (bandgap(x, T) - Eg_base) * dEc_dEg(x)

def valence_band_top(x, T=300.):
    Eg_base = bandgap(0., 300.)
    return 0.0 - (bandgap(x, T) - Eg_base) * (1 - dEc_dEg(x))

def electron_dos_effective_mass(x):
    return np.piecewise(x, [x < 0.45, x >= 0.45],
                        [lambda t: 0.063 + 0.083 * t,
                         lambda t: 0.85 - 0.14 * t])

def hole_dos_effective_mass(x):
    return 0.51 + 0.25 * x

def cb_effective_dos(x, T=300.):
    mn = electron_dos_effective_mass(x)
    return 4.82e15 * (mn * T)**1.5

def vb_effective_dos(x, T=300.):
    mh = hole_dos_effective_mass(x)
    return 4.82e15 * (mh * T)**1.5

def electron_mobility(x):
    p_direct = np.array([1e4, -2.2e4, 8e3])
    p_indirect = np.array([-720., 1160., -255])
    return np.piecewise(x, [x <= 0.45, x > 0.45],
                        [lambda t: np.polyval(p_direct, t),
                         lambda t: np.polyval(p_indirect, t)])

def hole_mobility(x):
    return 370. - 970 * x + 740. * x**2

def srh_electron_lifetime():
    return 5e-9

def srh_hole_lifetime():
    return 5e-9

def radiative_recombination_coeff():
    return 1.8e-10

def auger_coefficient_n(x):
    return np.piecewise(
        x,
        [(x >= 0) & (x < 0.1),
         (x >= 0.1) & (x < 0.2),
         x >= 0.2],
        [lambda t: 1.9e-31 - 7e-31 * t,
         lambda t: 1.2e-31 - 5e-31 * (t - 0.1),
         lambda t: 0.7e-31 + t * 0.0]
    )

def auger_coefficient_p(x):
    return np.piecewise(
        x,
        [(x >= 0) & (x < 0.1),
         (x >= 0.1) & (x < 0.2),
         x >= 0.2],
        [lambda t: 12e-31 - 35e-31 * t,
         lambda t: 8.5e-31 - 24e-31 * (t - 0.1),
         lambda t: 6.1e-31 + t * 0.0]
    )

def permittivity(x):
    return 12.9 - 2.84 * x

def refractive_index(x, wavelength: 1e-4):
    """
    AlGaAs refractive index at 300K.

    Reference: S. Adachi, Journal of Applied Physics, 58, R1 (1985).
    doi: 10.1063/1.336070
    """
    def f(chi):
        return (2 - np.sqrt(1 + chi) - np.sqrt(1 - chi)) / chi**2

    a0 = 6.3 + 19. * x
    b0 = 9.4 - 10.2 * x
    E0 = 1.425 + 1.155 * x + 0.37 * x**2         # different from `bandgap`
    E0_delta0 = 1.765 + 1.115 * x + 0.37 * x**2  # + spin-orbit splitting
    E_ph = const.h * const.c / (wavelength * const.q)
    chi = E_ph / E0
    chi_so = E_ph / E0_delta0
    eps = a0 * (f(chi) + f(chi_so) / 2. * (E0 / E0_delta0)**1.5) + b0
    return np.sqrt(np.real(eps))

def free_carrier_absorption_electrons():
    return 4e-18

def free_carrier_absorption_holes():
    return 12e-18


class AlGaAs(Material):
    def __init__(self, name='AlGaAs', **kwargs):
        kwargs['T'] = kwargs.get('T', 300.0)  # set default temperature
        super().__init__(name=name, args=('x', 'T', 'wavelength'), **kwargs)
        self.set_params(
            Eg=bandgap,
            Ec=conduction_band_bottom,
            Ev=valence_band_top,
            Nc=cb_effective_dos,
            Nv=vb_effective_dos,
            mu_n=electron_mobility,
            mu_p=hole_mobility,
            tau_n=srh_electron_lifetime,
            tau_p=srh_hole_lifetime,
            B=radiative_recombination_coeff,
            Cn=auger_coefficient_n,
            Cp=auger_coefficient_p,
            eps=permittivity,
            n_refr=refractive_index,
            fca_e=free_carrier_absorption_electrons,
            fca_h=free_carrier_absorption_holes
        )


if __name__ == '__main__':
    def print_value(func, label, *args, **kwargs):
        print(f'{label} = {func(*args, **kwargs)}')

    x = np.array([0.0, 0.25, 0.4, 0.55])
    print(f'x = {x}')
    print_value(bandgap, 'Eg', x, T=300.)
    print_value(cb_effective_dos, 'Nc', x)
    print_value(vb_effective_dos, 'Nv', x)
    print_value(electron_mobility, 'mu_n', x)
    print_value(hole_mobility, 'mu_h', x)
    print_value(srh_electron_lifetime, 'tau_n')
    print_value(srh_hole_lifetime, 'tau_p')
    print_value(radiative_recombination_coeff, 'B')
    print_value(auger_coefficient_n, 'Cn', x)
    print_value(auger_coefficient_p, 'Cp', x)
    print_value(permittivity, 'eps', x)
    print_value(refractive_index, 'n_refr', x, wavelength=0.875e-4)
    print_value(free_carrier_absorption_electrons, 'fca_e')
    print_value(free_carrier_absorption_holes, 'fca_h')
