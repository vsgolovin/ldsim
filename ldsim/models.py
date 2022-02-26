import numpy as np
from ldsim.preprocessing.design import LaserDiode
from ldsim.mesh import generate_nonuniform_mesh
from ldsim import units


class LaserDiodeModel1d(LaserDiode):
    input_params_nodes = [
        'Ev', 'Ec', 'Eg', 'Nd', 'Na', 'C_dop', 'Nc', 'Nv', 'mu_n', 'mu_p',
        'tau_n', 'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr', 'g0', 'N_tr',
        'fca_e', 'fca_h', 'T']
    input_params_boundaries = ['Ev', 'Ec', 'Nc', 'Nv', 'mu_n', 'mu_p']
    params_active = ['g0', 'N_tr']
    solution_arrays = ['psi', 'phi_n', 'phi_p']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # mesh
        self.xn = None     # nodes
        self.xb = None     # volume boundaries
        self.ar_ix = None  # active region mask for xn

        # parameters at mesh nodes and volume boundaries
        self.vn = dict.fromkeys(self.input_params_nodes)
        self.vb = dict.fromkeys(self.input_params_boundaries)
        self.sol = dict.fromkeys(self.solution_arrays)
        self.S = 1e-12

    def generate_nonuniform_mesh(self, step_uni=5e-8, step_min=1e-7,
                                 step_max=20e-7, sigma=100e-7,
                                 y_ext=[None, None]):
        """
        Generate nonuniform mesh in which step size is proportional to local
        change in bandgap (Eg). Initially creates uniform mesh with step size
        `step_uni`. See `ldsim.mesh.generate_nonuiform_mesh` for detailed
        description.
        """
        assert not self.is_dimensionless
        thickness = self.get_thickness()
        num_points = int(round(thickness // step_uni))
        x = np.linspace(0, thickness, num_points)
        y = self.calculate('Eg', x)
        self.xn = generate_nonuniform_mesh(x, y, step_min, step_max,
                                           sigma, y_ext)[0]
        self.xb = (self.xn[1:] + self.xn[:-1]) / 2
        self.ar_ix = self._get_ar_mask(self.xn)
        self._calculate_all_params()

    def _calculate_all_params(self):
        """
        Calculate values of all parameters at mesh nodes and boundaries.
        """
        # nodes
        inds, dx = self._inds_dx(self.xn)
        for param in self.vn:
            if param in self.params_active:
                continue
            self.vn[param] = self.calculate(
                param, self.xn, z=0, inds=inds, dx=dx)
        # active region
        inds, dx = inds[self.ar_ix], dx[self.ar_ix]
        for param in self.params_active:
            self.vn[param] = self.calculate(
                param, self.xn[self.ar_ix], z=0, inds=inds, dx=dx)

        # boundaries
        inds, dx = self._inds_dx(self.xb)
        for param in self.vb:
            self.vb[param] = self.calculate(
                param, self.xb, z=0, inds=inds, dx=dx)

    def make_dimensionless(self):
        "Make every parameter dimensionless."
        for param in self.vn:
            self.vn[param] /= units.dct[param]
        for param in self.vb:
            self.vb[param] /= units.dct[param]
        return super().make_dimensionless()

    def original_units(self):
        "Convert all values back to original units."
        for param in self.vn:
            self.vn[param] *= units.dct[param]
        for param in self.vb:
            self.vb[param] *= units.dct[param]
        return super().original_units()
