"""
Classes for defining laser diode vertical (epitaxial) and lateral design.
"""

import numpy as np


params = ('Ev', 'Ec', 'Nd', 'Na', 'Nc', 'Nv', 'mu_n', 'mu_p', 'tau_n',
          'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr', 'Eg', 'C_dop',
          'fca_e', 'fca_h')
params_active = ('g0', 'N_tr')


class Layer:
    def __init__(self, name, thickness, active=False):
        """
        Semiconductor diode layer.

        Parameters
        ----------
            name : str
                Layer name.
            thickness : float
                Layer thickness (cm).
            active : bool
                Whether layer is a part of active region, i.e. takes part
                in stimulated light emission.

        """
        self.name = name
        self.dx = thickness
        self.active = active

        # initialize `dct_x` and `dct_z`, that store spatial dependencies of all
        # the parameters as lists of polynomial coefficients
        self.dct_x = dict.fromkeys(params, [np.nan])
        self.dct_x['C_dop'] = self.dct_x['Nd'] = self.dct_x['Na'] = [0.0]
        self.dct_z = dict.fromkeys(params, [1.0])
        self.dct_z['C_dop'] = self.dct_z['Nd'] = self.dct_z['Na'] = [1.0]
        if active:
            self.dct_x.update(dict.fromkeys(params_active, [np.nan]))
            self.dct_z.update(dict.fromkeys(params_active, [1.0]))

    def __repr__(self):
        s1 = 'Layer \"{}\"'.format(self.name)
        s2 = '{} um'.format(self.dx * 1e4)
        Cdop = self.calculate('C_dop', [0, self.dx])
        if Cdop[0] > 0 and Cdop[1] > 0:
            s3 = 'n-type'
        elif Cdop[0] < 0 and Cdop[1] < 0:
            s3 = 'p-type'
        else:
            s3 = 'i-type'
        if self.active:
            s4 = 'active'
        else:
            s4 = 'not active'
        return ' / '.join((s1, s2, s3, s4))

    def __str__(self):
        return self.name

    def _choose_dict(self, axis):
        if axis == 'x':
            return self.dct_x
        elif axis == 'z':
            return self.dct_z
        raise ValueError(f'Unknown axis {axis}.')

    def calculate(self, param, x, z=0.0):
        "Calculate value of parameter `param` at location (`x`, `z`)."
        p_x = self.dct_x[param]
        p_z = self.dct_z[param]
        return np.polyval(p_x, x) * np.polyval(p_z, z)

    def update(self, d, axis='x'):
        "Update polynomial coefficients of parameters."
        # check input
        assert isinstance(d, dict)
        dct = self._choose_dict(axis)

        # update dictionary values
        for k, v in d.items():
            if k not in self.d:
                raise Exception(f'Unknown parameter {k}')
            if isinstance(v, (int, float)):
                dct[k] = [v]
            else:
                dct[k] = v
        if 'Ec' in d or 'Ev' in d:
            self._update_Eg(axis)
        if 'Nd' in d or 'Na' in d:
            self._update_Cdop(axis)

    def _update_Eg(self, axis):
        dct = self._choose_dict(axis)
        p_Ec = np.asarray(dct['Ec'])
        p_Ev = np.asarray(dct['Ev'])
        delta = len(p_Ec) - len(p_Ev)
        if delta > 0:
            p_Ev = np.concatenate([np.zeros(delta), p_Ev])
        elif delta < 0:
            p_Ec = np.concatenate([np.zeros(-delta), p_Ec])
        dct['Eg'] = p_Ec - p_Ev

    def _update_Cdop(self, axis):
        dct = self._choose_dict(axis)
        p_Nd = np.asarray(dct['Nd'])
        p_Na = np.asarray(dct['Na'])
        delta = len(p_Nd) - len(p_Na)
        if delta > 0:
            p_Na = np.concatenate([np.zeros(delta), p_Na])
        elif delta < 0:
            p_Nd = np.concatenate([np.zeros(-delta), p_Nd])
        dct['C_dop'] = p_Nd - p_Na

    def make_gradient_layer(self, other, name, thickness, active=False, deg=1):
        """
        Create a layer where all parameters gradually change from their
        endvalues in the current layer to values in `other` at x = 0.
        By default all parameters change linearly, this can be change by
        increasing polynomial degree `deg`.
        """
        layer_new = Layer(name, thickness, active=active)
        x = np.array([0, thickness])
        y = np.zeros(2)
        for key in self.d:
            y[0] = self.calculate(key, self.dx)
            y[1] = other.calculate(key, 0)
            p = np.polyfit(x=x, y=y, deg=deg)
            layer_new.update({key: p}, axis='x')
        layer_new.update(self.dct_z, axis='z')  # same f(z) as in current layer
        return layer_new


class EpiDesign(list):
    "A list of `Layer` objects."
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def boundaries(self):
        "Get an array of layer boundaries."
        return np.cumsum([0.0] + [layer.dx for layer in self])

    def get_thickness(self):
        "Get sum of all layers' thicknesses."
        return self.boundaries()[-1]

    def _ind_dx(self, x):
        if x == 0:
            return 0, 0.0
        xi = 0.0
        for i, layer in enumerate(self):
            if x <= (xi + layer.dx):
                return i, x - xi
            xi += layer.dx
        return np.nan, np.nan

    def _inds_dx(self, x):
        inds = np.zeros_like(x)
        dx = np.zeros_like(x)
        for i, xi in enumerate(x):
            inds[i], dx[i] = self._ind_dx(xi)
        return inds, dx

    def calculate(self, param, x, inds=None, dx=None):
        "Calculate values of `param` at locations `x`."
        y = np.zeros_like(x)
        if isinstance(x, (float, int)):
            ind, dx = self._ind_dx(x)
            return self[ind].calculate(param, dx)
        else:
            if inds is None or dx is None:
                inds, dx = self._inds_dx(x)
            for i, layer in enumerate(self):
                ix = (inds == i)
                if ix.any():
                    y[ix] = layer.calculate(param, dx[ix])
        return y
