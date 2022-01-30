"""
Classes for defining laser diode vertical (epitaxial) and lateral design.
"""

import numpy as np


params = ['Ev', 'Ec', 'Nd', 'Na', 'Nc', 'Nv', 'mu_n', 'mu_p', 'tau_n',
          'tau_p', 'B', 'Cn', 'Cp', 'eps', 'n_refr', 'Eg', 'C_dop',
          'fca_e', 'fca_h']
params_active = ['g0', 'N_tr']


class Layer(object):
    def __init__(self, name, dx, active=False):
        """
        Parameters:
            name : str
                Layer name.
            dx : float
                Layer thickness (cm).
            active : bool
                Whether layer is part of laser diode active region.
        """
        self.name = name
        self.dx = dx
        self.active = active
        self.d = dict.fromkeys(params, [np.nan])
        self.d['C_dop'] = self.d['Nd'] = self.d['Na'] = [0.0]
        if active:
            self.d.update(dict.fromkeys(params_active, [np.nan]))

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

    def __eq__(self, other_name):
        assert isinstance(other_name, str)
        return self.name == other_name

    def calculate(self, param, x):
        "Calculate value of parameter `param` at location `x`."
        p = self.d[param]
        return np.polyval(p, x)

    def update(self, d):
        "Update polynomial coefficients of parameters."
        assert isinstance(d, dict)
        for k, v in d.items():
            if k not in self.d:
                raise Exception(f'Unknown parameter {k}')
            if isinstance(v, (int, float)):
                self.d[k] = [v]
            else:
                self.d[k] = v
        if 'Ec' in d or 'Ev' in d:
            self._update_Eg()
        if 'Nd' in d or 'Na' in d:
            self._update_Cdop()

    def _update_Eg(self):
        p_Ec = np.asarray(self.d['Ec'])
        p_Ev = np.asarray(self.d['Ev'])
        delta = len(p_Ec) - len(p_Ev)
        if delta > 0:
            p_Ev = np.concatenate([np.zeros(delta), p_Ev])
        elif delta < 0:
            p_Ec = np.concatenate([np.zeros(-delta), p_Ec])
        self.d['Eg'] = p_Ec - p_Ev

    def _update_Cdop(self):
        p_Nd = np.asarray(self.d['Nd'])
        p_Na = np.asarray(self.d['Na'])
        delta = len(p_Nd) - len(p_Na)
        if delta > 0:
            p_Na = np.concatenate([np.zeros(delta), p_Na])
        elif delta < 0:
            p_Nd = np.concatenate([np.zeros(-delta), p_Nd])
        self.d['C_dop'] = p_Nd - p_Na

    def make_gradient_layer(self, l2, name, dx, active=False, deg=1):
        """
        Create a layer where all parameters gradually change from their
        endvalues in the current layer to values in `l2` at x = 0.
        By default all parameters change linearly, this can be change by
        increasing polynomial degree `deg`.
        """
        lnew = Layer(name=name, dx=dx, active=active)
        x = np.array([0, dx])
        y = np.zeros(2)
        for key in self.d:
            y[0] = self.calculate(key, self.dx)
            y[1] = l2.calculate(key, 0)
            p = np.polyfit(x=x, y=y, deg=deg)
            lnew.update({key: p})
        return lnew


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
