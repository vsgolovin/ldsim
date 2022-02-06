"""
Tests for `ldsim.preprocessing.design`
"""

import numpy as np
from ldsim.preprocessing.design import Layer


def test_layer_Eg():
    layer = Layer('test', 0.2)
    layer.update(dict(Ev=0.0, Ec=[3., 1.2]))
    layer.update(dict(Ev=[-1.0]), axis='z')
    x, z = np.meshgrid(np.array([0, 0.1, 0.2]), np.array([0., 1.0]))
    Eg = layer.calculate('Eg', x, z)
    Eg_0 = layer.calculate('Eg', x)
    ans = np.array([[1.2, 1.5, 1.8], [2.2, 2.5, 2.8]])
    assert np.allclose(Eg, ans) and np.allclose(Eg_0, ans[0])


def test_layer_Cdop():
    layer = Layer('test', 2e-4)
    layer.update(dict(Nd=0.0, Na=[-4e21, 1e18]), axis='x')
    layer.update(dict(Nd=1e14), axis='z')
    x, z = np.meshgrid(np.array([0, 1e-4, 2e-4]), np.array([0., 1e3, 2e3]))
    C_dop = layer.calculate('C_dop', x, z)
    C_dop_0 = layer.calculate('C_dop', x)
    ans = np.array([[1e18, 6e17, 2e17],
                    [9e17, 5e17, 1e17],
                    [8e17, 4e17, 0    ]]) * (-1)
    assert np.allclose(C_dop, ans) and np.allclose(C_dop_0, ans[0])
