"""
Tests for `ldsim.preprocessing.design`
"""

import numpy as np
from ldsim.preprocessing.design import CustomLayer, LaserDiode


def test_layer_Eg():
    layer = CustomLayer('test', 0.2)
    layer.update(dict(Ev=0.0, Ec=[3., 1.2]))
    layer.update(dict(Ev=[-1.0]), axis='z')
    x, z = np.meshgrid(np.array([0, 0.1, 0.2]), np.array([0., 1.0]))
    Eg = layer.calculate('Eg', x, z)
    Eg_0 = layer.calculate('Eg', x)
    ans = np.array([[1.2, 1.5, 1.8], [2.2, 2.5, 2.8]])
    assert np.allclose(Eg, ans) and np.allclose(Eg_0, ans[0])


def test_layer_Cdop():
    layer = CustomLayer('test', 2e-4)
    layer.update(dict(Nd=0.0, Na=[-4e21, 1e18]), axis='x')
    layer.update(dict(Nd=1e14), axis='z')
    x, z = np.meshgrid(np.array([0, 1e-4, 2e-4]), np.array([0., 1e3, 2e3]))
    C_dop = layer.calculate('C_dop', x, z)
    C_dop_0 = layer.calculate('C_dop', x)
    ans = np.array([[1e18, 6e17, 2e17],
                    [9e17, 5e17, 1e17],
                    [8e17, 4e17, 0]]) * (-1)
    assert np.allclose(C_dop, ans) and np.allclose(C_dop_0, ans[0])


def test_laserdiode_boundaries():
    layer_mid = CustomLayer('a', 1.0)
    layer_top = CustomLayer('b', 2.0)
    layer_bot = CustomLayer('c', 3.0)
    stack = [layer_bot, layer_mid, layer_top]
    ld = LaserDiode(stack, L=0.1, w=0.01, R1=0.5, R2=0.2, lam=0.87e-4, ng=3.9,
                    alpha_i=0.5, beta_sp=1e-5)
    boundaries = ld.get_boundaries()
    ans = np.array([0.0, 3.0, 4.0, 6.0])
    assert np.allclose(boundaries, ans)


def test_laserdiode_calculate():
    p_layer = CustomLayer('P', 1e-4)
    p_layer.update(dict(Ec=[-2e3, 1.9], Ev=[1e3, 0.0]))
    i_layer = CustomLayer('i', 0.7e-4)
    i_layer.update(dict(Ec=1.6, Ev=0.2))
    n_layer = CustomLayer('N', 2e-4)
    n_layer.update(dict(Ec=2.0, Ev=[-1e3, 0.1]))
    ld = LaserDiode([p_layer, i_layer, n_layer], 0.1, 0.01, 0.3, 0.3, 0.87e-4,
                    3.9, 0.5, 1e-5)
    x = np.array([0, 0.9, 1.1, 1.6, 1.8, 2.0, 3.6]) * 1e-4
    y = ld.calculate('Eg', x)
    y_ans = np.array([1.9, 1.63, 1.4, 1.4, 1.91, 1.93, 2.09])
    assert np.allclose(y, y_ans)
