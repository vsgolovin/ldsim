"""
Tests for `ldsim.materials`
"""

from ldsim.materials import Material


def test_overwrite():
    """
    Checks if overwriting a parameter works correctly.
    Also tests `Material.calculate_all` method.
    """
    def func_1(a, b):
        return a + b
    def func_2(b, a):
        return 10 * a - b
    def func_3(x):
        return x - 42
    def func_4(b):
        return b**2

    mat = Material('generic name', ('a', 'b', 'x'))
    mat.set_param('A', func_1)
    mat.set_param('B', func_2)
    mat.set_param('C', func_3)
    mat.set_param('A', func_4)

    ans_args = {'A': ('b',), 'B': ('b', 'a'), 'C': ('x',)}
    ans_params = {'a': ['B'], 'b': ['B', 'A'], 'x': ['C']}
    ans_rv = {'A': 16, 'B': 104, 'C': -21}

    assert (mat.params == ans_params and mat.args == ans_args
            and mat.calculate_all(a=10, b=-4, x=21) == ans_rv)
