import numpy as np
from scipy.interpolate import interp1d


def generate_nonuniform_mesh(x, y, step_min=1e-7, step_max=20e-7,
                             sigma=100e-7, y_ext=[None, None]):
    """
    Use uniform 1D mesh `x` and some parameter values at these mesh points `y`
    to generate nonuniform mesh. Minimizes step size at intervals with largest
    `y` discontinuities.

    Parameters
    ----------
    x : np.ndarray
        Uniform mesh along the axis.
    y : np.ndarray
        Parameter values at `x`.
    step_min : float
        Minimum step size (spacing) between mesh nodes.
    step_max : float
        Maximum step size.
    sigma : float
        Standard deviation of the gaussian used for smoothing the delta_y(x)
        curve, where delta_y is the discontinuity of `y` (or, equivalently,
        the finite difference approximation of dy/dx derivative).
    y_ext : [Float, Float] or [None, None]
        The values of `y` on the boundaries (outside `x`). This may be useful
        to ensure small mesh step size near the boundaries. If `None` use the
        same value as inside, so there is no discontinuity on the boundaries.

    Returns
    -------
    xn : np.ndarray
        Generated mesh nodes.
    delta_y : np.ndarray
        Smooth `y` discontinuity values.

    """
    # add external values to y
    if not isinstance(y_ext[0], (float, int)):
        y_ext[0] = y[0]
    if not isinstance(y_ext[1], (float, int)):
        y_ext[1] = y[-1]

    # absolute value of change in y
    f = np.zeros_like(y)
    f[1:-1] = np.abs(y[2:] - y[:-2])
    f[0] = abs(y[1] - y_ext[0])
    f[-1] = abs(y_ext[1] - y[-2])

    # Gauss function (normal distribution) for smoothing
    g = np.zeros(len(f) * 2)
    i1 = len(f) // 2
    i2 = i1 + len(f)
    g[i1:i2] = np.exp(-(x - x[i1])**2 / (2 * sigma**2))

    # perform convolution for smoothing
    fg = np.zeros_like(f)
    for i in range(len(fg)):
        fg[i] = np.sum(f * g[len(f)-i:len(f)*2-i])
    fg_fun = interp1d(x, fg / fg.max())

    # generate new grid
    new_grid = []
    xi = 0
    while xi <= x[-1]:
        new_grid.append(xi)
        xi += step_min + (step_max - step_min) * (1 - fg_fun(xi))
    xn = np.array(new_grid)
    delta_y = fg_fun(xn)
    return xn, delta_y
