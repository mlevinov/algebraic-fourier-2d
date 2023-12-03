import sys
import mpmath as mpm
import numpy as np
import matplotlib.pyplot as plt
import single_jump_param_recovery as sjpr
from test_functions import TestFunctions
import constants as const

if __name__ == "__main__":
    X = np.linspace(-const.NP_PI, const.NP_PI - const.EPS)
    nx = len(X)
    tf_type_1 = TestFunctions(dps=5, func_type=const.FUNC_TYPE_F2)

    reconstruction_order = 2
    jump_mag = mpm.matrix(1, reconstruction_order + 1)
    Y = mpm.matrix(1, nx)

    for i in range(reconstruction_order + 1):
        jump_mag[0, i] = 1

    for ix in range(nx):
        Y[0, ix] = sjpr.phi_func_val_at_x(x=X[ix], reconstruction_order=reconstruction_order,
                                   jump_loc=const.DEFAULT_JUMP_LOC, jump_mag_array=jump_mag)


