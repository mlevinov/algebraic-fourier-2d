import sys
import mpmath as mpm
import numpy as np
import matplotlib.pyplot as plt
import single_jump_param_recovery as sjpr
from test_functions import TestFunctions
import constants as const
from tqdm import tqdm
from mpmath_tools import mpm_matrix_to_mpmath_numpy_array as mp_mat_to_np_mat
if __name__ == "__main__":
    mpm.mp.dps = 25
    jump_loc = -mpm.pi
    ro = 0
    psi_jump_loc = -const.MP_PI
    tf = TestFunctions(func_type=const.FUNC_TYPE_F2)
    ny = 256
    X = np.linspace(-const.NP_PI, const.NP_PI - const.EPS, ny)
    Y = mpm.matrix(len(X), 1)
    jump_mag = mpm.ones(ro+1,1)
    for ix in tqdm(range(len(X))):
        # Y[ix, 0] = sjpr.__vn_func_val_at_x(x=X[ix], n=4, jump_loc=1)
        Y[ix, 0] = sjpr.phi_func_val_at_x(x=X[ix], reconstruction_order=ro, jump_loc=jump_loc, jump_mag_array=jump_mag)
    ny = mp_mat_to_np_mat(Y)
    plt.plot(X,Y)
    plt.show()




