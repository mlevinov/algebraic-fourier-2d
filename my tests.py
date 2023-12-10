import sys
import mpmath as mpm
import numpy as np
import matplotlib.pyplot as plt
import single_jump_param_recovery as sj
import constants as const
import mpmath_tools as mpt
from test_functions import TestFunctions
from tqdm import tqdm


if __name__ == "__main__":
    x = 2
    ro = 1
    tf1 = TestFunctions(func_type=const.FUNC_TYPE_F3)
    moy = 16
    mox = moy**2
    nx = mox
    ny = 256
    Y = np.linspace(-const.NP_PI + const.EPS, const.NP_PI, ny)
    func_coeff = mpm.matrix(2 * mox + 1, 1)
    psi_oy_at_x = mpm.matrix(2 * moy + 1, 1)
    f_at_x = mpm.matrix(ny, 1)
    for oy in tqdm(range (-moy, moy + 1)):
        for ox in range(-mox, mox + 1):
            func_coeff[mox + ox, 0] = tf1.get_func_fourier_coefficient(ox, oy)
        approx_psi_jump_mag = sj.approximate_jump_magnitudes(reconstruction_order=ro, func_coeff_array=func_coeff, approximated_jump_location=mpm.pi)
        psi_oy_at_x[oy + moy, 0] = sj.func_val_at_x(x=x, reconstruction_order=ro, func_coeff_array=func_coeff, jump_loc=mpm.pi, jump_mag_array=approx_psi_jump_mag)

    f_coeff = psi_oy_at_x[:, 0]
    approx_f_jump_loc_at_x = sj.approximate_jump_location(reconstruction_order=ro, func_coeff_array=f_coeff, half_order_flag=False)
    print(approx_f_jump_loc_at_x)
    approx_f_jump_mag_at_x = sj.approximate_jump_magnitudes(reconstruction_order=ro, func_coeff_array=f_coeff, approximated_jump_location=approx_f_jump_loc_at_x)
    for iy in tqdm(range(ny)):
        f_at_x[iy, 0] = sj.func_val_at_x(Y[iy], reconstruction_order=ro, func_coeff_array=f_coeff, jump_loc=approx_f_jump_loc_at_x, jump_mag_array=approx_f_jump_mag_at_x)
    YY = mpt.mpm_matrix_to_mpmath_numpy_array(f_at_x)
    plt.plot(Y, np.real(YY[:, 0]))
    plt.show()