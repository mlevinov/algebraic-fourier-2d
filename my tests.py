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
    mpm.mp.dps = 25
    x = -1.25
    ro = 2
    psi_jump_loc = -const.MP_PI
    tf1 = TestFunctions(func_type=const.FUNC_TYPE_F2)
    moy = 16
    mox = np.power(moy, 2)
    ny = 256
    Y = np.linspace(-const.NP_PI + const.EPS, const.NP_PI, ny)
    # Y = np.linspace(-const.NP_PI , const.NP_PI - const.EPS, ny)
    func_coeff = mpm.matrix(2 * mox + 1, 1)
    psi_oy_at_x = mpm.matrix(2 * moy + 1, 1)
    f_at_x = mpm.matrix(ny, 1)
    for oy in tqdm(range (-moy, moy + 1)):
        for ox in range(-mox, mox + 1):
            func_coeff[mox + ox, 0] = tf1.get_func_fourier_coefficient(ox, oy)

        approx_psi_jump_mag = sj.approximate_jump_magnitudes(reconstruction_order=ro, func_coeff_array=func_coeff,
                                                             approximated_jump_location=psi_jump_loc, known_jump_loc=True)
        psi_oy_at_x[moy + oy, 0] = sj.func_val_at_x(x=x, reconstruction_order=ro, func_coeff_array=func_coeff,
                                                    jump_loc=psi_jump_loc, jump_mag_array=approx_psi_jump_mag)
    approx_f_jump_loc_at_x = sj.approximate_jump_location(reconstruction_order=ro, func_coeff_array=psi_oy_at_x, half_order_flag=False)
    print(approx_f_jump_loc_at_x)
    approx_f_jump_mag_at_x = sj.approximate_jump_magnitudes(reconstruction_order=ro, func_coeff_array=psi_oy_at_x, approximated_jump_location=approx_f_jump_loc_at_x)
    for iy in tqdm(range(ny)):
        f_at_x[iy, 0] = sj.func_val_at_x(Y[iy], reconstruction_order=ro, func_coeff_array=psi_oy_at_x, jump_loc=approx_f_jump_loc_at_x, jump_mag_array=approx_f_jump_mag_at_x)
    YY = mpt.mpm_matrix_to_mpmath_numpy_array(f_at_x)
    plt.plot(Y, np.real(YY[:, 0]))
    plt.show()