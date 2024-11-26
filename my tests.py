import sys
import mpmath as mpm
import numpy as np
import matplotlib.pyplot as plt
import single_jump_param_recovery as sj
import constants as const
import mpmath_tools as mpt
from test_functions import TestFunctions
from tqdm import tqdm
import inspect


def create_oy_values(oy_strt_val=15, num_of_oy_vals=10, inc_oy=10):
    end_oy_val = oy_strt_val + (num_of_oy_vals * inc_oy)
    oy_arr = []
    for oy in range(oy_strt_val, end_oy_val, inc_oy):
        oy_arr.append(oy)
    return oy_arr

def approx_coefficients_using_psi_at_x(x, oy, num_of_coeff_for_psi, test_func_type, reconstruction_order):
    psi_jump_loc = -const.MP_PI
    tf = TestFunctions(func_type=test_func_type)
    exact_coeff_arr = tf.get_func_fourier_coefficient_const_oy_range_ox(oy, num_of_coeff_for_psi)
    psi_jump_mag = sj.approximate_jump_magnitudes(reconstruction_order, func_coeff_array=exact_coeff_arr,
                                                  approximated_jump_location=psi_jump_loc, known_jump_loc=True)
    approx_psi_oy_at_x = sj.psi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, func_coeff_array=exact_coeff_arr,
                                              jump_loc=psi_jump_loc, jump_mag_array=psi_jump_mag)
    return approx_psi_oy_at_x


def create_coefficients_for_test_func_at_x(x, oy, num_of_coeff_for_psi, test_func_type, reconstruction_order, coeff_array):
    psi_jump_loc = -const.MP_PI
    tf = TestFunctions(func_type=test_func_type)
    m = num_of_coeff_for_psi
    exact_coeff_arr = tf.get_func_fourier_coefficient_const_oy_range_ox(oy, num_of_coeff_for_psi)
    psi_jump_mag = sj.approximate_jump_magnitudes(reconstruction_order, func_coeff_array=exact_coeff_arr,
                                                  approximated_jump_location=psi_jump_loc, known_jump_loc=True)
    approx_psi_oy_at_x = sj.psi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, func_coeff_array=exact_coeff_arr,
                                              jump_loc=psi_jump_loc, jump_mag_array=psi_jump_mag)
    psi_oy_at_x[moy + oy, 0] = sj.func_val_at_x(x=x, reconstruction_order=ro, func_coeff_array=func_coeff,
                                                jump_loc=psi_jump_loc, jump_mag_array=psi_jump_mag)
    return 0


if __name__ == "__main__":
    mpm.mp.dps = 25
    x = 1.1
    ro = 2
    psi_jump_loc = -const.MP_PI
    tf1 = TestFunctions(func_type=const.FUNC_TYPE_F2)
    moy = 20
    mox = np.power(moy, 2)
    ny = 128
    # Y = np.linspace(-const.NP_PI + const.EPS, const.NP_PI, ny)
    Y = np.linspace(-const.NP_PI, const.NP_PI - const.EPS, ny)
    func_coeff = mpm.matrix(2 * mox + 1, 1)
    psi_oy_at_x = mpm.matrix(2 * moy + 1, 1)
    f_at_x = mpm.matrix(ny, 1)
    for oy in tqdm(range(-moy, moy + 1)):
        for ox in range(-mox, mox + 1):
            func_coeff[mox + ox, 0] = tf1.get_func_fourier_coefficient(ox, oy)

        approx_psi_jump_mag = sj.approximate_jump_magnitudes(reconstruction_order=ro, func_coeff_array=func_coeff,
                                                             approximated_jump_location=psi_jump_loc, known_jump_loc=True)
        psi_oy_at_x[moy + oy, 0] = sj.func_val_at_x(x=x, reconstruction_order=ro, func_coeff_array=func_coeff,
                                                    jump_loc=psi_jump_loc, jump_mag_array=approx_psi_jump_mag)
    approx_f_jump_loc_at_x = sj.approximate_jump_location(reconstruction_order=ro, func_coeff_array=psi_oy_at_x, half_order_flag=False)
    print()
    print('f jump location:')
    print(approx_f_jump_loc_at_x)
    approx_f_jump_mag_at_x = sj.approximate_jump_magnitudes(reconstruction_order=ro, func_coeff_array=psi_oy_at_x,
                                                            approximated_jump_location=approx_f_jump_loc_at_x, known_jump_loc=False)
    print()
    print('f jump mag at x:')
    print(approx_f_jump_mag_at_x)
    print()

    for iy in tqdm(range(ny)):
        f_at_x[iy, 0] = sj.func_val_at_x(Y[iy], reconstruction_order=ro, func_coeff_array=psi_oy_at_x, jump_loc=approx_f_jump_loc_at_x, jump_mag_array=approx_f_jump_mag_at_x)
    YY = mpt.mpm_matrix_to_mpmath_numpy_array(f_at_x)
    plt.plot(Y, np.real(YY[:, 0]))
    plt.show()
