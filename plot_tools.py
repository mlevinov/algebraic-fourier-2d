import mpmath as mpm
import numpy as np
import matplotlib.pyplot as plt
import single_jump_param_recovery as sj
import constants as const
import mpmath_tools as mpt
from test_functions import TestFunctions
from tqdm import tqdm


def create_oy_values(oy_strt_val=15, num_of_oy_vals=10, inc_oy=10, reconstruction_order=0):
    end_oy_val = oy_strt_val + (num_of_oy_vals * inc_oy)
    oy_arr = []
    decay_rate = []
    for oy in range(oy_strt_val, end_oy_val, inc_oy):
        oy_arr.append(oy)
        decay_rate.append(pow(oy, -reconstruction_order-1))
    return oy_arr, decay_rate


def approx_coefficients_for_fx_using_psi(x, n_oy, test_func_type, reconstruction_order):
    n_ox = pow(n_oy, 2)
    coeff_for_fx = mpm.matrix(2 * n_oy + 1, 1)
    for oy in range(-n_oy, n_oy + 1):
        exact_coeff_arr = TestFunctions(func_type=test_func_type).get_func_fourier_coefficient_const_oy_range_ox(num_of_oxs=n_ox, oy=oy)
        psi_jump_mag = sj.approximate_jump_magnitudes(reconstruction_order, func_coeff_array=exact_coeff_arr,
                                                      approximated_jump_location=const.PSI_JUMP_LOC, known_jump_loc=True)
        coeff_for_fx[n_oy + oy, 0] = sj.psi_func_val_at_x(x=x, reconstruction_order=reconstruction_order,
                                                          func_coeff_array=exact_coeff_arr, jump_loc=const.PSI_JUMP_LOC,
                                                          jump_mag_array=psi_jump_mag)
    return coeff_for_fx


def get_approxFx_approxJumpLoc_approxJumpMag(x, Y, n_oy, test_func_type, reconstruction_order):
    ny = len(Y)
    approx_fx = mpm.matrix(ny, 1)
    coeff_for_fx = approx_coefficients_for_fx_using_psi(x=x, n_oy=n_oy, test_func_type=test_func_type,
                                                        reconstruction_order=reconstruction_order)
    approx_jump_loc = sj.approximate_jump_location(reconstruction_order=reconstruction_order, func_coeff_array=coeff_for_fx,
                                                   half_order_flag=False, get_omega_flag=False)
    approx_jump_mag = sj.approximate_jump_magnitudes(reconstruction_order=reconstruction_order, func_coeff_array=coeff_for_fx,
                                                     approximated_jump_location=approx_jump_loc, known_jump_loc=False)
    for iy in range(ny):
        y = Y[iy]
        approx_fx[iy, 0] = sj.func_val_at_x(x=y, reconstruction_order=reconstruction_order, func_coeff_array=coeff_for_fx,
                                            jump_loc=approx_jump_loc, jump_mag_array=approx_jump_mag)
    return approx_fx, approx_jump_loc, approx_jump_mag


def get_fxErr_jumpLocErr_jumpMagErr(x, Y, n_oy, test_func_type, reconstruction_order):
    tf = TestFunctions(func_type=test_func_type)
    exactFx = tf.get_func_slice_at_x(x=x, Y=Y)
    exactJumpLoc = tf.get_jump_loc_of_fx(x=x)
    exactJumpMag = tf.get_jump_magnitudes_of_fx(x=x)
    approx_fx, approx_jump_loc, approx_jump_mag = get_approxFx_approxJumpLoc_approxJumpMag(x=x, Y=Y, n_oy=n_oy,
                                                                                           test_func_type=test_func_type,
                                                                                           reconstruction_order=reconstruction_order)
    fx_max_err = mpt.get_max_err_val(exact_vals=exactFx, approx_vals=approx_fx)
    jump_loc_err = mpm.fabs(mpm.fsub(exactJumpLoc, approx_jump_loc))
    jump_mag_err = mpt.elementwise_norm_matrix(exactJumpMag[:approx_jump_mag.rows, 0], approx_jump_mag)

    return fx_max_err, jump_loc_err, jump_mag_err


def get_fxErr_jumpLocErr_jumpMagErr_different_oy_vals(x, Y, oy_strt_val, num_of_oy_vals, inc_oy, test_func_type, reconstruction_order):
    oy_vals_arr, decay_rate = create_oy_values(oy_strt_val=oy_strt_val, num_of_oy_vals=num_of_oy_vals, inc_oy=inc_oy,
                                               reconstruction_order=reconstruction_order)
    n_oy = len(oy_vals_arr)
    fx_max_err_arr = mpm.matrix(n_oy, 1)
    jump_loc_err_arr = mpm.matrix(n_oy, 1)
    jump_mag_err_arr = mpm.matrix(n_oy, reconstruction_order + 1)
    for n in tqdm(range(n_oy)):
        oy = oy_vals_arr[n]
        t = get_fxErr_jumpLocErr_jumpMagErr(x=x, Y=Y, n_oy=oy, test_func_type=test_func_type, reconstruction_order=reconstruction_order)
        fx_max_err_arr[n, 0] = t[0]
        jump_loc_err_arr[n, 0] = t[1]
        jump_mag_err_arr[n, :] = t[2].T[0, :]

    return oy_vals_arr, decay_rate, fx_max_err_arr, jump_loc_err_arr, jump_mag_err_arr


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
