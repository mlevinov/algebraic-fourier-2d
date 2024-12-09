import mpmath as mpm
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
    trunc_fourier_sum = TestFunctions(func_type=test_func_type).compute_1d_fourier_of_fx(x=x, Y=Y, M=pow(n_oy, 2), N=n_oy)
    return approx_fx, approx_jump_loc, approx_jump_mag, trunc_fourier_sum


def get_fxErr_jumpLocErr_jumpMagErr(x, Y, n_oy, test_func_type, reconstruction_order):
    tf = TestFunctions(func_type=test_func_type)
    exactFx = tf.get_func_slice_at_x(x=x, Y=Y)
    exactJumpLoc = tf.get_jump_loc_of_fx(x=x)
    exactJumpMag = tf.get_jump_magnitudes_of_fx(x=x)
    tpl = get_approxFx_approxJumpLoc_approxJumpMag(x=x, Y=Y, n_oy=n_oy, test_func_type=test_func_type,
                                                   reconstruction_order=reconstruction_order)
    fx_max_err = mpt.get_max_err_val(exact_vals=exactFx, approx_vals=tpl[0])
    jump_loc_err = mpm.fabs(mpm.fsub(exactJumpLoc, tpl[1]))
    jump_mag_err = mpt.elementwise_norm_matrix(exactJumpMag[:tpl[2].rows, 0], tpl[2])
    trunc_fourier_sum_max_err = mpt.get_max_err_val(exact_vals=exactFx, approx_vals=mpt.numpy_array_to_mpmath_matrix(tpl[3]))

    return fx_max_err, jump_loc_err, jump_mag_err, trunc_fourier_sum_max_err


def get_fxErr_jumpLocErr_jumpMagErr_truncFourierSumErr_different_oy_vals(x, Y, oy_strt_val, num_of_oy_vals, inc_oy, test_func_type, reconstruction_order):
    oy_vals_arr, decay_rate = create_oy_values(oy_strt_val=oy_strt_val, num_of_oy_vals=num_of_oy_vals, inc_oy=inc_oy,
                                               reconstruction_order=reconstruction_order)
    n_oy = len(oy_vals_arr)
    trunc_fourier_fx_max_err = mpm.matrix(n_oy, 1)
    fx_max_err_arr = mpm.matrix(n_oy, 1)
    jump_loc_err_arr = mpm.matrix(n_oy, 1)
    jump_mag_err_arr = mpm.matrix(n_oy, reconstruction_order + 1)
    for n in tqdm(range(n_oy)):
        oy = oy_vals_arr[n]
        t = get_fxErr_jumpLocErr_jumpMagErr(x=x, Y=Y, n_oy=oy, test_func_type=test_func_type, reconstruction_order=reconstruction_order)
        fx_max_err_arr[n, 0] = t[0]
        jump_loc_err_arr[n, 0] = t[1]
        jump_mag_err_arr[n, :] = t[2].T[0, :]
        trunc_fourier_fx_max_err[n, 0] = t[3]
    return oy_vals_arr, decay_rate, fx_max_err_arr, jump_loc_err_arr, jump_mag_err_arr, trunc_fourier_fx_max_err


def get_approxF_approxJumpCurve_approxJumpMags(X, Y, n_oy, test_func_type, reconstruction_order):
    nx = len(X)
    ny = len(Y)
    approx_f_vals = mpm.matrix(ny, nx)
    approx_jump_curve = mpm.matrix(nx, 1)
    for ix in tqdm(range(len(X))):
        t = get_approxFx_approxJumpLoc_approxJumpMag(x=X[ix], Y=Y, n_oy=n_oy, test_func_type=test_func_type,
                                                     reconstruction_order=reconstruction_order)
        approx_f_vals[:, ix] = t[0][:, 0]
        approx_jump_curve[ix] = t[1]
    return approx_f_vals, approx_jump_curve