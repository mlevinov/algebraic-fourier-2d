import sys
import mpmath as mpm
import numpy as np
import constants as const
from mpmath import libmp

def set_mpmath_precision(dps=20):
    mpm.mp.dps=dps
def phi_func_val_at_x(x, reconstruction_order, jump_loc, jump_mag_array):
    if jump_mag_array.cols != 1:
        jump_mag = jump_mag_array.T
    d = reconstruction_order
    s = 0
    for l in range(d + 1):
        a_l = jump_mag_array[l, 0]
        vl = __vn_func_val_at_x(x, l, jump_loc)
        s1 = mpm.fmul(a_l, vl)
        s = mpm.fadd(s, s1)
    return s
def psi_func_val_at_x(x, reconstruction_order, func_coeff_array, jump_loc, jump_mag_array):
    if func_coeff_array.cols != 1:
        func_coeff_array = func_coeff_array.T
    if jump_mag_array.cols != 1:
        jump_mag_array = jump_mag_array.T

    m = int(np.floor(func_coeff_array.cols / 2))
    s = 0
    for k in range(-m, m + 1):
        phi_k_coeff = __calc_coeff_phi(k, reconstruction_order=reconstruction_order, jump_loc=jump_loc,
                                       jump_mag_array=jump_mag_array)
        psi_k_coeff = __calc_coeff_psi(func_coeff_at_k=func_coeff_array[m+k, 0], phi_coeff_at_k=phi_k_coeff)
        s = mpm.fadd(s, mpm.fmul(psi_k_coeff, mpm.expj(mpm.fmul(k, x))))
    return s
def func_val_at_x(x, reconstruction_order, func_coeff_array, jump_loc, jump_mag_array):
    if func_coeff_array.cols != 1:
        func_coeff_array = func_coeff_array.T
    if jump_mag_array.cols != 1:
        jump_mag_array = jump_mag_array.T

    phi_val = phi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, jump_loc=jump_loc,
                                jump_mag_array=jump_mag_array)
    psi_val = psi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, func_coeff_array=func_coeff_array,
                                jump_loc=jump_loc, jump_mag_array=jump_mag_array)
    return mpm.fadd(psi_val, phi_val)
def __vn_func_val_at_x(x, n, jump_loc):
    c = mpm.fdiv(-mpm.power(const.TWO_PI, n), mpm.factorial(n + 1))
    z = mpm.fsub(x, jump_loc)
    zz = 0
    if 0 <= z < const.TWO_PI:
        s = mpm.fmul(c, mpm.bernpoly(n + 1, mpm.fdiv(z, const.TWO_PI)))
    else: # -const.TWO_PI <= z < 0:
        s = mpm.fmul(c, mpm.bernpoly(n + 1, mpm.fdiv(mpm.fadd(z, const.TWO_PI), const.TWO_PI)))
    return s
def __calc_coeff_phi(k, reconstruction_order, jump_loc, jump_mag_array):
    if jump_mag_array.cols != 1:
        jump_mag_array = jump_mag_array.T
    if k==0:
        return 0
    else:
        r1 = mpm.fdiv(mpm.expj(-mpm.fmul(jump_loc, k)), const.TWO_PI)
        r2 = 0
        for l in range(reconstruction_order + 1):
            r3 = mpm.fdiv(jump_mag_array[l, 0], mpm.power(mpm.fmul(1j, k), l + 1))
            r2 = mpm.fadd(r2, r3)
        return mpm.fmul(r1, 2)
def __calc_coeff_psi(func_coeff_at_k, phi_coeff_at_k):
    return mpm.fsub(func_coeff_at_k, phi_coeff_at_k)
def poly_roots(polynomial_coefficients_array):
    if polynomial_coefficients_array.cols != 1:
        polynomial_coefficients_array = polynomial_coefficients_array.T
    tries = 10
    max_steps = 50
    extra_prec = 10
    polynomial_roots = []
    convergenceFlag = True
    while tries > 0:
        try:
            polynomial_roots = mpm.polyroots(polynomial_coefficients_array, maxsteps=max_steps,
                                             extraprec=extra_prec)
            convergenceFlag = True
            break
        except mpm.libmp.libhyper.NoConvergence:
            convergenceFlag = False
            max_steps += 25
            extra_prec += 10
            tries -= 1
        except ZeroDivisionError:
            print('\nZeroDivisionError:')
            print('\ncoefficients for polynomial are:\n{}'.format(polynomial_coefficients_array))
            sys.exit('\nfirst coefficient must be NOT ZERO\n')
    if not convergenceFlag:
        # raise NoRootConvergenceError(tries, max_steps, extra_prec)
        print('')
    return polynomial_roots
def _polynomial_coefficients(reconstruction_order, func_coeff_array, half_order_flag=False):
    if func_coeff_array.cols != 1:
        func_coeff_array = func_coeff_array.T

    m = func_coeff_array.cols // 2

    coefficients = mpm.matrix(reconstruction_order + 1, 1)
    if half_order_flag:
        for i in range(reconstruction_order + 2):
            mk_tilde_val = const.TWO_PI
            a = mpm.power(-1, i)
            b = mpm.binomial(reconstruction_order + 1, i)
            c = mpm.fmul(mpm.fmul(a, b), mk_tilde_val)
            coefficients[i, 0] = c
