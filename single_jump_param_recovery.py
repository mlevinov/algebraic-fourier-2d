import sys
import mpmath as mpm
import numpy as np
import constants as const
from mpmath import libmp

def set_mpmath_precision(dps=20):
    mpm.mp.dps=dps
def phi_func_val_at_x(x, reconstruction_order, jump_loc, jump_mag_array):
    jump_mag_col_vec = __to_column_vec(jump_mag_array)
    d = reconstruction_order
    s = 0
    for l in range(d + 1):
        a_l = jump_mag_col_vec[l, 0]
        vl = __vn_func_val_at_x(x, l, jump_loc)
        s1 = mpm.fmul(a_l, vl)
        s = mpm.fadd(s, s1)
    return s
def psi_func_val_at_x(x, reconstruction_order, func_coeff_array, jump_loc, jump_mag_array):
    func_coeff_col_vec = __to_column_vec(func_coeff_array)
    jump_mag_col_vec = __to_column_vec(jump_mag_array)

    m = int(np.floor(func_coeff_col_vec.cols / 2))
    s = 0
    for k in range(-m, m + 1):
        phi_k_coeff = __calc_coeff_phi(k, reconstruction_order=reconstruction_order, jump_loc=jump_loc,
                                       jump_mag_array=jump_mag_col_vec)
        psi_k_coeff = __calc_coeff_psi(func_coeff_at_k=func_coeff_col_vec[m+k, 0], phi_coeff_at_k=phi_k_coeff)
        s = mpm.fadd(s, mpm.fmul(psi_k_coeff, mpm.expj(mpm.fmul(k, x))))
    return s
def func_val_at_x(x, reconstruction_order, func_coeff_array, jump_loc, jump_mag_array):
    func_coeff_col_vec = __to_column_vec(func_coeff_array)
    jump_mag_col_vec = __to_column_vec(jump_mag_array)

    phi_val = phi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, jump_loc=jump_loc,
                                jump_mag_array=jump_mag_col_vec)
    psi_val = psi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, func_coeff_array=func_coeff_col_vec,
                                jump_loc=jump_loc, jump_mag_array=jump_mag_col_vec)
    return mpm.fadd(psi_val, phi_val)
def poly_roots(reconstruction_order, func_coeff_array, half_order_flag=True):
    coefficients = __create_polynomial_coefficients(reconstruction_order, func_coeff_array, half_order_flag)
    tries = 10
    max_steps = 50
    extra_prec = 10
    polynomial_roots = []
    convergenceFlag = True
    while tries > 0:
        try:
            polynomial_roots = mpm.polyroots(coefficients, maxsteps=max_steps, extraprec=extra_prec)
            convergenceFlag = True
            break
        except mpm.libmp.libhyper.NoConvergence:
            convergenceFlag = False
            max_steps += 25
            extra_prec += 10
            tries -= 1
        except ZeroDivisionError:
            print('\nZeroDivisionError:')
            print('\ncoefficients for polynomial are:\n{}'.format(coefficients))
            sys.exit('\nfirst coefficient must be NOT ZERO\n')
    if not convergenceFlag:
        # raise NoRootConvergenceError(tries, max_steps, extra_prec)
        print('')
    return polynomial_roots
def approximate_jump_location(reconstruction_order, func_coeff_array,
                              known_jump_location_flag_on=False, half_order_flag=False):
    if known_jump_location_flag_on:
        return -mpm.pi
    else:
        func_coeff_col_vec = __to_column_vec(func_coeff_array)
        if half_order_flag:
            closest_root = __closest_root_to_unit_disk(poly_roots(reconstruction_order, func_coeff_col_vec, half_order_flag=True))
            approximated_jump_location = -mpm.arg(closest_root)
            return approximated_jump_location
        else:
            m = func_coeff_col_vec.cols // 2
            d = reconstruction_order // 2
            half_order_root = approximate_jump_location(d, func_coeff_array,known_jump_location_flag_on=False,
                                                        half_order_flag=True)
            half_order_omega = mpm.exp(mpm.fmul(-1j,half_order_root))
            n = m // (reconstruction_order + 2)
            if n == 0:
                print('M = {} -> floor(M/(d+2)) = 0')
                return 1
            z_n = __closest_root_to_unit_disk(poly_roots(reconstruction_order, func_coeff_col_vec, half_order_flag=False))
            closest_root_to_half_order_root = z_n
            min_dist_z_N_and_half_order_root = mpm.inf
            for k in range(n):
                z_k = mpm.root(z_n, n, k=k)
                current_distance_from_half_order_root_and_z_k = mpm.norm(mpm.fsub(half_order_omega, z_k))
                if current_distance_from_half_order_root_and_z_k < min_dist_z_N_and_half_order_root:
                    min_dist_z_N_and_half_order_root = current_distance_from_half_order_root_and_z_k
                    closest_root_to_half_order_root = z_k
            full_order_approximated_jump_location = -mpm.arg(closest_root_to_half_order_root)
            return full_order_approximated_jump_location
def approximate_jump_magnitudes(reconstruction_order, func_coeff_array, approximated_jump_location):
    func_coeff_col_vec = __to_column_vec(func_coeff_array)
    omega = mpm.expj(-approximated_jump_location)
    m = func_coeff_col_vec.rows // 2
    n = m // (reconstruction_order + 2)
    b = mpm.matrix(reconstruction_order + 1, 1)
    VNd = mpm.matrix(reconstruction_order + 1, reconstruction_order + 1)
    approximated_jump_magnitudes = mpm.matrix(reconstruction_order + 1, 1)

    for i in range( reconstruction_order + 1):
        index = (i + 1) * n
        mk_tilde_val = mpm.fmul(mpm.fmul(const.TWO_PI, mpm.power(mpm.fmul(1j, index), reconstruction_order + 1)), func_coeff_col_vec[m + index, 0])
        b[i, 0] = mpm.fmul(mk_tilde_val, mpm.power(omega, -index))
        for j in range(reconstruction_order + 1):
            VNd[i, j] = mpm.power(index, j)
    sol = mpm.lu_solve(VNd, b)
    # extracting the actual jump magnitudes:
    # denote A_l as the jump magnitude of d^l/dx^l(Fx) at its singularity then
    # denote a_l = sol[l]-> sol = [a_0, a_1, ..., a_l]^T, then
    # A_l = (-i)^(r0-l) * a_(ro-l)
    for l in range(reconstruction_order + 1):
        # sol = [alpha_0,..., alpha_d]
        # s = alpha_d -> alpha_0
        s = sol[reconstruction_order - l]
        c = mpm.power(-1j, reconstruction_order - l)
        approximated_jump_magnitudes[l, 0] = mpm.fmul(c, s)
    return approximated_jump_magnitudes
def __closest_root_to_unit_disk(roots):
    if not roots:
        print('root list is empty')
        return 1
    roots_col_vec = __to_column_vec(roots)
    closest_root_to_unit_disk = roots_col_vec[0,0]
    min_distance_of_root_to_unit_disk = mpm.inf
    for i in range(len(roots)):
        root = roots_col_vec[i, 0]
        if root != 0:
            current_distance_of_root_to_unit_disk = mpm.fabs(mpm.fsub(root, mpm.fdiv(root,mpm.norm(root))))
            if current_distance_of_root_to_unit_disk < min_distance_of_root_to_unit_disk:
                min_distance_of_root_to_unit_disk = current_distance_of_root_to_unit_disk
                closest_root_to_unit_disk = root

    normalized_closest_root_to_unit_disk = mpm.fdiv(closest_root_to_unit_disk, mpm.fabs(closest_root_to_unit_disk))
    # return closest_root_to_unit_disk
    return normalized_closest_root_to_unit_disk
def __to_column_vec(vec):
    col_vec = vec
    if vec.cols != 1:
        return col_vec.T
    else:
        return col_vec
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
    jump_mag_col_vec = __to_column_vec(jump_mag_array)
    if k==0:
        return 0
    else:
        r1 = mpm.fdiv(mpm.expj(-mpm.fmul(jump_loc, k)), const.TWO_PI)
        r2 = 0
        for l in range(reconstruction_order + 1):
            r3 = mpm.fdiv(jump_mag_col_vec[l, 0], mpm.power(mpm.fmul(1j, k), l + 1))
            r2 = mpm.fadd(r2, r3)
        return mpm.fmul(r1, 2)
def __calc_coeff_psi(func_coeff_at_k, phi_coeff_at_k):
    return mpm.fsub(func_coeff_at_k, phi_coeff_at_k)
def __create_polynomial_coefficients(reconstruction_order, func_coeff_array, half_order_flag=True):
    func_coeff_col_vec = __to_column_vec(func_coeff_array)
    m = func_coeff_col_vec.rows // 2
    n = m // (reconstruction_order + 2)
    d = reconstruction_order + 1

    polynom_coefficients = mpm.matrix(d, 1)
    if half_order_flag:
        for j in range(d + 1):
            # here maybe an issue with array out of bounds
            index = m - reconstruction_order - 1 + j
            ck = func_coeff_col_vec[m + index, 0]
            mk_tilde_val = mpm.fmul(mpm.fmul(const.TWO_PI, mpm.power(mpm.fmul(1j,index),reconstruction_order + 1)), ck)
            polynom_coefficients[j, 0] = mpm.fmul(mpm.fmul(mpm.power(-1, j), mpm.binomial(reconstruction_order + 1, j)), mk_tilde_val)
    else:
        for j in range(d + 1):
            index = (j+1) * n
            ck = func_coeff_col_vec[m + index, 0]
            mk_tilde_val = mpm.fmul(mpm.fmul(const.TWO_PI, mpm.power(mpm.fmul(1j,index),reconstruction_order + 1)), ck)
            polynom_coefficients[j, 0] = mpm.fmul(mpm.fmul(mpm.power(-1, j), mpm.binomial(reconstruction_order + 1, j)), mk_tilde_val)
    return  polynom_coefficients
