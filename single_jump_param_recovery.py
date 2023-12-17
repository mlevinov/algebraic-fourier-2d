import sys
import mpmath as mpm
import numpy as np
import constants as const
from mpmath import libmp

def set_mpmath_precision(dps=20):
    mpm.mp.dps=dps
def phi_func_val_at_x(x, reconstruction_order, jump_loc, jump_mag_array):
    d = reconstruction_order
    s = 0
    for l in range(d + 1):
        al = jump_mag_array[l, 0]
        vl = __vn_func_val_at_x(x, l, jump_loc)
        s1 = mpm.fmul(al, vl)
        s = mpm.fadd(s, s1)
    return s
def psi_func_val_at_x(x, reconstruction_order, func_coeff_array, jump_loc, jump_mag_array):
    m = func_coeff_array.rows // 2
    s = 0
    for k in range(-m, m + 1):
        phi_k_coeff = __calc_coeff_phi(k, reconstruction_order=reconstruction_order, jump_loc=jump_loc,
                                       jump_mag_array=jump_mag_array)
        psi_k_coeff = mpm.fsub(func_coeff_array[m + k, 0], phi_k_coeff)
        s1 = mpm.expj(mpm.fmul(k, x))
        s2 = mpm.fmul(psi_k_coeff, s1)
        s3 = mpm.fmul(s1, s2)
        s = mpm.fadd(s, s3)
    return s
def func_val_at_x(x, reconstruction_order, func_coeff_array, jump_loc, jump_mag_array):
    phi_val = phi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, jump_loc=jump_loc,
                                jump_mag_array=jump_mag_array)
    psi_val = psi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, func_coeff_array=func_coeff_array,
                                jump_loc=jump_loc, jump_mag_array=jump_mag_array)
    return mpm.fadd(psi_val, phi_val)
def poly_roots(reconstruction_order, func_coeff_array, half_order_flag=False):
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
def approximate_jump_location(reconstruction_order, func_coeff_array, half_order_flag=False, get_omega_flag=False):
    m = func_coeff_array.rows // 2
    if half_order_flag:
        roots = poly_roots(reconstruction_order, func_coeff_array, half_order_flag=True)
        closest_root = __closest_root_to_unit_disk(roots)
        if get_omega_flag:
            return closest_root
        approximated_jump_location = -mpm.arg(closest_root)
        return approximated_jump_location
    else:
        half_order_root = approximate_jump_location(reconstruction_order, func_coeff_array, half_order_flag=True, get_omega_flag=True)
        # half_order_omega = mpm.exp(mpm.fmul(-1j, half_order_root))
        n = m // (reconstruction_order + 2)
        if n == 0:
            print('M = {} -> floor(M/(d+2)) = 0')
            return 1
        z_n = __closest_root_to_unit_disk(poly_roots(reconstruction_order, func_coeff_array, half_order_flag=False))
        closest_root_to_half_order_root = z_n
        min_dist_z_N_and_half_order_root = mpm.inf
        for k in range(n):
            z_k = mpm.root(z_n, n, k=k)
            current_distance_from_half_order_root_and_z_k = mpm.norm(mpm.fsub(half_order_root, z_k))
            if current_distance_from_half_order_root_and_z_k < min_dist_z_N_and_half_order_root:
                min_dist_z_N_and_half_order_root = current_distance_from_half_order_root_and_z_k
                closest_root_to_half_order_root = z_k
        if get_omega_flag:
            return closest_root_to_half_order_root
        full_order_approximated_jump_location = -mpm.arg(closest_root_to_half_order_root)
        return full_order_approximated_jump_location
def approximate_jump_magnitudes(reconstruction_order, func_coeff_array, approximated_jump_location):
    omega = mpm.expj(-approximated_jump_location)
    m = func_coeff_array.rows // 2
    n = m // (reconstruction_order + 2)
    b = mpm.matrix(reconstruction_order + 1, 1)
    VNd = mpm.matrix(reconstruction_order + 1, reconstruction_order + 1)
    approximated_jump_magnitudes = mpm.matrix(reconstruction_order + 1, 1)

    for i in range( reconstruction_order + 1):
        index = (i + 1) * n
        mk_tilde_val = mpm.fmul(mpm.fmul(const.TWO_PI, mpm.power(mpm.fmul(1j, index), reconstruction_order + 1)),
                                func_coeff_array[m + index, 0])
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
    if len(roots) == 1:
        closest_root_to_unit_disk = roots[0]
        normalized_closest_root_to_unit_disk = mpm.fdiv(closest_root_to_unit_disk, mpm.fabs(closest_root_to_unit_disk))
        return normalized_closest_root_to_unit_disk
    closest_root_to_unit_disk = roots[0]
    min_distance_of_root_to_unit_disk = mpm.inf
    for i in range(len(roots)):
        root = roots[i]
        if root != 0:
            current_distance_of_root_to_unit_disk = mpm.fabs(mpm.fsub(root, mpm.fdiv(root,mpm.norm(root))))
            if current_distance_of_root_to_unit_disk < min_distance_of_root_to_unit_disk:
                min_distance_of_root_to_unit_disk = current_distance_of_root_to_unit_disk
                closest_root_to_unit_disk = root

    normalized_closest_root_to_unit_disk = mpm.fdiv(closest_root_to_unit_disk, mpm.fabs(closest_root_to_unit_disk))
    return normalized_closest_root_to_unit_disk
def __vn_func_val_at_x(x, n, jump_loc):
    s = -mpm.fdiv(mpm.power(const.TWO_PI, n), mpm.factorial(n + 1))
    z = mpm.fsub(x, jump_loc)
    if 0 <= z < const.TWO_PI:
        s1 = mpm.fdiv(z, const.TWO_PI)
        s2 = mpm.bernpoly(n + 1, s1)
        return mpm.fmul(s, s2)
    elif -const.TWO_PI <= z < 0:
        s1 = mpm.fadd(z, const.TWO_PI)
        s2 = mpm.fdiv(s1, const.TWO_PI)
        s3 = mpm.bernpoly(n + 1, s2)
        return mpm.fmul(s, s3)
    else:
        return -mpm.inf
def __calc_coeff_phi(k, reconstruction_order, jump_loc, jump_mag_array):
    if k==0:
        return 0
    else:
        r1 = mpm.fdiv(mpm.expj(-mpm.fmul(jump_loc, k)), const.TWO_PI)
        r2 = 0
        for l in range(reconstruction_order + 1):
            r3 = mpm.fdiv(jump_mag_array[l, 0], mpm.power(mpm.fmul(1j, k), l + 1))
            r2 = mpm.fadd(r2, r3)
        return mpm.fmul(r1, r2)
def __calc_coeff_psi(func_coeff_at_k, phi_coeff_at_k):
    return mpm.fsub(func_coeff_at_k, phi_coeff_at_k)
def __create_polynomial_coefficients(reconstruction_order, func_coeff_col_vec, half_order_flag=True):
    m = func_coeff_col_vec.rows // 2
    def __mk__(ro, k, ck):
        a1 = mpm.power(mpm.fmul(1j, k), ro + 1)
        a2 = mpm.fmul(a1, ck)
        val = mpm.fmul(const.TWO_PI, a2)
        b1 = mpm.power(-1, j)
        b2 = mpm.binomial(ro + 1, j)
        b3 = mpm.fmul(b1, b2)
        return mpm.fmul(b3, val)

    if half_order_flag:
        half_order = reconstruction_order // 2
        polynom_coefficients = mpm.matrix(half_order + 2, 1)
        for j in range(half_order+2):
            k = m - half_order - 1
            polynom_coefficients[j, 0] = __mk__(ro=half_order, k=k+j, ck=func_coeff_col_vec[m + k + j ,0])
        return polynom_coefficients
    else:
        n = m // (reconstruction_order + 2)
        polynom_coefficients = mpm.matrix(reconstruction_order + 2, 1)
        for j in range(reconstruction_order + 2):
            k = (j+1) * n
            polynom_coefficients[j, 0] = __mk__(ro=reconstruction_order, k=k, ck=func_coeff_col_vec[m + k, 0])
        return  polynom_coefficients
