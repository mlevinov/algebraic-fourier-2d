import sys
import mpmath as mpm
import constants as const
from mpmath import libmp


def set_mpmath_precision(dps=20):
    """
    setting the precision for the mpmath library
    args:
        dps: (int) - number of significant numbers
    returns: no return
    """
    mpm.mp.dps = dps


def phi_func_val_at_x(x, reconstruction_order, jump_loc, jump_mag_array):
    """
    calculate :math:`\phi(x)`
    Args:
        x: (mpmath.mpf or float) :math:`x \in [-\pi,\pi)`
        reconstruction_order: (int) - chosen reconstruction order for :math:`F_{x}` or :math:`\psi_{\omega_y}`
        jump_loc: (mpmath.mpc or float) - approximated or exact point of the jump location for :math:`F_{x}` or :math:`\psi_{\omega_y}`
        jump_mag_array: (mpmath.matrix) - approximated or exact magnitudes of the jumps at jump_loc for :math:`F_{x}\text{ or }\psi_{\omega_y`

    Returns:
        returns a mpmath.mpc value representing :math:`\phi(x)`

    """
    d = reconstruction_order
    s = 0
    for l in range(d + 1):
        al = jump_mag_array[l, 0]
        vl = __vn_func_val_at_x(x, l, jump_loc)
        s1 = mpm.fmul(al, vl)
        s = mpm.fadd(s, s1)
    return s


def psi_func_val_at_x(x, reconstruction_order, func_coeff_array, jump_loc, jump_mag_array):
    """
    calculate the value of :math:`\psi_{\omega_y}(x)`
    Args:
        x: :math:`x\in [-\pi, \pi)`
        reconstruction_order: (int) - chosen reconstruction order for :math:`\psi_{\omega_y}`
        func_coeff_array: (mpmath.matrix) - array of exact Fourier coefficients for reconstructing :math:`\psi_{\omega_y}`
        jump_loc: (mpmath.mpc) - value representing the jump location of :math:`\psi_{\omega_y}`
        jump_mag_array: (mpmath.matrix) - an array containing the jump magnitudes of :math:`\psi_{\omega_y}`'s derivatives

    Returns:
        returns the approximated value of :math:`\psi_{\omega_y}(x)`

    """
    m = func_coeff_array.rows // 2
    s = 0
    for k in range(-m, m + 1):
        phi_k_coeff = __calc_coeff_phi(k, reconstruction_order=reconstruction_order, jump_loc=jump_loc,
                                       jump_mag_array=jump_mag_array)
        psi_k_coeff = mpm.fsub(func_coeff_array[m + k, 0], phi_k_coeff)
        s1 = mpm.expj(mpm.fmul(k, x))
        s2 = mpm.fmul(psi_k_coeff, s1)
        s = mpm.fadd(s, s2)
    return s


def func_val_at_x(x, reconstruction_order, func_coeff_array, jump_loc, jump_mag_array):
    """
    approximated value of :math:`F_x(y)` (x is just a placeholder for a value in :math:`[-\pi, \pi)`
    Args:
        x: :math:`x \in [-\pi, \pi)`
        reconstruction_order: (int) - reconstruction order for :math:`\psi_{\omega_y}\text{ and } F_x`
        func_coeff_array: (mpmath.matrix) - exact Fourier coefficients for reconstructing :math:`\psi_{\omega_y}`
        jump_loc: (mpmath.mpc) - approximated location of :math:`F_x` jump location
        jump_mag_array: (mpmath.matrix) - an array of jump magnitudes of :math:`F_x` at jump_loc

    Returns:
        returns the approximated value of :math:`F_x` at the given point

    """
    phi_val = phi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, jump_loc=jump_loc,
                                jump_mag_array=jump_mag_array)
    psi_val = psi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, func_coeff_array=func_coeff_array,
                                jump_loc=jump_loc, jump_mag_array=jump_mag_array)
    return mpm.fadd(psi_val, phi_val)


def poly_roots(reconstruction_order, func_coeff_array, half_order_flag=False):
    """
    calculates the roots of :math:`q_N^d` where d is the reconstruction order
    Args:
        reconstruction_order: (int) - reconstruction order for :math:`\psi_{\omega_y}\text{ and } F_x`
        func_coeff_array: (mpmath.matrix) - exact Fourier coefficients for :math:`\psi_{\omega_y}` or approximated Fourier coefficients for :math:`Fx`
        half_order_flag: (boolean) - a flag to indicate the method for calculating the roots

    Returns:
        returns a list of :math:`q_N^d`'s roots

    """
    coefficients = __create_polynomial_coefficients(reconstruction_order, func_coeff_array, half_order_flag)
    tries = 10
    max_steps = 50
    extra_prec = 10
    polynomial_roots = []
    convergenceflag = True
    while tries > 0:
        try:
            polynomial_roots = mpm.polyroots(coefficients, maxsteps=max_steps, extraprec=extra_prec)
            convergenceflag = False
            break
        except mpm.libmp.libhyper.noconvergence:
            convergenceflag = False
            max_steps += 25
            extra_prec += 10
            tries -= 1
        except ZeroDivisionError:
            print('\nZeroDivisionError:')
            print('\ncoefficients for polynomial are:\n{}'.format(coefficients))
            sys.exit('\nfirst coefficient must be not zero\n')
    if not convergenceflag:
        # raise norootconvergenceerror(tries, max_steps, extra_prec)
        print('')
    return polynomial_roots


def approximate_jump_location(reconstruction_order, func_coeff_array, half_order_flag=False, get_omega_flag=False):
    """

    Args:
        reconstruction_order:
        func_coeff_array:
        half_order_flag:
        get_omega_flag:

    Returns:

    """
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
        n = m // (reconstruction_order + 2)
        if n == 0:
            print('m = {} -> floor(m/(d+2)) = 0')
            return 1
        z_n = __closest_root_to_unit_disk(poly_roots(reconstruction_order, func_coeff_array, half_order_flag=False))
        closest_root_to_half_order_root = z_n
        min_dist_z_n_and_half_order_root = mpm.inf
        for k in range(n):
            z_k = mpm.root(z_n, n, k=k)
            current_distance_from_half_order_root_and_z_k = mpm.norm(mpm.fsub(half_order_root, z_k))
            if current_distance_from_half_order_root_and_z_k < min_dist_z_n_and_half_order_root:
                min_dist_z_n_and_half_order_root = current_distance_from_half_order_root_and_z_k
                closest_root_to_half_order_root = z_k
        full_order_approximated_jump_location = -mpm.arg(closest_root_to_half_order_root)
        return full_order_approximated_jump_location


def approximate_jump_magnitudes(reconstruction_order, func_coeff_array, approximated_jump_location, known_jump_loc=False):
    if known_jump_loc:
        omega = -1
    else:
        omega = mpm.expj(-approximated_jump_location)
    m = func_coeff_array.rows // 2
    n = m // (reconstruction_order + 2)

    vnd = mpm.matrix(reconstruction_order + 1, reconstruction_order + 1)
    approximated_jump_magnitudes = mpm.matrix(reconstruction_order + 1, 1)
    # creating b
    b = mpm.matrix(reconstruction_order + 1, 1)
    for i in range(reconstruction_order + 1):
        index = (i + 1) * n
        mk_val = __mk(ro=reconstruction_order, k=index, ck=func_coeff_array[m + index, 0])
        try:
            b[i, 0] = mpm.fmul(mk_val, mpm.power(omega, -index))
        except ZeroDivisionError:
            print()
            print('devision by zero error')
            print('applying the next fix to continue b[i,0] = b[i-1, 0] if i>=0, else b[0,0] = 0')
            print('notice that the results would lose accuracy')
            print()
            if i > 0:
                b[i, 0] = b[i - 1, 0]
            else:
                b[i, 0] = 0
        # continue to create row i in vnd
        for j in range(reconstruction_order + 1):
            vnd[i, j] = mpm.power(index, j)

    # solving vnd * x = b
    sol = mpm.lu_solve(vnd, b)
    # extracting the actual jump magnitudes:
    # denote a_l as the jump magnitude of d^l/dx^l(fx) at its singularity then
    # denote a_l = sol[l]-> sol = [a_0, a_1, ..., a_l]^t, then
    # a_l = (-i)^(r0-l) * a_(ro-l)
    for l in range(reconstruction_order + 1):
        # sol = [alpha_0,..., alpha_d]
        # s = alpha_d -> alpha_0
        s = sol[reconstruction_order - l]
        c = mpm.power(1j, reconstruction_order - l)
        approximated_jump_magnitudes[l, 0] = mpm.fdiv(s, c)
    return approximated_jump_magnitudes


def __closest_root_to_unit_disk(roots):
    m = len(roots)
    if not roots:
        print('root list is empty')
        return 1
    closest_root_to_unit_disk = roots[0]
    min_distance_of_root_to_unit_disk = mpm.inf
    for i in range(m):
        root = roots[i]
        if root != 0:
            normalized_root = mpm.fdiv(root, mpm.norm(root))
            current_distance_of_root_to_unit_disk = mpm.fabs(mpm.fsub(root, normalized_root))
            if current_distance_of_root_to_unit_disk < min_distance_of_root_to_unit_disk:
                min_distance_of_root_to_unit_disk = current_distance_of_root_to_unit_disk
                closest_root_to_unit_disk = root
    return closest_root_to_unit_disk


def __vn_func_val_at_x(x, n, jump_loc):
    a1 = mpm.power(const.two_pi, n)
    a2 = mpm.factorial(n + 1)
    a3 = -mpm.fdiv(a1, a2)
    z = mpm.fsub(x, jump_loc)
    if 0 <= z < const.two_pi:
        zz = mpm.fdiv(z, const.two_pi)
    elif -const.two_pi <= z < 0:
        zz = mpm.fdiv(mpm.fadd(z, const.two_pi), const.two_pi)
    else:
        return -mpm.inf
    a4 = mpm.bernpoly(n + 1, zz)
    s = mpm.fmul(a3, a4)
    return s


def __calc_coeff_phi(k, reconstruction_order, jump_loc, jump_mag_array):
    if k != 0:
        omega = mpm.expj(-mpm.fmul(jump_loc, k))
        c = mpm.fdiv(omega, const.two_pi)
        r = 0
        for l in range(reconstruction_order + 1):
            al = jump_mag_array[l, 0]
            r1 = mpm.fmul(1j, k)
            r2 = mpm.power(r1, l + 1)
            r3 = mpm.fdiv(al, r2)
            r = mpm.fadd(r, r3)
        return mpm.fmul(c, r)
    else:
        return 0


def __mk(ro, k, ck):
    a1 = mpm.power(mpm.fmul(1j, k), ro + 1)
    a2 = mpm.fmul(a1, ck)
    return mpm.fmul(const.two_pi, a2)


def __create_polynomial_coefficients(reconstruction_order, func_coeff_col_vec, half_order_flag=True):
    m = func_coeff_col_vec.rows // 2
    if half_order_flag:
        half_order = reconstruction_order // 2
        polynom_coefficients = mpm.matrix(half_order + 2, 1)
        for j in range(half_order + 2):
            k = m - half_order - 1
            mk_val = __mk(ro=half_order, k=k + j, ck=func_coeff_col_vec[m + k + j, 0])
            b1 = mpm.power(-1, j)
            b2 = mpm.binomial(half_order + 1, j)
            b3 = mpm.fmul(b1, b2)
            polynom_coefficients[j, 0] = mpm.fmul(b3, mk_val)
        return polynom_coefficients
    else:
        n = m // (reconstruction_order + 2)
        polynom_coefficients = mpm.matrix(reconstruction_order + 2, 1)
        for j in range(reconstruction_order + 2):
            k = (j + 1) * n
            mk_val = __mk(ro=reconstruction_order, k=k, ck=func_coeff_col_vec[m + k, 0])
            b1 = mpm.power(-1, j)
            b2 = mpm.binomial(reconstruction_order + 1, j)
            b3 = mpm.fmul(b1, b2)
            polynom_coefficients[j, 0] = mpm.fmul(b3, mk_val)
        return polynom_coefficients
