import mpmath as mpm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import single_jump_param_recovery as sj
import constants as const
import mpmath_tools as mpt
from test_functions import TestFunctions
from tqdm import tqdm
import plot_tools as plt_t


def plot_exact_f_and_exact_jump_curve(X, Y, func_vals, test_func_type):
    XV, YV = np.meshgrid(X, Y)
    plt.figure(figsize=[14, 14])
    ax = plt.axes(projection='3d')
    np_func_val = np.real(mpt.mpm_matrix_to_mpmath_numpy_array(func_vals))
    ax.plot_surface(XV, YV, np_func_val, color='blue')
    minF = np.min(np_func_val)
    Zs = np.zeros(shape=(len(X)))
    for k in range(len(X)):
        Zs[k] = minF
    if test_func_type == 1 or test_func_type == 2:
        plt.suptitle('exact F with ' + r'$\xi(x)=$%s' % 'x')
        ax.scatter(X, X, Zs, c='r', marker='^')
    else:  # type = 3
        plt.suptitle('exact F with ' + r'$\xi(x)=$%s' % 'x/2')
        ax.scatter(X, X / 2, Zs, c='r', marker='^')
    # fake lines for adding legend to a surface plot
    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    ax.legend([fake2Dline1, fake2Dline2], [r'$f$', r'$\xi$'], numpoints=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(15, 280)
    plt.show()


def plot_approx_f_and_approx_jump_curve(X, Y, f_tilde_vals, jump_curve_tilde, test_func_type):
    XV, YV = np.meshgrid(X, Y)
    plt.figure(figsize=[14, 14])
    ax = plt.axes(projection='3d')
    np_func_val = np.real(mpt.mpm_matrix_to_mpmath_numpy_array(f_tilde_vals))
    np_jump_curve_val = np.real(mpt.mpm_matrix_to_mpmath_numpy_array(jump_curve_tilde))[:, 0]
    ax.plot_surface(XV, YV, np_func_val, color='blue')
    minF = np.min(np_func_val)
    Zs = np.zeros(shape=(len(X)))
    for k in range(len(X)):
        Zs[k] = minF
    if test_func_type == 1 or test_func_type == 2:
        plt.suptitle(r'$\tilde{f}$' + ' and ' + r'$\tilde{\xi}ֿ\approx$%s' % 'x')
        ax.scatter(X, np_jump_curve_val, Zs, c='r', marker='^')
    else:  # type 3
        plt.suptitle(r'$\tilde{f}$' + ' and ' + r'$\tilde{\xi}ֿ\approx$%s' % 'x/2')
        ax.scatter(X, np_jump_curve_val, Zs, c='r', marker='^')
    # fake lines for adding legend to a surface plot
    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    ax.legend([fake2Dline1, fake2Dline2], [r'$\tilde{f}$', r'$\tilde{\xi}$'], numpoints=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(15, 280)
    plt.show()


def plot_err_in_approx_f_at_x_vs_n(x, Y, oy_strt_val, num_of_oy_vals, inc_oy, reconstruction_order, test_func_type=1):
    oy_vals = plt_t.create_oy_values(oy_strt_val=oy_strt_val, num_of_oy_vals=num_of_oy_vals, inc_oy=inc_oy)
    n_oy = len(oy_vals)
    for n in range(n_oy):
        plt_t.get_approx_of_fx_jump_loc_jump_mag(x=x, Y=Y, n_oy=n, test_func_type=test_func_type,
                                                 reconstruction_order=reconstruction_order)
    np_exact_f_at_x_val = np.real(mpt.mpm_matrix_to_mpmath_numpy_array(exact_f_at_x))
    np_f_tilde_at_x_val = np.real(mpt.mpm_matrix_to_mpmath_numpy_array(f_tilde_at_x))
    for k in range(len(X)):
        Zs[k] = minF
    if test_func_type == 1 or test_func_type == 2:
        plt.suptitle(r'$\tilde{f}$' + ' and ' + r'$\tilde{\xi}ֿ\approx$%s' % 'x')
        ax.scatter(X, np_jump_curve_val, Zs, c='r', marker='^')
    else:  # type 3
        plt.suptitle(r'$\tilde{f}$' + ' and ' + r'$\tilde{\xi}ֿ\approx$%s' % 'x/2')
        ax.scatter(X, np_jump_curve_val, Zs, c='r', marker='^')
    # fake lines for adding legend to a surface plot
    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    ax.legend([fake2Dline1, fake2Dline2], [r'$\tilde{f}$', r'$\tilde{\xi}$'], numpoints=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(15, 280)
    plt.show()
    return 0


def plot_err_in_approx_jump_location_at_x_vs_n():
    return 0


def plot_err_in_approx_jump_magnitudes_vs_n():
    return 0


if __name__ == "__main__":
    mpm.mp = 25

    #################### Error in F_x vs Moy ####################
    func_type = const.FUNC_TYPE_F2
    psi_jump_loc = -const.MP_PI
    x = 1
    ny = 64
    Y = np.linspace(-const.NP_PI, const.NP_PI - const.EPS, ny)
    #################### Error in xi vs Moy ####################

    #################### Error in A_l vs Moy ####################

    # #################### Exact F vs xi ####################
    # print()
    # print('##### creating exact f #####')
    # print()
    # nx = ny
    # X = np.linspace(-const.NP_PI, const.NP_PI - const.EPS, nx)
    # tf = TestFunctions(func_type=func_type)
    # func_val = tf.get_func_val(X, Y)
    # plot_exact_f_and_exact_jump_curve(X, Y, func_val, test_func_type=tf.get_func_type())

    ###################  F_tilde vs xi_tilde ####################
    print()
    print('##### creating psi #####')
    print()
    nx = ny
    X = np.linspace(-const.NP_PI, const.NP_PI - const.EPS, nx)
    tf = TestFunctions(func_type=func_type)
    n = 25
    m = pow(n, 2)
    ro = 0
    psi_vals = mpm.matrix(2 * n + 1, nx)
    for oy in tqdm(range(-n, n + 1)):
        # creating coefficients for psi at oy
        func_coeff = mpm.matrix(2 * m + 1, 1)
        for ox in range(-m, m + 1):
            func_coeff[m + ox, 0] = tf.get_func_fourier_coefficient(ox, oy)
        # approximating psi_oy over X
        for ix in range(nx):
            x = X[ix]
            approx_psi_jump_mag = sj.approximate_jump_magnitudes(reconstruction_order=ro, func_coeff_array=func_coeff,
                                                                 approximated_jump_location=psi_jump_loc, known_jump_loc=True)
            psi_vals[n + oy, ix] = sj.func_val_at_x(x=x, reconstruction_order=ro, func_coeff_array=func_coeff,
                                                    jump_loc=psi_jump_loc, jump_mag_array=approx_psi_jump_mag)
    print()
    print('##### creating f tilde #####')
    print()
    f_tilde = mpm.matrix(ny, nx)
    jump_curve_tilde = mpm.matrix(nx, 1)
    for ix in tqdm(range(nx)):
        psi_oy_at_x = psi_vals[:, ix]
        jump_curve_tilde[ix, 0] = sj.approximate_jump_location(reconstruction_order=ro, func_coeff_array=psi_oy_at_x,
                                                               half_order_flag=False)

        approx_f_jump_mag_at_x = sj.approximate_jump_magnitudes(reconstruction_order=ro, func_coeff_array=psi_oy_at_x,
                                                                approximated_jump_location=jump_curve_tilde[ix, 0],
                                                                known_jump_loc=False)

        for iy in range(ny):
            f_tilde[iy, ix] = sj.func_val_at_x(x=Y[iy], reconstruction_order=ro, func_coeff_array=psi_oy_at_x,
                                               jump_loc=jump_curve_tilde[ix, 0], jump_mag_array=approx_f_jump_mag_at_x)

    plot_approx_f_and_approx_jump_curve(X, Y, f_tilde, jump_curve_tilde, test_func_type=tf.get_func_type())
