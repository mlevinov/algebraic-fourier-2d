import os
import mpmath as mpm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import constants as const
import mpmath_tools as mpt
import plot_tools as plt_t
from test_functions import TestFunctions


def plot_exact_f_and_exact_jump_curve(X, Y, func_vals, test_func_type, show_plot=True, save_plot=False):
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
    ax.view_init(16,-137,0)

    if save_plot:
        path = "plots/exact f and exact jump curve/"
        name = "exact_f-fCase-{}-Nx_{}-Ny_{}-fRo-{}.pdf".format(test_func_type, len(X), len(Y), reconstruction_order)
        try:
            os.mkdir("./plots")
        except OSError:
            pass
        try:
            os.mkdir("./plots/exact f and exact jump curve")
        except OSError:
            pass
        plt.savefig(path + name, format="pdf")

    if not show_plot:
        plt.close()
    else:
        plt.show()


def plot_approx_f_and_approx_jump_curve(X, Y, f_tilde_vals, jump_curve_tilde, test_func_type, n_oy, show_plot=True, save_plot=False):
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
    ax.view_init(16,-137,0)
    

    if save_plot:
        path = "plots/approx f and approx jump curve/"
        name = "approx_f-fCase-{}-N_{}-fRo-{}.pdf".format(test_func_type, n_oy, reconstruction_order)
        try:
            os.mkdir("./plots")
        except OSError:
            pass
        try:
            os.mkdir("./plots/approx f and approx jump curve")
        except OSError:
            pass
        plt.savefig(path + name, format="pdf")

    if not show_plot:
        plt.close()
    else:
        plt.show()


def plot_err_in_approx_fx_vs_n(x, oy_vals_arr, decay_rate_arr, approx_fx_err, reconstruction_order,
                               test_func_type=const.FUNC_TYPE_F1, show_plot=True, save_plot=False):
    err_fx_np = mpt.mpm_matrix_to_mpmath_numpy_array(approx_fx_err)
    fig, ax = plt.subplots(figsize=[10, 6])
    title01 = r'$\Delta F_x$ vs $N$'
    title02 = r'$\text{reconstruction order} = %d,\; x = %s $' % (reconstruction_order, x)
    ax.set_title(title01 + '\n' + title02, fontsize="20")
    ax.plot(oy_vals_arr, decay_rate_arr, '^-.', label=r'$N^{-%d}$' % (reconstruction_order + 1))
    ax.plot(oy_vals_arr, np.real(err_fx_np), '*--', label=r'$\Delta(F_x)$')
    ax.set_xlabel(r'$N$', fontsize="20")
    ax.set_ylabel(r'$\Delta F_x$', fontsize="20")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize="18", markerscale=1.5)
    if save_plot:
        path = "plots/err in fx vs n/"
        name = "errInFx_x-{}_fCase-{}_N-{}-to-{}_fRo-{}.pdf".format(x, test_func_type, oy_vals_arr[0],
                                                                    oy_vals_arr[len(oy_vals_arr) - 1], reconstruction_order)
        try:
            os.mkdir("./plots")
        except OSError:
            pass
        try:
            os.mkdir("./plots/err in fx vs n")
        except OSError:
            pass
        plt.savefig(path + name, format="pdf")

    if not show_plot:
        plt.close()
    else:
        plt.show()
    return 0


def plot_err_in_approx_jump_location_at_x_vs_n(x, oy_vals_arr, decay_rate_arr, approx_fx_jump_loc_err, reconstruction_order,
                                               test_func_type=const.FUNC_TYPE_F1, show_plot=True, save_plot=False):
    approx_fx_jump_loc_err_np = mpt.mpm_matrix_to_mpmath_numpy_array(approx_fx_jump_loc_err)
    fig, ax = plt.subplots(figsize=[10, 6])
    title01 = r'$\Delta \xi(x)$ vs $N$'
    title02 = r'$\text{reconstruction order} = %d,\; x = %s $' % (reconstruction_order, x)
    ax.set_title(title01 + '\n' + title02, fontsize="20")
    ax.plot(oy_vals_arr, decay_rate_arr, '^-.', label=r'$N^{-%d}$' % (reconstruction_order + 1))
    ax.plot(oy_vals_arr, np.real(approx_fx_jump_loc_err_np), '*--', label=r'$\Delta(F_x)$')
    ax.set_xlabel(r'$N$', fontsize="20")
    ax.set_ylabel(r'$\Delta \xi(x)$', fontsize="20")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize="18", markerscale=1.5)
    if save_plot:
        path = "plots/err in jump_loc vs n/"
        name = "errInJumpLoc-{}_fCase-{}_N-{}-to-{}_fRo-{}.pdf".format(x, test_func_type, oy_vals_arr[0],
                                                                       oy_vals_arr[len(oy_vals_arr) - 1], reconstruction_order)
        try:
            os.mkdir("./plots")
        except OSError:
            pass
        try:
            os.mkdir("./plots/err in jump_loc vs n")
        except OSError:
            pass
        plt.savefig(path + name, format="pdf")

    if not show_plot:
        plt.close()
    else:
        plt.show()
    return 0


def plot_err_in_approx_jump_magnitudes_vs_n(x, oy_vals_arr, decay_rate_arr, approx_fx_jump_mag_err, reconstruction_order,
                                            test_func_type=const.FUNC_TYPE_F1, show_plot=True, save_plot=False):
    fig, ax = plt.subplots(figsize=[10, 6])
    title01 = r'$\Delta A_l(x)$ vs $N$'
    title02 = r'$\text{reconstruction order} = %d,\; x = %s $' % (reconstruction_order, x)
    ax.set_title(title01 + '\n' + title02, fontsize="20")
    ax.plot(oy_vals_arr, decay_rate_arr, '^-.', label=r'$N^{-%d}$' % (reconstruction_order + 1))
    for l in range(reconstruction_order + 1):
        approx_fx_jump_mag_err_np = mpt.mpm_matrix_to_mpmath_numpy_array(approx_fx_jump_mag_err[:, l])
        ax.plot(oy_vals_arr, np.real(approx_fx_jump_mag_err_np), '*--', label=r'$\Delta(A_{%d}(x))$' % l)
    ax.set_xlabel(r'$N$', fontsize="20")
    ax.set_ylabel(r'$\Delta A_l(x)$', fontsize="20")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize="18", markerscale=1.5)
    if save_plot:
        path = "plots/err in jump_mag vs n/"
        name = "errInJumpMag-{}_fCase-{}_N-{}-to-{}_fRo-{}.pdf".format(x, test_func_type, oy_vals_arr[0],
                                                                       oy_vals_arr[len(oy_vals_arr) - 1], reconstruction_order)
        try:
            os.mkdir("./plots")
        except OSError:
            pass
        try:
            os.mkdir("./plots/err in jump_mag vs n")
        except OSError:
            pass
        plt.savefig(path + name, format="pdf")

    if not show_plot:
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    ##################################################################
    #################### General Input Parameters ####################
    ##################################################################
    mpm.mp = 30
    func_type = const.FUNC_TYPE_F2
    psi_jump_loc = -const.MP_PI
    x = 1
    ny = 16
    Y = np.linspace(-const.NP_PI, const.NP_PI - const.EPS, ny)
    oy_str_val = 15
    num_of_oy_vals = 5
    inc_oy = 5
    test_func_type = 2
    reconstruction_order = 0
    show_plot = True
    save_plot = False

    ############################################################################
    ####################### Exact F and Exact Jump Curve #######################
    ############################################################################

    plot_exact_f_and_exact_jump_curve(X=Y, Y=Y, func_vals=TestFunctions(func_type=test_func_type).get_func_val(X=Y, Y=Y),
                                      test_func_type=test_func_type, show_plot=show_plot, save_plot=save_plot)

    ##########################################################################################
    ####################### Approximated F and Approximated Jump Curve #######################
    ##########################################################################################

    tpl = plt_t.get_approxF_approxJumpCurve_approxJumpMags(X=Y, Y=Y, n_oy=oy_str_val, test_func_type=test_func_type,
                                                           reconstruction_order=reconstruction_order)
    plot_approx_f_and_approx_jump_curve(X=Y, Y=Y, f_tilde_vals=tpl[0], jump_curve_tilde=tpl[1], test_func_type=test_func_type,
                                        n_oy=oy_str_val, show_plot=show_plot, save_plot=save_plot)

    #####################################################################################################
    ########## Approximated Fx and Approximated Jump Location and Approximated Jump Magnitudes ##########
    #####################################################################################################

    tpl = plt_t.get_fxErr_jumpLocErr_jumpMagErr_different_oy_vals(x=x, Y=Y, oy_strt_val=oy_str_val, num_of_oy_vals=num_of_oy_vals,
                                                                  inc_oy=inc_oy, test_func_type=test_func_type,
                                                                  reconstruction_order=reconstruction_order)

    plot_err_in_approx_fx_vs_n(x=x, oy_vals_arr=tpl[0], decay_rate_arr=tpl[1], approx_fx_err=tpl[2], reconstruction_order=reconstruction_order,
                               test_func_type=test_func_type, show_plot=show_plot, save_plot=save_plot)

    plot_err_in_approx_jump_location_at_x_vs_n(x=x, oy_vals_arr=tpl[0], decay_rate_arr=tpl[1], approx_fx_jump_loc_err=tpl[3],
                                               reconstruction_order=reconstruction_order, test_func_type=test_func_type,
                                               show_plot=show_plot, save_plot=save_plot)

    plot_err_in_approx_jump_magnitudes_vs_n(x=x, oy_vals_arr=tpl[0], decay_rate_arr=tpl[1], approx_fx_jump_mag_err=tpl[4],
                                            reconstruction_order=reconstruction_order, test_func_type=const.FUNC_TYPE_F1,
                                            show_plot=show_plot, save_plot=save_plot)
