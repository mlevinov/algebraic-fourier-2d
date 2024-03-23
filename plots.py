import sys
import os
import mpmath as mpm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import single_jump_param_recovery as sj
import constants as const
import mpmath_tools as mpt
from test_functions import TestFunctions
from tqdm import tqdm


def plot_f_and_jump_curve(X, Y, func_vals, test_func_type, show_plot=False, save_plot=True):
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
        plt.title('exact F with ' + r'$\xi(x)=$%s' % 'x')
        ax.scatter(X, X, Zs, c='r', marker='^')
    else:  # type = 3
        plt.title('exact F with ' + r'$\xi(x)=$%s' % 'x/2')
        ax.scatter(X, X / 2, Zs, c='r', marker='^')
    # fake lines for adding legend to a surface plot
    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    ax.legend([fake2Dline1, fake2Dline2], [r'$f$', r'$\xi$'], numpoints=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(16, 245)
    if save_plot:
        path = "plots/exact f and xi/"
        name = "f_and_xi-num_xs-{}_num_ys-{}_func_type-{}.pdf".format(len(X), len(Y), test_func_type)
        try:
            os.mkdir("./plots")
        except OSError:
            pass
        try:
            os.mkdir("./plots/exact f and xi")
        except OSError:
            pass
        plt.savefig(path + name, format="pdf")

    if not show_plot:
        plt.close()
    else:
        plt.show()


def plot_approximation_of_f_and_jump_curve(X, Y, reconstruction_order, moy_val, f_tilde_vals, jump_curve_tilde, test_func_type, show_plot=False, save_plot=True):
    XV, YV = np.meshgrid(X, Y)
    plt.figure(figsize=[14, 14])
    ax = plt.axes(projection='3d')
    np_func_val = np.real(mpt.mpm_matrix_to_mpmath_numpy_array(f_tilde_vals))
    np_jump_curve_val = np.real(mpt.mpm_matrix_to_mpmath_numpy_array(jump_curve_tilde))
    ax.plot_surface(XV, YV, np_func_val, color='blue')
    minF = np.min(np_func_val)
    Zs = np.zeros(shape=(len(X)))
    for k in range(len(X)):
        Zs[k] = minF
    s2 = 'reconstruction order is {}'.format(reconstruction_order)
    if test_func_type == 1 or test_func_type == 2:
        s1 = r'$\tilde{f}$' + ' & ' + r'$\tilde{\xi}(x)\approx$%s' % 'x'
        # plt.title(r'$\tilde{f}$' + ' and ' + r'$\tilde{\xi}\approx$%s' % 'x')
        plt.title(s1 + '\n' + s2)
        ax.scatter(X, np_jump_curve_val, Zs, c='r', marker='^')
    else:  # type 3
        s1 = r'$\tilde{f}$' + ' & ' + r'$\tilde{\xi}(x)\approx$%s' % 'x/2'
        # plt.title(r'$\tilde{f}$' + ' and ' + r'$\tilde{\xi}\approx$%s' % 'x/2')
        plt.title(s1 + '\n' + s2)
        ax.scatter(X, np_jump_curve_val, Zs, c='r', marker='^')
    # fake lines for adding legend to a surface plot
    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    ax.legend([fake2Dline1, fake2Dline2], [r'$\tilde{f}$', r'$\tilde{\xi}$'], numpoints=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(16, 245)
    if save_plot:
        path = "plots/f_tilde and xi_tilde/"
        name = "f_and_xi_tilde-num_xs-{}_num_ys-{}_func_type-{}_ro-{}_moy-{}.pdf".format(len(X), len(Y), test_func_type,
                                                                                                reconstruction_order, moy_val)
        try:
            os.mkdir("./plots")
        except OSError:
            pass
        try:
            os.mkdir("./plots/f_tilde and xi_tilde")
        except OSError:
            pass
        plt.savefig(path + name, format="pdf")

    if not show_plot:
        plt.close()
    else:
        plt.show()


def plot_err_in_f_at_x_vs_moy(x, moy_vals, reconstruction_order, err_f_tilde_at_x, err_fourier_f_at_x, show_plot=False, save_plot=False):
    rate_of_decay = mpm.matrix(moy_vals.rows, 1)
    for r in range(moy_vals.rows):
        rate_of_decay[r, 0] = mpm.power(moy_vals[r, 0], -reconstruction_order - 1)
    np_moy_vals = mpt.mpm_matrix_to_mpmath_numpy_array(moy_vals)
    np_rate_of_decay = mpt.mpm_matrix_to_mpmath_numpy_array(rate_of_decay)
    np_delta_f = mpt.mpm_matrix_to_mpmath_numpy_array(err_f_tilde_at_x)
    np_delta_fourier = mpt.mpm_matrix_to_mpmath_numpy_array(err_fourier_f_at_x)
    # plotting:
    t0 = r'$\Delta f_x$ vs $\Delta\mathcal{T}(f_x)$ vs $M_{\omega_y}$'
    t1 = r'$d=%d,\;x=%g$' % (reconstruction_order, x)

    plt.title(t0 + '\n' + t1)
    plt.plot(np_moy_vals, np_rate_of_decay, '^-.', label=r'$M_{\omega_y}^{-%d}$' % (reconstruction_order + 1))
    plt.plot(np_moy_vals, np_delta_f, '*--', label=r'$\Delta(f_x)$')
    plt.plot(np_moy_vals, np_delta_fourier, 'd:', label=r'$\Delta\mathcal{T}(f_x)$')
    plt.xlabel(r'$M_{\omega_y}$')
    plt.ylabel('error magnitude')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    if save_plot:
        path = "plots/err in fx vs fourier vs oy/"
        name = "x-{}_moy-{}-to-{}_ro-{}.pdf".format(x, int(np_moy_vals[0, 0]), int(np_moy_vals[np_moy_vals.size - 1, 0]), reconstruction_order)
        try:
            os.mkdir("./plots")
        except OSError:
            pass
        try:
            os.mkdir("./plots/err in fx vs fourier vs oy")
        except OSError:
            pass
        plt.savefig(path + name, format="pdf")

    if not show_plot:
        plt.close()
    else:
        plt.show()


def plot_err_in_xi_at_x_vs_moy(x, moy_vals, reconstruction_order, err_xi_tilde_at_x, func_type, show_plot=False, save_plot=True):
    rate_of_decay = mpm.matrix(moy_vals.rows, 1)
    for r in range(moy_vals.rows):
        rate_of_decay[r, 0] = mpm.power(moy_vals[r, 0], -reconstruction_order - 2)
    np_moy_vals = mpt.mpm_matrix_to_mpmath_numpy_array(moy_vals)
    np_rate_of_decay = mpt.mpm_matrix_to_mpmath_numpy_array(rate_of_decay)
    np_delta_xi = mpt.mpm_matrix_to_mpmath_numpy_array(err_xi_tilde_at_x)

    # plotting:
    if func_type != const.FUNC_TYPE_F3:
        xi = 'x'
    else:
        xi = r'$x\over 2$'
    t0 = r'$\Delta \xi(x)$ vs $M_{\omega_y}$'
    t1 = r'$d=%d,\; \xi(x) = %s,\; x=%g$' % (reconstruction_order, xi, x)

    plt.title(t0 + '\n' + t1)
    plt.plot(np_moy_vals, np_rate_of_decay, '^-.', label=r'$M_{\omega_y}^{-%d}$' % (reconstruction_order + 2))
    plt.plot(np_moy_vals, np_delta_xi, '*--', label=r'$\Delta(\xi)$')
    plt.xlabel(r'$M_{\omega_y}$')
    plt.ylabel('error magnitude')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    if save_plot:
        path = "plots/err in xi vs oy/"
        name = "x-{}_ftype-{}_moy-{}-to-{}_ro-{}.pdf".format(x, func_type, int(np_moy_vals[0, 0]),
                                                             int(np_moy_vals[np_moy_vals.size - 1, 0]), reconstruction_order)
        try:
            os.mkdir("./plots")
        except OSError:
            pass
        try:
            os.mkdir("./plots/err in xi vs oy")
        except OSError:
            pass
        plt.savefig(path + name, format="pdf")

    if not show_plot:
        plt.close()
    else:
        plt.show()


def plot_err_in_jump_mag_vs_moy(x, moy_vals, err_jump_mag, func_type, show_plot=False, save_plot=True):
    reconstruction_order = err_jump_mag.rows - 1
    rate_of_decay = mpm.matrix(moy_vals.rows, 1)
    for r in range(moy_vals.rows):
        rate_of_decay[r, 0] = mpm.power(moy_vals[r, 0], -reconstruction_order - 2)
    np_moy_vals = mpt.mpm_matrix_to_mpmath_numpy_array(moy_vals)
    np_rate_of_decay = mpt.mpm_matrix_to_mpmath_numpy_array(rate_of_decay)
    np_err_jump_mag = mpt.mpm_matrix_to_mpmath_numpy_array(err_jump_mag)

    # plotting:
    if func_type != const.FUNC_TYPE_F3:
        xi = 'x'
    else:
        xi = r'$x\over 2$'
    t0 = r'$\Delta A_l(x)$ vs $M_{\omega_y}$'
    t1 = r'$d=%d,\; \xi(x) = %s,\; x=%g$' % (reconstruction_order, xi, x)

    for l in range(reconstruction_order + 1):
        plt.plot(np_moy_vals, np_err_jump_mag[l, :], '*--', label=r'$\Delta A_{%d}(x)$' % l)

    plt.plot(np_moy_vals, np_rate_of_decay, '^-.', label=r'$M_{\omega_y}^{-%d}$' % (reconstruction_order + 1))
    plt.title(t0 + '\n' + t1)
    plt.xlabel(r'$M_{\omega_y}$')
    plt.ylabel('error magnitude')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    if save_plot:
        path = "plots/err in jump magnitudes vs oy/"
        name = "x-{}_ftype-{}_moy-{}-to-{}_ro-{}.pdf".format(x, func_type, int(np_moy_vals[0, 0]), int(np_moy_vals[np_moy_vals.size - 1, 0]),
                                                             reconstruction_order)
        try:
            os.mkdir("./plots")
        except OSError:
            pass
        try:
            os.mkdir("./plots/err in jump magnitudes vs oy")
        except OSError:
            pass
        plt.savefig(path + name, format="pdf")

    if not show_plot:
        plt.close()
    else:
        plt.show()

# |-----------------------------------------------------------------------|
# |-------------------------------- Tests --------------------------------|
# |-----------------------------------------------------------------------|


if __name__ == "__main__":
    def f_tilde_xi_tilde_jump_mag_tilde(show_plot=False, save_plot=True):
        dps = 100
        mpm.mp.dps = dps
        func_type = const.FUNC_TYPE_F2
        tf = TestFunctions(dps=dps, func_type=func_type)
        ny = 8
        Y = np.linspace(-const.NP_PI, const.NP_PI - const.EPS, ny)
        x = 1.1
        reconstruction_order = 7
        strt_moy_val = 22
        num_of_moy_vals = 14
        inc_moy = 10
        end_moy_val = strt_moy_val + (num_of_moy_vals * inc_moy)
        psi_jump_loc = -const.MP_PI
        exact_f_at_x = tf.get_func_slice_at_x(x=x, Y=Y)
        moy_vals = []
        max_err_f_tilde = mpm.matrix(num_of_moy_vals + 1, 1)
        max_err_f_fourier = mpm.matrix(num_of_moy_vals + 1, 1)
        err_xi_tilde = mpm.matrix(num_of_moy_vals + 1, 1)
        err_jump_mag_tilde = mpm.matrix(reconstruction_order + 1, num_of_moy_vals + 1)
        i = 0
        for moy in tqdm(range(strt_moy_val, end_moy_val + 1, inc_moy)):
            moy_vals.append(moy)
            mox = pow(moy, 2)
            fourier_f_at_x = tf.compute_1D_fourier_at_x(x=x, Y=Y, mox=mox, moy=moy, func_type=func_type)
            psi_oy_at_x = mpm.matrix(2 * moy + 1, 1)
            func_coeff = mpm.matrix(2 * mox + 1, 1)
            for oy in tqdm(range(-moy, moy + 1)):
                for ox in range(-mox, mox + 1):
                    func_coeff[mox + ox, 0] = tf.get_func_fourier_coefficient(ox, oy)
                approx_psi_jump_mag = sj.approximate_jump_magnitudes(reconstruction_order=reconstruction_order,
                                                                     func_coeff_array=func_coeff,
                                                                     approximated_jump_location=psi_jump_loc,
                                                                     known_jump_loc=True)
                psi_oy_at_x[moy + oy, 0] = sj.func_val_at_x(x=x, reconstruction_order=reconstruction_order,
                                                            func_coeff_array=func_coeff, jump_loc=psi_jump_loc,
                                                            jump_mag_array=approx_psi_jump_mag)
            approx_f_jump_loc_at_x = sj.approximate_jump_location(reconstruction_order=reconstruction_order,
                                                                  func_coeff_array=psi_oy_at_x, half_order_flag=False)
            err_xi_tilde[i, 0] = mpm.fabs(mpm.fsub(approx_f_jump_loc_at_x, tf.get_slice_at_x_jump_loc(x)))

            approx_f_jump_mag_at_x = sj.approximate_jump_magnitudes(reconstruction_order=reconstruction_order,
                                                                    func_coeff_array=psi_oy_at_x,
                                                                    approximated_jump_location=approx_f_jump_loc_at_x,
                                                                    known_jump_loc=False)
            exact_jump_magnitudes_at_x = tf.get_jump_magnitudes_at_x(x)[:reconstruction_order + 1, 0]
            err_jump_mag_tilde[:, i] = mpt.elementwise_norm_matrix(approx_f_jump_mag_at_x, exact_jump_magnitudes_at_x)[:, 0]

            f_tilde_at_x = mpm.matrix(ny, 1)
            for iy in range(ny):
                f_tilde_at_x[iy, 0] = sj.func_val_at_x(Y[iy], reconstruction_order=reconstruction_order,
                                                       func_coeff_array=psi_oy_at_x, jump_loc=approx_f_jump_loc_at_x,
                                                       jump_mag_array=approx_f_jump_mag_at_x)

            # find max err
            err_f_tilde_at_x = mpt.elementwise_norm_matrix(exact_f_at_x, f_tilde_at_x)
            max_err_f_tilde[i, 0] = err_f_tilde_at_x[mpt.find_max_val_index(err_f_tilde_at_x)[0], 0]
            err_fourier_f_at_x = mpt.elementwise_norm_matrix(exact_f_at_x, fourier_f_at_x)
            max_err_f_fourier[i, 0] = err_fourier_f_at_x[mpt.find_max_val_index(err_fourier_f_at_x)[0], 0]
            i += 1
            # print('for moy={},\nmax_err_f={}'.format(moy, max_err_f_tilde[i, 0]))

        # max_max_err_f_tilde = max_err_f_tilde[mpt.find_max_val_index(max_err_f_tilde)[0], 0]
        # max_max_err_f_fourier = max_err_f_fourier[mpt.find_max_val_index(max_err_f_fourier)[0], 0]
        # print()
        # print('xi_err:\n{}'.format(err_xi_tilde))
        # print()

        # |-------------------- Error in F_x vs Moy --------------------|
        plot_err_in_f_at_x_vs_moy(x=x, moy_vals=mpm.matrix(moy_vals), reconstruction_order=reconstruction_order, err_f_tilde_at_x=max_err_f_tilde,
                                  err_fourier_f_at_x=max_err_f_fourier, show_plot=show_plot, save_plot=save_plot)
        # |-------------------- Error in xi vs Moy ---------------------|
        plot_err_in_xi_at_x_vs_moy(x=x, moy_vals=mpm.matrix(moy_vals), reconstruction_order=reconstruction_order, err_xi_tilde_at_x=err_xi_tilde,
                                   func_type=tf.get_func_type(), show_plot=show_plot, save_plot=save_plot)
        # |------------------- Error in A_l vs Moy ---------------------|
        plot_err_in_jump_mag_vs_moy(x=x, moy_vals=mpm.matrix(moy_vals), err_jump_mag=err_jump_mag_tilde, func_type=tf.get_func_type(),
                                    show_plot=show_plot, save_plot=save_plot)
    def exact_f_and_xi(show_plot=False, save_plot=True):
        print('plotting exact f and xi')
        dps = 100
        mpm.mp.dps = dps
        func_type = const.FUNC_TYPE_F2
        nx = 128
        ny = 128
        X = np.linspace(-const.NP_PI, const.NP_PI - const.EPS, nx)
        Y = np.linspace(-const.NP_PI, const.NP_PI - const.EPS, ny)
        tf = TestFunctions(dps=dps, func_type=func_type)
        exact_func_val = tf.get_func_val(X, Y)
        plot_f_and_jump_curve(X, Y, exact_func_val, test_func_type=tf.get_func_type(), show_plot=show_plot, save_plot=save_plot)
    def f_tilde_and_xi_tilde(show_plot=False, save_plot=True):
        print('plotting f_tilde and xi_tilde')
        dps = 100
        mpm.mp.dps = dps
        func_type = const.FUNC_TYPE_F2
        nx =  32
        ny = 32
        X = np.linspace(-const.NP_PI + const.EPS, const.NP_PI - const.EPS, nx)
        Y = np.linspace(-const.NP_PI + const.EPS, const.NP_PI - const.EPS, ny)
        tf = TestFunctions(dps=dps, func_type=func_type)
        psi_jump_loc = -const.MP_PI
        moy = 18
        mox = pow(moy, 2)
        ro = 1
        psi_vals = mpm.matrix(2 * moy + 1, nx)
        # calculating coefficients
        print('calculating coefficients\n')
        for oy in tqdm(range(-moy, moy + 1)):
            # creating coefficients for psi at oy
            func_coeff = mpm.matrix(2 * mox + 1, 1)
            for ox in range(-mox, mox + 1):
                func_coeff[mox + ox, 0] = tf.get_func_fourier_coefficient(ox, oy)
            # approximating psi_oy over X
            for ix in range(nx):
                x = X[ix]
                approx_psi_jump_mag = sj.approximate_jump_magnitudes(reconstruction_order=ro, func_coeff_array=func_coeff,
                                                                     approximated_jump_location=psi_jump_loc, known_jump_loc=True)
                psi_vals[moy + oy, ix] = sj.func_val_at_x(x=x, reconstruction_order=ro, func_coeff_array=func_coeff,
                                                          jump_loc=psi_jump_loc, jump_mag_array=approx_psi_jump_mag)
        f_tilde = mpm.matrix(ny, nx)
        jump_curve_tilde = mpm.matrix(nx, 1)
        # calculatinf f_tilde and xi_tilde
        print('calculating f_tilde and xi_tilde\n')
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

        plot_approximation_of_f_and_jump_curve(X=X, Y=Y, reconstruction_order=ro, moy_val=moy,  f_tilde_vals=f_tilde,
                                               jump_curve_tilde=jump_curve_tilde, test_func_type=tf.get_func_type(),
                                               show_plot=show_plot, save_plot=save_plot)

    # exact_f_and_xi(True, False)
    f_tilde_and_xi_tilde(True, False)
    f_tilde_xi_tilde_jump_mag_tilde(True, False)
