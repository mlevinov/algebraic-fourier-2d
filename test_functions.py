import mpmath as mpm
import numpy as np
import single_jump_param_recovery as sjpr
from tqdm import tqdm
import constants as const
import mpmath_tools as mpt


class TestFunctions:
    def __init__(self, dps=const.DEFAULT_DPS, func_type=const.FUNC_TYPE_F1):
        self.dps = dps
        self.func_type = func_type
        if func_type == const.FUNC_TYPE_F1:
            self.smoothness_order = const.F1_SMOOTHNESS_ORDER
        else:
            self.smoothness_order = const.F2_F3_SMOOTHNESS_ORDER

    def get_jump_magnitudes_of_fx(self, x):
        r"""
        Get the exact jump magnitudes of a slice Fx

        Args:
            x: (mpmath.mpf) a point in :math:`[-\pi, \pi)`

        Returns:
            returns the exact jump magnitudes of :math:`F_x` in a mpmath.matrix type

        """
        if self.func_type == const.FUNC_TYPE_F1:
            jump_magnitudes = mpm.matrix(const.F1_SMOOTHNESS_ORDER + 1, 1)
            for l in range(const.F1_SMOOTHNESS_ORDER + 1):
                jump_magnitudes[l, 0] = mpm.fmul(mpm.fdiv(x, const.MP_PI), mpm.fdiv(1, l + 1))
                return jump_magnitudes
        else:
            return mpm.ones(const.F2_F3_SMOOTHNESS_ORDER + 1, 1)

    def get_func_type(self):
        return self.func_type

    def get_smoothness_order(self):
        return self.smoothness_order

    def get_func_val_at_point(self, x, y):
        if self.func_type == const.FUNC_TYPE_F1:
            return self.__get_func_type_1_val_at_point(x, y)
        elif self.func_type == const.FUNC_TYPE_F2:
            return self.__get_func_type_2_val_at_point(x, y)
        else:
            return self.__get_func_type_3_val_at_point(x, y)

    def get_jump_loc_of_fx(self, x):
        r"""
        Get the exact jump location of a slice :math:`F_x`

        Args:
            x: (mpmath.mpf) a point in :math:`[-\pi, \pi)`

        Returns:
            returns the exact value for the jump location of a slice :math:`F_x` of type mpmath.mpf
        """
        if self.func_type == const.FUNC_TYPE_F1 or self.func_type == const.FUNC_TYPE_F2:
            return x
        else:
            return x / 2

    def __get_func_type_1_val_at_point(self, x, y):
        new_x = mpm.fsub(y, x)
        jump_magnitudes = self.get_jump_magnitudes_of_fx(x)
        s = sjpr.phi_func_val_at_x(x=new_x, reconstruction_order=const.F1_SMOOTHNESS_ORDER,
                                   jump_loc=const.DEFAULT_JUMP_LOC, jump_mag_array=jump_magnitudes)
        return s

    def __get_func_type_2_val_at_point(self, x, y):
        new_x = mpm.fsub(y, x)
        jump_magnitudes = self.get_jump_magnitudes_of_fx(x)
        s = sjpr.phi_func_val_at_x(x=new_x, reconstruction_order=const.F2_F3_SMOOTHNESS_ORDER,
                                   jump_loc=const.DEFAULT_JUMP_LOC, jump_mag_array=jump_magnitudes)
        return s

    def __get_func_type_3_val_at_point(self, x, y):
        new_x = mpm.fsub(y, mpm.fdiv(x, 2))
        jump_magnitudes = self.get_jump_magnitudes_of_fx(x)
        # jump_magnitudes = mpm.matrix(1, const.F2_F3_SMOOTHNESS_ORDER + 1)
        # for l in range(const.F2_F3_SMOOTHNESS_ORDER + 1):
        #     jump_magnitudes[0, l] = 1
        s = sjpr.phi_func_val_at_x(x=new_x, reconstruction_order=const.F2_F3_SMOOTHNESS_ORDER, jump_loc=const.DEFAULT_JUMP_LOC,
                                   jump_mag_array=jump_magnitudes)
        return s

    def get_func_slice_at_x(self, x, Y):
        r"""
        Get the exact values of a slice :math:`F_x`

        Args:
            x: (mpmath.mpf) a point in :math:`[-\pi, \pi)`
            Y: (mpmath.matrix or numpy.ndarray) a subset of :math:`[-\pi, \pi)`

        Returns:
            returns a mpmath.matrix which represents the exact
            values of :math:`F_x` over :math:`Y \subset [-\pi, \pi)`
        """
        ny = len(Y)
        func_values_at_x = mpm.matrix(ny, 1)
        if isinstance(Y, mpm.matrix) and Y.rows != ny:
            Y = Y.T
        elif isinstance(Y, np.ndarray) and Y.shape[0] != ny:
            Y = Y.T
        for iy in range(ny):
            func_values_at_x[iy, 0] = self.get_func_val_at_point(x=x, y=Y[iy])
        return func_values_at_x

    def get_func_val(self, X, Y):
        """
        get the exact values of :math:`F`

        Args:
            X: (mpmath.matrix or numpy.ndarray) :math:`X \subseteq [-\pi,\pi)`
            Y: (mpmath.matrix or numpy.ndarray) :math:`Y \subseteq [-\pi,\pi)`

        Returns:
            returns a mpmath.matrix representing
            :math:`F` over :math:`X\times Y`
        """
        ny = len(Y)
        nx = len(X)
        func_val = mpm.matrix(ny, nx)
        try:
            if X.cols != 1:
                X = X.T
            if Y.cols != 1:
                Y = Y.T
        except AttributeError:
            print('assuming X and Y are numpy ndarray type')
            YY = mpm.matrix(Y)
            XX = mpm.matrix(X)
            for ix in tqdm(range(nx)):
                func_val[:, ix] = self.get_func_slice_at_x(XX[ix, 0], YY)
            return func_val
        for ix in tqdm(range(nx)):
            func_val[:, ix] = self.get_func_slice_at_x(X[ix, 0], Y)
        return func_val

    def get_func_discontinuity_curve(self, nx=64):
        r"""
        Get the discontinuity curve of :math:`F`
        
        Args:
            nx: (int) number of points to sample from :math:`[-\pi, \pi)` for the discontinuity curve.
        
        Returns:
            returns a mpmath.matrix representing the discontinuity curve of :math:`F`
        """
        X = np.linspace(-np.pi + const.EPS, np.pi, nx)
        if self.func_type == const.FUNC_TYPE_F1 or self.func_type == const.FUNC_TYPE_F2:
            return X
        else:
            return X / 2

    def get_func_fourier_coefficient(self, ox, oy):
        if self.func_type == const.FUNC_TYPE_F1:
            return self.__get_fourier_func_type_1_coefficient(ox, oy)
        elif self.func_type == const.FUNC_TYPE_F2:
            return self.__get_fourier_func_type_2_coefficient(ox, oy)
        elif self.func_type == const.FUNC_TYPE_F3:
            return self.__get_fourier_func_type_3_coefficient(ox, oy)
        else:
            return 1

    def get_func_fourier_coefficient_const_oy_range_ox(self, num_of_oxs, oy):
        r"""
        Get Fourier coefficients for the specific test function where oy is constant.
        those coefficients are for approximating the :math:`\psi` function which in turn will
        be used as an approximation of the Fourier coefficients of :math:`F`
        
        Args:
            num_of_oxs: (int) number of points in :math:`[-\pi,\pi)`
            oy: (int) a constant for calculating :math:`\left\{\ c_{oy,ox}(F)\right\}_{|ox|\leq M}`
        
        Returns:
            returns a mpmath.matrix array with coefficients for :math:`\psi_{\omega_y}`
        """
        m = num_of_oxs
        coeff_array = mpm.matrix(2 * m + 1, 1)
        for ox in range(-m, m + 1):
            coeff_array[m + ox, 0] = self.get_func_fourier_coefficient(ox, oy)
        return coeff_array

    def compute_1d_fourier_of_fx(self, x, Y, M, N):
        Z = np.zeros(shape=Y.shape, dtype=complex)
        x_arr = np.full(Y.shape, x)
        for m in range(-M, M + 1):
            for n in range(-N, N + 1):
                cmn = mpt.mpmath_num_to_numpy(self.get_func_fourier_coefficient(m, n))
                Z += cmn * np.exp(1j * (m * x_arr + n * Y))
        return Z

    def __get_fourier_func_type_1_coefficient(self, ox, oy):
        if oy == 0 or ox == -oy:
            return 0
        else:
            a1 = mpm.fmul(60, mpm.power(oy, 5))
            a2 = mpm.fmul(30j, mpm.power(oy, 4))
            a3 = mpm.fmul(20, mpm.power(oy, 3))
            a4 = mpm.fmul(15j, mpm.power(oy, 2))
            a5 = mpm.fmul(12, oy)

            a6 = mpm.fsub(a1, a2)
            a7 = mpm.fsub(a4, a3)
            a8 = mpm.fsub(a5, 10j)
            a9 = mpm.fadd(a6, mpm.fadd(a7, a8))
            a10 = mpm.power(-1, mpm.fadd(ox, oy))

            a = mpm.fmul(a10, a9)

            b1 = mpm.fmul(120, mpm.power(oy, 6))
            b2 = mpm.fadd(ox, oy)
            b3 = mpm.power(mpm.pi, 2)
            b4 = mpm.fmul(b2, b3)

            b = mpm.fmul(b1, b4)
            s = mpm.fdiv(a, b)
            return s

    def __get_fourier_func_type_2_coefficient(self, ox, oy):
        if ox == -oy and oy != 0:
            a1 = mpm.fsub(mpm.power(oy, 2), 1)
            a2 = mpm.fsub(mpm.power(oy, 6), mpm.power(oy, 4))
            a3 = mpm.fsub(mpm.power(oy, 10), mpm.power(oy, 8))

            a4 = mpm.fadd(a1, mpm.fadd(a2, a3))
            a5 = mpm.fmul(mpm.fmul(2j, mpm.fsub(1j, oy)), a4)
            a = mpm.fmul(mpm.pi, a5)

            b1 = mpm.power(oy, 12)
            b = mpm.fmul(b1, mpm.power(mpm.fmul(2, mpm.pi), 2))
            s = mpm.fdiv(a, b)

            return s
        else:
            return 0

    def __get_fourier_func_type_3_coefficient(self, ox, oy):
        if oy == 0:
            return 0
        else:
            a1 = mpm.fsub(mpm.power(oy, 10), mpm.power(oy, 8))
            a2 = mpm.fsub(mpm.power(oy, 6), mpm.power(oy, 4))
            a3 = mpm.fsub(mpm.power(oy, 2), 1)

            a4 = mpm.fadd(a1, mpm.fadd(a2, a3))
            a5 = mpm.fmul(1j, mpm.fsub(1j, oy))
            a6 = mpm.fmul(a5, a4)

            if oy != -2 * ox:
                aa1 = mpm.fmul(ox, mpm.pi)
                aa2 = mpm.fdiv(mpm.fmul(oy, mpm.pi), 2)
                into_sin = mpm.fadd(aa1, aa2)
                a = mpm.fmul(a6, mpm.sin(into_sin))

                b1 = mpm.fmul(mpm.power(oy, 12), mpm.power(mpm.pi, 2))
                b2 = mpm.fadd(2 * ox, oy)
                b = mpm.fmul(b1, b2)
                s = mpm.fdiv(a, b)
                return s
            # n == -2 * m:
            else:
                b = mpm.fmul(mpm.power(oy, 12), mpm.fmul(2, mpm.pi))
                s = mpm.fdiv(a6, b)
                return s
