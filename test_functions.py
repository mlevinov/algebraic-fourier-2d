import mpmath as mpm
import numpy as np
import matplotlib.pyplot as plt
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
    def get_jump_magnitudes_at_x(self, x):
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
    def __get_func_type_1_val_at_point(self, x, y):
        new_x = mpm.fsub(y, x)
        jump_magnitudes = self.get_jump_magnitudes_at_x(x)
        s = sjpr.phi_func_val_at_x(x=new_x, reconstruction_order=const.F1_SMOOTHNESS_ORDER,
                                   jump_loc=const.DEFAULT_JUMP_LOC, jump_mag_array=jump_magnitudes)
        return s
    def __get_func_type_2_val_at_point(self, x, y):
        new_x = mpm.fsub(y, x)
        jump_magnitudes = self.get_jump_magnitudes_at_x(x)
        s = sjpr.phi_func_val_at_x(x=new_x, reconstruction_order=const.F2_F3_SMOOTHNESS_ORDER,
                                   jump_loc=const.DEFAULT_JUMP_LOC, jump_mag_array=jump_magnitudes)
        return s
    def __get_func_type_3_val_at_point(self, x, y):
        new_x = mpm.fsub(y, mpm.fdiv(x, 2))
        jump_magnitudes = self.get_jump_magnitudes_at_x(x)
        # jump_magnitudes = mpm.matrix(1, const.F2_F3_SMOOTHNESS_ORDER + 1)
        # for l in range(const.F2_F3_SMOOTHNESS_ORDER + 1):
        #     jump_magnitudes[0, l] = 1
        s = sjpr.phi_func_val_at_x(x=new_x, reconstruction_order=const.F2_F3_SMOOTHNESS_ORDER,jump_loc=const.DEFAULT_JUMP_LOC,
                                   jump_mag_array=jump_magnitudes)
        return s
    def get_func_slice_at_x(self, x, Y):
        # assuming Y is a mpm matrix of order n * 1 and returning same order matrix
        ny = len(Y)
        func_values_at_x = mpm.matrix(ny, 1)
        try:
            if Y.cols != 1:
                Y = Y.T
            for iy in range(ny):
                func_values_at_x[iy, 0] = self.get_func_val_at_point(x=x, y=Y[iy, 0])
            return func_values_at_x
        except AttributeError:
            print('assuming Y is a numpy array')
            for iy in range(ny):
                func_values_at_x[iy, 0] = self.get_func_val_at_point(x=x, y=Y[iy])
            return func_values_at_x
    def get_func_val(self, X, Y):
        # assuming Y and X are column mpmath vectors (of order n * 1 and m * 1 respectively)
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
            for ix in range(nx):
                func_val[:, ix] = self.get_func_slice_at_x(XX[ix, 0], YY)
            return func_val
        for ix in range(nx):
            func_val[:, ix] = self.get_func_slice_at_x(X[ix, 0], Y)
        return func_val
    def get_func_discontinuity_curve(self, nx=64):
        X = np.linspace(-np.pi + const.EPS, np.pi, nx)
        if self.func_type == const.FUNC_TYPE_F1 or self.func_type == const.FUNC_TYPE_F2:
            return X
        else:
            return X/2
    def get_func_fourier_coefficient(self, ox, oy):
        if self.func_type == const.FUNC_TYPE_F1:
            return self.__get_fourier_func_type_1_coefficient(ox, oy)
        elif self.func_type == const.FUNC_TYPE_F2:
            return self.__get_fourier_func_type_2_coefficient(ox, oy)
        elif self.func_type == const.FUNC_TYPE_F3:
            return self.__get_fourier_func_type_3_coefficient(ox, oy)
        else:
            return 1
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
    def compute_1D_fourier_at_x(self, x, Y, mox, moy, func_type):
        z = np.zeros(shape=Y.shape, dtype=complex)
        x_arr = np.full(Y.shape, x)
        for ox in range(-mox, mox + 1):
            for oy in range(-moy, moy + 1):
                if func_type == const.FUNC_TYPE_F1:
                    cmn = mpt.mpmath_num_to_numpy(self.__get_fourier_func_type_1_coefficient(ox, oy))
                elif func_type == const.FUNC_TYPE_F2:
                    cmn = mpt.mpmath_num_to_numpy(self.__get_fourier_func_type_2_coefficient(ox, oy))
                # func_type = 3
                else:
                    cmn = mpt.mpmath_num_to_numpy(self.__get_fourier_func_type_3_coefficient(ox, oy))
                z += cmn * np.exp(1j * (ox * x_arr + oy * Y))
        return mpt.numpy_array_to_mpmath_matrix(z)