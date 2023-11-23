import mpmath as mpm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# constants:
eps = pow(10,-1)
F_TYPE_F1 = 1
F1_SMOOTHNESS_ORDER = 5
F_TYPE_F2 = 2
F2_SMOOTHNESS_ORDER = 11
F_TYPE_F3 = 3
F3_SMOOTHNESS_ORDER = 11

class TestFunc:
    def __int__(self, dps=20, func_type=F_TYPE_F1, m_ox=10, m_oy = 100, func_smoothness_order = 0):
        # create test for valid func_type input
        mpm.mp.dps = dps
        self.func_type = func_type
        self.m_ox = m_ox
        self.m_oy = m_oy
        self.func_smoothness_order = func_smoothness_order

    def get_func_val_at_point(self, x, y):
        if self.func_type == F_TYPE_F1:
            return self.__get_func_type_1_val_at_point(x, y)
        elif self.func_type == F_TYPE_F2:
            return self.__get_func_type_2_val_at_point(x, y)
        else:
            return self.__get_func_type_3_val_at_point(x, y)
    def __get_func_type_1_val_at_point(self, x, y):
        new_x = mpm.fsub(y, x)
        jump_magnitudes = mpm.matrix(1, F1_SMOOTHNESS_ORDER + 1)
        for l in range(F1_SMOOTHNESS_ORDER + 1):
            a1 = mpm.fdiv(x, mpm.pi)
            a2 = mpm.fdiv(1, mpm.fadd(l, 1))
            a3 = mpm.fmul(a1, a2)
            jump_magnitudes[0, l] = a3
        res = phi_at_x(x=new_x, degree_of_phi=F1_SMOOTHNESS_ORDER, jump_location=0, jump_magnitudes=jump_magnitudes)
        return res

    def __get_func_type_2_val_at_point(self, x, y):
        new_x = mpm.fsub(y, x)
        jump_magnitudes = mpm.matrix(1, F2_SMOOTHNESS_ORDER + 1)
        for l in range(F2_SMOOTHNESS_ORDER + 1):
            jump_magnitudes[0, l] = 1
        res = phi_at_x(x=new_x, degree_of_phi=F2_SMOOTHNESS_ORDER, jump_location=0, jump_magnitudes=jump_magnitudes)
        return res

    def __get_func_type_3_val_at_point(self, x, y):
        new_x = mpm.fsub(y, mpm.fdiv(x, 2))
        jump_magnitudes = mpm.matrix(1, F3_SMOOTHNESS_ORDER + 1)
        for l in range(F3_SMOOTHNESS_ORDER + 1):
            jump_magnitudes[0, l] = 1
        res = phi_at_x(x=new_x, degree_of_phi=F3_SMOOTHNESS_ORDER, jump_location=0, jump_magnitudes=jump_magnitudes)
        return res

    def get_func_slice_at_x(self, x, Y):
        # assuming Y is a mpmath matrix of order n * 1 and returning same order matrix
        ny = len(Y)
        func_values_at_x = mpm.matrix(ny, 1)
        for iy in range(ny):
            func_values_at_x[iy, 0] = self.get_func_val_at_point(x=x, y=Y[iy, 0])
        return func_values_at_x

    def get_func_val(self, X, Y):
#         assuming Y is mpmath matrix of order n * 1 and X is of 1 * m
        ny = len(Y)
        nx = len(X)
        func_val = mpm.matrix(ny, nx)
        for ix in range(nx):
            func_val[:, ix] = self.get_func_slice_at_x(X[0, ix], Y)
        return func_val

    # def plot(self, nx, ny, Z):
    #     # assuming that X, Y are lists or arrays, NOT of type mpmath
    #     XV, YV = np.meshgrid(X, Y)
    #     ax = plt.axes(projection='3d')
    #     ax.plot_surface(XV, YV, np.real(Z))
    #     ax.scatter(X, Y, np.ones(shape=len(X)))
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.view_init(30, 45)
    #     plt.grid(False)
    #     plt.axis('off')
    #     plt.show()

