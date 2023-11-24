import mpmath as mpm
import numpy as np

# Constants
TWO_PI = mpm.fmul(2, mpm.pi)
class SingleJumpParamRecovery:

    def __init__(self, dps=20):
        mpm.mp.dps = dps
    def __vn_func_val_at_x(self, x, n, jump_loc):
        c = mpm.fdiv(-mpm.power(TWO_PI, n), mpm.factorial(n + 1))
        z = mpm.fsub(x, jump_loc)
        zz = 0
        if 0 <= z < TWO_PI:
            s = mpm.fmul(c, mpm.bernpoly(n + 1, mpm.fdiv(z, TWO_PI)))
        else: # -TWO_PI <= z < 0:
            s = mpm.fmul(c, mpm.bernpoly(n + 1, mpm.fdiv(mpm.fadd(z, TWO_PI), TWO_PI)))
        return s
    def __calc_coeff_phi(self, k, reconstruction_order, jump_loc, jump_mag):
        if k==0:
            return 0
        else:
            r1 = mpm.fdiv(mpm.expj(-mpm.fmul(jump_loc, k)), TWO_PI)
            r2 = 0
            for l in range(reconstruction_order + 1):
                r3 = mpm.fdiv(jump_mag[0, l], mpm.power(mpm.fmul(1j, k), l + 1))
                r2 = mpm.fadd(r2, r3)
            return mpm.fmul(r1, 2)
    def __calc_coeff_psi(self, func_coeff_at_k, phi_coeff_at_k):
        return mpm.fsub(func_coeff_at_k, phi_coeff_at_k)
    def phi_func_val_at_x(self, x, reconstruction_order, jump_loc, jump_mag):
        d = reconstruction_order
        s = 0
        for l in range(d + 1):
            a_l = jump_mag[0, l]
            vl = self.__vn_func_val_at_x(x, l, jump_loc)
            s1 = mpm.fmul(a_l, vl)
            s = mpm.fadd(s, s1)
        return s
    def psi_func_val_at_x(self, x, reconstruction_order, func_coeff, jump_loc, jump_mag):
        m = int(np.floor(func_coeff.cols / 2))
        s = 0
        for k in range(-m, m + 1):
            phi_k_coeff = self.__calc_coeff_phi(k, reconstruction_order=reconstruction_order,
                                                jump_loc=jump_loc, jump_mag=jump_mag)
            psi_k_coeff = self.__calc_coeff_psi(func_coeff_at_k=func_coeff[0,m+k],
                                                 phi_coeff_at_k=phi_k_coeff)
            s = mpm.fadd(s, mpm.fmul(psi_k_coeff, mpm.expj(mpm.fmul(k, x))))
        return s

    def func_val_at_x(self, x, reconstruction_order, func_coeff, jump_loc, jump_mag):
        phi_val = self.phi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, jump_loc=jump_loc,
                                         jump_mag=jump_mag)
        psi_val = self.psi_func_val_at_x(x=x, reconstruction_order=reconstruction_order, jump_loc=jump_loc,
                                         jump_mag=jump_mag)
        return mpm.fadd(psi_val, phi_val)