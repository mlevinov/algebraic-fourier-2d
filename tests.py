import sys
import mpmath as mpm
import numpy as np
import matplotlib.pyplot as plt
import single_jump_param_recovery as sjpr
from test_functions import TestFunctions
import constants as const
from tqdm import tqdm
from mpmath_tools import mpm_matrix_to_mpmath_numpy_array as mp_mat_to_np_mat
if __name__ == "__main__":
    print(mpm.mp)
    print(mpm.mp.dps)
    x = mpm.pi + 1j*5
    print(x)
    # s = mpm.nstr(x, 15)
    # print(s)





