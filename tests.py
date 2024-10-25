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
    a = mpm.matrix([[1, 2], [3, 4], [5, 6]])
    print(a)
    print()
    a0 = a[0, :1]
    print(a0)
    # aa0 = a[:, 0]
    # print(aa0)




