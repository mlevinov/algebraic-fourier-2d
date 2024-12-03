import sys
import mpmath
import numpy as np
import matplotlib.pyplot as plt
import single_jump_param_recovery as sjpr
from test_functions import TestFunctions
import constants as const
from tqdm import tqdm
from mpmath_tools import mpm_matrix_to_mpmath_numpy_array as mp_mat_to_np_mat


def is_numpy_array(arg):
    return isinstance(arg, np.ndarray)


def is_mpmath_matrix(arg):
    return isinstance(arg, mpmath.matrix)


if __name__ == "__main__":
    # # Examples
    # arr = np.array([1, 2, 3])
    # not_arr = [1, 2, 3]
    #
    # print(is_numpy_array(arr))  # Output: True
    # print(is_numpy_array(not_arr))  # Output: False

    # Examples
    mat = mpmath.matrix([[1, 2], [3, 4]])
    not_mat = [[1, 2], [3, 4]]

    print(is_mpmath_matrix(mat))  # Output: True
    print(is_mpmath_matrix(not_mat))  # Output: False