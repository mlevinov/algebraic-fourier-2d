import numpy as np
import mpmath as mpm
import constants as const


def mpmath_array_to_numpy_array(mp_arr):
    if mp_arr.cols == 1 and mp_arr.rows == 1:

    if (mp_arr.cols == 1 and mp_arr.rows >1) or (mp_arr.cols > 1 and mp_arr.rows == 1):