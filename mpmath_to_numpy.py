import numpy as np
import mpmath as mpm
import constants as const


def convert_mpm_to_mpmath_numpy(mpmath_num):
    if isinstance(mpmath_num, mpm.mpf):
        s = float(mpm.nstr(mpmath_num, mpm.mp.dps))
        return s
    elif isinstance(mpmath_num, mpm.mpc):
        re = mpm.nstr(mpm.re(mpmath_num), mpm.mp.dps)
        im = mpm.nstr(mpm.im(mpmath_num), mpm.mp.dps)
        s = complex(float(re), float(im))
        return s
    elif isinstance(mpmath_num, float) or isinstance(mpmath_num, complex) or isinstance(mpmath_num, int):
        return mpmath_num
    else:
        print('not an mpmath number type')
        return 1
def mpm_array_to_mpmath_numpy_array(mpmath_arr):
    if isinstance(mpmath_arr, np.ndarray):
        return mpmath_arr
    elif isinstance(mpmath_arr, mpm.matrix):
        print('not an mpmath matrix and not a numpy array type')
        return 1
    else:
        mat_type = _check_type_of_mpmath_matrix(mpmath_arr)
        return np.array(mpmath_arr.tolist(), dtype=mat_type)

def _check_type_of_mpmath_matrix(mpmath_mat):
    if not isinstance(mpmath_mat, mpm.matrix):
        print('not an mpmath matrix')
        return 1
    cols = mpmath_mat.cols
    rows = mpmath_mat.rows
    for r in range(rows):
        for c in range(cols):
            if isinstance(mpmath_mat[r, c], mpm.mpc):
                return complex
    return float