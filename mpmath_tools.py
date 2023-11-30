import numpy as np
import mpmath as mpm
import constants as const


def numpy_array_to_mpmath_matrix(np_arr):
    return mpm.matrix(np_arr.tolist())
def mpm_matrix_to_mpmath_numpy_array(mpmath_arr):
    if isinstance(mpmath_arr, np.ndarray):
        return mpmath_arr
    # TODO: understand how to check if the input is of type mpmath.matrix
    # elif isinstance(mpmath_arr, mpm.matrix):
    #     print('not an mpmath matrix and not a numpy array type')
    #     return 1
    else:
        mat_type = _check_type_of_mpmath_matrix(mpmath_arr)
        return np.array(mpmath_arr.tolist(), dtype=mat_type)
def find_max_val_index(mpmath_arr):
    np_arr = mpm_matrix_to_mpmath_numpy_array(mpmath_arr)
    ind = np_arr.unravel_index(np.argmax(np_arr, axis=None), np_arr.shape)
    # TODO: returning a tuple of max_val_index, consider returning separately ind[0], ind[1]
    return ind
def find_min_val_index(mpmath_arr):
    np_arr = mpm_matrix_to_mpmath_numpy_array(mpmath_arr)
    ind = np_arr.unravel_index(np.argmin(np_arr, axis=None), np_arr.shape)
    # TODO: returning a tuple of max_val_index, consider returning separately ind[0], ind[1]
    return ind
def _mpmath_num_to_numpy(mpmath_num):
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
def elementwise_norm_matrix(mp_arr1, mp_arr2):
    try:
        try:
            norm_mat = mp_arr1 - mp_arr2
        except ValueError as ve:
            print('incompatible dimensions for subtraction, trying transposing')
            norm_mat = mp_arr1 - mp_arr2.T
    except ValueError as ve:
        print(ve)
        exit(1)
    print('transposing worked, continuing\n')
    rows = norm_mat.rows
    cols = norm_mat.cols
    for r in range(rows):
        for c in range(cols):
            norm_mat[r, c] = mpm.norm(norm_mat[r, c], p=2)
    return norm_mat


if __name__ == '__main__':
    mp_mat1 = mpm.randmatrix(1,3)
    mp_mat2 = mpm.randmatrix(3, 1)

    print(mp_mat1)
    print()
    print(mp_mat2)
    print()
    diff_mat = elementwise_norm_matrix(mp_mat1, mp_mat2)
    print(diff_mat)