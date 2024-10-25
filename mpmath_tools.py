import mpmath as mpm
import numpy as np


def numpy_array_to_mpmath_matrix(np_arr):
    """
    converting from numpy ndarray to mpmath.mtrix type

    Args:
        np_arr: (ndarray) numpy ndarry to convert to mpmath.matrix

    Returns:
        mpmath.matrix: a matrix of type mpmath.matrix from np_arr

    """
    return mpm.matrix(np_arr.tolist())


def mpm_matrix_to_mpmath_numpy_array(mpmath_arr):
    """
    Converting a mpmath matrix to a ndarry

    Args:
        mpmath_arr: (mpmath.matrix) a matrix of type mpmath.matrix to convert

    Returns:
        ndarray: returns a ndarray from the given mpmath.matrix

    """
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
    """
    finding the index of the maximum value in a given matrix

    Args:
        mpmath_arr: (mpmath.matrix) a matrix containing the maximal value

    Returns:
        tuple: row and column indexes for the maximal value in mpmath_arr

    """
    np_arr = mpm_matrix_to_mpmath_numpy_array(mpmath_arr)
    ind = np.unravel_index(np.argmax(np_arr, axis=None), np_arr.shape)
    return ind


def find_min_val_index(mpmath_arr):
    """
        finding the index of the minimal value in a given matrix

        Args:
            mpmath_arr: (mpmath.matrix) a matrix containing the minimal value

        Returns:
            tuple: row and column indexes for the minimal value in mpmath_arr

        """
    np_arr = mpm_matrix_to_mpmath_numpy_array(mpmath_arr)
    ind = np.unravel_index(np.argmin(np_arr, axis=None), np_arr.shape)
    return ind


def mpmath_num_to_numpy(mpmath_num):
    """
    Converting a mpmath type value to a numpy type value

    Args:
        mpmath_num: (mpmath type) a value represented by the mpmath library

    Returns:
        numpy: same value as mpmath_num but using numpy library

    """
    if isinstance(mpmath_num, mpm.mpf):
        s = float(mpm.nstr(mpmath_num, mpm.mp.dps))
        return s
    elif isinstance(mpmath_num, mpm.mpc):
        re = mpm.nstr(mpm.re(mpmath_num), n=mpm.mp.dps)
        im = mpm.nstr(mpm.im(mpmath_num), n=mpm.mp.dps)
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
    """
    Calculates :math:`\\lvert \\text{mp_arr1[i,j]-mp_arr2[i,j]} \\rvert` for each :math:`1\\leq i\\leq m`
    and :math:`1\\leq j\\leq n` where mp_arr1, mp_arr2 are of order :math:`m\\times n`

    Args:
        mp_arr1: (mpmath.matrix)
        mp_arr2: (mpmath.matrix)

    Returns:
        mpmath.matrix: a matrix where each element is the absolut value of the difference between mp_arr1[i,j]
        and mp_arr2[i,j]

    """
    try:
        try:
            norm_mat = mp_arr1 - mp_arr2
        except ValueError:
            print('incompatible dimensions for subtraction, trying transposing')
            norm_mat = mp_arr1 - mp_arr2.T
            print('transposing worked, continuing\n')
    except ValueError as ve:
        print(ve)
        exit(1)
    rows = norm_mat.rows
    cols = norm_mat.cols
    for r in range(rows):
        for c in range(cols):
            norm_mat[r, c] = mpm.fabs(norm_mat[r, c])
    return norm_mat

