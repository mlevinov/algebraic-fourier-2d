import mpmath as mpm
import numpy as np

# used for [-pi, pi) segment
EPS = pow(10, -1)
# indicators for our test functions
FUNC_TYPE_F1 = 1
F1_SMOOTHNESS_ORDER = 5
FUNC_TYPE_F2 = 2
F2_F3_SMOOTHNESS_ORDER = 11
FUNC_TYPE_F3 = 3
# commonly used values
TWO_PI = mpm.fmul(2, mpm.pi)
MP_PI = mpm.pi
NP_PI = np.pi