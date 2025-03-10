import mpmath as mpm
import numpy as np

MP_PI = mpm.pi
NP_PI = np.pi
DEFAULT_DPS = 20

EPS = pow(10, -1)
DEFAULT_JUMP_LOC = 0
FUNC_TYPE_F1 = 1
F1_SMOOTHNESS_ORDER = 5
FUNC_TYPE_F2 = 2
F2_F3_SMOOTHNESS_ORDER = 11
FUNC_TYPE_F3 = 3
TWO_PI = mpm.fmul(2, mpm.pi)
PSI_JUMP_LOC = -MP_PI
