import sys
import mpmath as mpm
import numpy as np
import matplotlib.pyplot as plt
import single_jump_param_recovery as sjpr
from test_functions import TestFunctions
import constants as const

if __name__ == "__main__":
    a = mpm.matrix([[1],[2],[3],[4]])
    print(a)
    print(a.rows)
    print(a[1:, 0])



