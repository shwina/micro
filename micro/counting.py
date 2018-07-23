import numpy as np
from numba import autojit, prange

@autojit
def external_match(a):
    if np.sum(a) == 1:
        return True
    return False

@autojit
def internal_match(a):
    if np.sum(a) == 3:
        return True
    return False

@autojit(parallel=True, nopython=True)
def count_objects(img):
    ny, nx = img.shape

    for j in range(nx):
        img[0, j] = 0
        img[-1, j] = 0

    for i in range(ny):
        img[i, 0] = 0
        img[i, -1] = 0

    E = 0
    I = 0
    for i in prange(ny-1):
        for j in range(nx-1):
            if external_match(img[i:i+2, j:j+2]):
                E += 1
            if internal_match(img[i:i+2, j:j+2]):
                I += 1
    return (E - I)/4
