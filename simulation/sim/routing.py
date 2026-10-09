import numpy as np

def escalate(s, cap=0.15):
    thr = np.quantile(s, 1 - cap)
    return s >= thr
