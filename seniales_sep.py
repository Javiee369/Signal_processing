import numpy as np
from random import random
import matplotlib.pyplot as plt

def signal_generation():
    # Parameters
    F = 60
    Ts = 1 / (100 * F)
    t = np.arange(0.0, 5 / 60, Ts)
    rs = np.random.RandomState()
    N = t.size
    Vf = np.empty([3, N], dtype=complex)
    Vf[0] = 220 * np.sqrt(2) * np.exp(2j * np.pi * F * t)
    Vf[1] = Vf[0] * np.exp(-2j * np.pi / 3)
    Vf[2] = Vf[1] * np.exp(-2j * np.pi / 3)
    # Wire impedance
    Zl1 = 0.009j
    Zl2 = 0.01j
    Zl3 = 0.01 + 0.001j
    # Node 1 measurements
    Z1 = 15j * random()
    V1 = Z1 / (Z1 + Zl1) * Vf
    I1 = V1 / Z1
    # Node 2 measurements
    Z2 = 0.1 + 1.5j * random()
    V2 = Z2 / (Z2 + Zl2) * Vf
    I2 = V2 / Z2
    # Node 3 measurements
    Z3 = np.array([20, 30, 25])[:, None] * random()
    V3 = Z3 / (Z3 + Zl3) * Vf
    I3 = V3 / Z3
    data = {
        "Node 1": (V1.real, I1.real),
        "Node 2": (V2.real, I2.real),
        "Node 3": (V3.real, I3.real),
    }
    return data

if __name__ == '__main__':
    pass