import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

U_gates = np.array([['ROT'], ['ROT'], ['ROT'], ['ROT']])
U_generators = np.array([['CROT'], ['CROT'], ['CROT'], ['CROT']])

Thets = np.random.normal(0, np.pi, (4,3,1))

print(Thets)

