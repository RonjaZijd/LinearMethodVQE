import numpy as np
from scipy import linalg as ln
import scipy as sp
np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

matrixA = np.loadtxt('matrixH_k0.31', delimiter=',', dtype=np.complex)
matrixB = np.loadtxt('matrixS', delimiter=',', dtype=np.complex)

print("Condition Numbers of matrixA and of matrixB:")
print(np.linalg.cond(matrixA))
print(np.linalg.cond(matrixB))

eigvals_eig, eigvecs_eig = sp.linalg.eig(matrixA, matrixB)
eigvals_eigh, eigvecs_eigh = sp.linalg.eigh(matrixA, matrixB)

print("Eigenvectors eig:")
print(eigvecs_eig)
print()
print("Eigenvectors eigh:")
print(eigvecs_eigh)

lowest_vec_eig = eigvecs_eig[:,np.argmin(np.real(eigvals_eig))]
lowest_vec_eigh = eigvecs_eigh[np.argmin(np.real(eigvals_eigh))]

print(np.dot(lowest_vec_eig, lowest_vec_eigh))

