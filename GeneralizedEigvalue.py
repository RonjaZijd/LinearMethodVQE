import numpy as np
from scipy import linalg as ln
import scipy as sp

np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

#matrixA = np.array([[4,2,1,6], [0,0,3,1], [6,7,1,8], [2,1,5,0]])  ###symmetric
#matrixB = np.array([[0,4,1,3], [1,3,4,5], [2,3,9,4], [1,2,0,1]])

size = 4

A = np.random.normal(0, 1000, (size,size))
B = np.random.normal(0, 1000, (size,size))
matrixA = (A + A.T)/2 ##this makes the matrices symmetric
matrixB = (B + B.T)/2
print(matrixA)

#now making B positive definite:
eigvaly_B, eigvecy_B = np.linalg.eigh(matrixB)
additi = np.abs(np.min(np.array(eigvaly_B)))
for i in range(len(matrixB)):
    for j in range(len(matrixB[i])):
        if i==j: #aka the diagonal
            matrixB[i][j] = matrixB[i][j]+additi

eigvaly_B_after_pos_def = np.linalg.eigh(matrixB)
print(eigvaly_B_after_pos_def)
#apparently I now have infs or nan's somewhere

eigvalsB_mat = np.zeros((size, size), dtype=np.complex128)
eigvalsB_mat_special = np.zeros((size, size), dtype = np.complex128)

eigvals_B, eigvecs_B = np.linalg.eig(matrixB)
eigvecs_B = np.matrix.transpose(eigvecs_B) #to have the columns be the eigenvectors

for i in range(size):
    for j in range(size):
        if i==j: #only along the diagonal
            eigvalsB_mat[i][j] = eigvals_B[i]
            if eigvals_B[i] == 0: 
                eigvalsB_mat_special[i][j] = 0
            else:  
                eigvalsB_mat_special[i][j] = 1/(np.sqrt(eigvals_B[i]))

eigvecB_tilde = np.matmul(eigvecs_B, eigvalsB_mat_special) #we don't have to transpose B_tilde, because it's already made with the correct things
A1 = np.matmul(matrixA, eigvecB_tilde)
matrixA_tilde = np.matmul((np.matrix.transpose(eigvecB_tilde)), A1) #seems to work
eigvalsA, eigvecsA = np.linalg.eig(matrixA_tilde)
eigvecsA = np.matrix.transpose(eigvecsA)
final_eigvec = np.matmul(eigvecB_tilde, eigvecsA)

print("The eigenvalues are: ")
print(eigvalsA)
print("The eigenvectors are: ")
print(final_eigvec)

print("And using scipy they would be: ")
eigvals, eigvecs = ln.eig(matrixA, matrixB)
print(eigvals)
print(eigvecs)

