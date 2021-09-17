import numpy as np
from scipy import linalg as ln
import scipy as sp

np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

#########################Setting up
size = 4
A = np.random.normal(0, 1000, (size,size))
B = np.random.normal(0, 1000, (size,size))
matrixA = (A + A.T)/2 ##this makes the matrices symmetric
matrixB = (B + B.T)/2

#########################Making B positive definite
eigvaly_B, eigvecy_B = np.linalg.eigh(matrixB)
additi = np.abs(np.min(np.array(eigvaly_B)))
for i in range(len(matrixB)):
    for j in range(len(matrixB[i])):
        if i==j: #aka the diagonal
            matrixB[i][j] = matrixB[i][j]+additi+0.00001

eigvaly_B_after_pos_def, eigvecty_after_pos_def = np.linalg.eigh(matrixB) #can be printed as a check

eigvalsB_mat = np.zeros((size, size), dtype=np.complex128)
eigvalsB_mat_special = np.zeros((size, size), dtype = np.complex128)

###########################Start of Algorithm
eigvals_B, eigvecs_B = np.linalg.eig(matrixB)
#eigvecs_B = np.matrix.transpose(eigvecs_B) #to have the columns be the eigenvectors as in the paper

for i in range(size):
    for j in range(size):
        if i==j: #only along the diagonal
            eigvalsB_mat[i][j] = eigvals_B[i]
            if eigvals_B[i] == 0: 
                eigvalsB_mat_special[i][j] = 0
            else:  
                eigvalsB_mat_special[i][j] = 1/(np.sqrt(eigvals_B[i]))

atest = np.matmul(eigvalsB_mat, eigvalsB_mat_special)
print(np.matmul(eigvalsB_mat_special, atest))   #thus my eigvalsB_mat_special is correctly done

eigvecB_tilde = np.matmul(eigvecs_B, eigvalsB_mat_special) #we don't have to transpose B_tilde, because it's already made with the correct things
A1 = np.matmul(matrixA, eigvecB_tilde)
matrixA_tilde = np.matmul((np.matrix.transpose(eigvecB_tilde)), A1) #creating A-tilde
eigvalsA, eigvecsA = np.linalg.eig(matrixA_tilde)
#eigvecsA = np.matrix.transpose(eigvecsA)
final_eigvec = np.matmul(eigvecB_tilde, eigvecsA)
print("The final eigenvalues without normalizing them are: ")
print(final_eigvec)
for i in range(len(final_eigvec)):
    final_eigvec[i] = final_eigvec[i]/final_eigvec[i][0] #need to properly normalize this
print("The eigenvalues are: ")
print(eigvalsA)
print("The eigenvectors are: ")
print(final_eigvec)

print("And using scipy they would be: ")
eigvals, eigvecs = ln.eig(matrixA, matrixB)
print("The SciPy eigenvalues are: ")
print(eigvecs)
for i in range(len(eigvecs)):
    eigvecs[i] = eigvecs[i]/eigvecs[i][0]
print(eigvals)
print(eigvecs)

