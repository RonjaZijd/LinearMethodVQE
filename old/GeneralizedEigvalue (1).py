import numpy as np
from scipy import linalg as ln
import scipy as sp

np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.3f}'.format})
np.random.seed(73)
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

# [D] The above loop can be carried out by using the numpy identity matrix:
#matrixB += np.eye(size)*(additi+0.00001)


eigvaly_B_after_pos_def, eigvecty_after_pos_def = np.linalg.eigh(matrixB) #can be printed as a check

eigvalsB_mat = np.zeros((size, size), dtype=np.complex128)
eigvalsB_mat_special = np.zeros((size, size), dtype = np.complex128)

###########################Start of Algorithm
eigvals_B, eigvecs_B = np.linalg.eig(matrixB)
# [D] This was too much:
#eigvecs_B = np.matrix.transpose(eigvecs_B) #to have the columns be the eigenvectors as in the paper

for i in range(size):
    for j in range(size):
        if i==j: #only along the diagonal
            eigvalsB_mat[i][j] = eigvals_B[i]
            # [D] We made B such that there will be _no_ 0 eigenvalue (because of the 0.00001)
            if eigvals_B[i] == 0: 
                eigvalsB_mat_special[i][j] = 0
            else:  
                eigvalsB_mat_special[i][j] = 1/(np.sqrt(eigvals_B[i]))

atest = np.matmul(eigvalsB_mat, eigvalsB_mat_special)
# [D] Commented this out for now
#print(np.matmul(eigvalsB_mat_special, atest))   #thus my eigvalsB_mat_special is correctly done

eigvecB_tilde = np.matmul(eigvecs_B, eigvalsB_mat_special) #we don't have to transpose B_tilde, because it's already made with the correct things
A1 = np.matmul(matrixA, eigvecB_tilde)
matrixA_tilde = np.matmul((np.matrix.transpose(eigvecB_tilde)), A1) #creating A-tilde
eigvalsA, eigvecsA = np.linalg.eig(matrixA_tilde)
# [D] This was too much:
#eigvecsA = np.matrix.transpose(eigvecsA)
final_eigvec = np.matmul(eigvecB_tilde, eigvecsA)

print()
print("The eigenvalues are: ")
print(eigvalsA)
print("The eigenvectors are: ")
# print(final_eigvec)
# [D] Print the normalized eigenvectors instead:
print(final_eigvec/ln.norm(final_eigvec, 2, 0))
print()

print("And using scipy they would be: ")
eigvals, eigvecs = ln.eig(matrixA, matrixB)
print(eigvals)
print(eigvecs)
print()


# [D] 
def gen_eigh(A, B):
    """Solve the generalized eigenvalue problem Av = lambda Bv."""

    # Some tests that there is no wrong inputs
    #  - Test for Hermiticity of input matrices
    assert np.allclose(B, B.T.conj()) and np.allclose(A, A.T.conj())
    #  - Test for positivity of B
    assert np.min(ln.eigvalsh(B))>0.0

    # Solve eigenvalue problem for B
    Lambda_B, Phi_B = ln.eig(B)
    # Define the modified eigenvectors of B (@ is np.matmul)
    Phi_B_tilde = Phi_B @ np.diag(Lambda_B**(-1/2))
    # Define transformed A matrix
    A_tilde = Phi_B_tilde.T @ A @ Phi_B_tilde
    # Solve eigenvalue problem for transformed A
    Lambda_A, Phi_A = ln.eig(A_tilde)
    # The eigenvalues of transformed A are the generalized eigenvalues
    Lambda = Lambda_A
    # The backtransformed eigenvectors of the transformed A are the gen. eigenvectors
    Phi = Phi_B_tilde @ Phi_A
    # Bonus: Normalize the columns (i.e. the eigenvectors) to 1
    Phi /= ln.norm(Phi, 2, axis=0)

    return Lambda, Phi

Lambda, Phi = gen_eigh(matrixA, matrixB)
print("Via new function:", Lambda, Phi, sep='\n')

print("Eigenvector of lowest eigenvalue: ")
print("SciPy algorithm: ")
print(eigvecs[np.argmin(eigvals)])
print("Own function: ")
print(Phi[np.argmin(Lambda)])
