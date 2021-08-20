import pennylane as qml
from pennylane import numpy as np
import scipy as sp


np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

#inputformat: 
#H_coeffs = [0.45, 0.34, 0.11, 0.62] #let's keep with the small one here and change to the big actual hamiltonian later.
#H_gates = [['Z2', 'X1'], ['Y2', 'X1', 'Z4'], ['Z3'], ['X2', 'Y1'] ]

H_coeffs = [-0.24274280513140462,-0.24274280513140462,-0.04207897647782276,0.1777128746513994,0.17771287465139946,0.12293305056183798,0.12293305056183798,0.1676831945771896,0.1676831945771896,0.17059738328801052,0.17627640804319591,-0.04475014401535161,-0.04475014401535161,0.04475014401535161,0.04475014401535161]
H_gates = [['Z3'],['Z4'],['I1'],['Z2'],['Z1'],['Z1', 'Z3'],['Z2','Z4'],['Z1','Z4'],['Z2','Z3'],['Z1','Z2'],['Z3', 'Z4'], ['Y1', 'Y2', 'X3', 'X4'],['X1', 'X2', 'Y3', 'Y4'], ['Y1', 'X2', 'X3', 'Y4'] ]



def from_gates_to_gates(wire, gate):
    X = [[0, 1], [1,0]]
    I = [[1, 0], [0, 1]]
    Z = [[1, 0], [0, -1]]
    Y = [[0, -1j], [1j, 0]]
    G1 = I
    G2 = I
    G3 = I
    G4 = I  
    if wire==1:
        if gate=='X':
            G1=X
        if gate=='Y': 
            G1=Y
        if gate=='Z':
            G1=Z
    if wire==2: 
        if gate=='X':
            G2=X
        if gate=='Y':
            G2=Y 
        if gate=='Z':
            G2=Z
    if wire==3:
        if gate=='X':
            G3=X
        if gate=='Y':
            G3=Y 
        if gate=='Z':
            G3=Z
    if wire==4: 
        if gate=='X':
            G4=X
        if gate=='Y':
            G4=Y 
        if gate=='Z':
            G4=Z
    return G1, G2, G3, G4
    
def creating_sub_matrix(G1, G2, G3, G4):
    M = np.kron(G3, G4)
    M2 = np.kron(G2, M)
    M3 = np.kron(G1, M2)
    return M3 

##first creating hamilmatrix of the correct size, for 4 wires this is 16 by 16
Hamil_matrix = np.zeros(shape=(16, 16), dtype=np.complex128)

for i in range(len(H_gates)):
    for j in range(len(H_gates[i])):
        as_characters = list(H_gates[i][j])
        gate = as_characters[0]
        wire = int(as_characters[1])
        G1, G2, G3, G4 = from_gates_to_gates(wire, gate)
    sub_mat = creating_sub_matrix(G1,G2,G3,G4)
    #print(sub_mat.shape)
    #print(Hamil_matrix.shape)
    Hamil_matrix = Hamil_matrix + H_coeffs[i]*sub_mat

eigvals, eigvecs = sp.linalg.eig(Hamil_matrix)
print(eigvals[np.argmin(eigvals)])

#print(Hamil_matrix)

#to do here: 
    #checking that it makes the hamiltonian matrix
    #write function to get the (lowest eigenvalue out of this)
    #put it all into seperate functions