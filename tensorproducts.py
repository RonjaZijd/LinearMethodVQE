import pennylane as qml
from pennylane import numpy as np
import scipy as sp


np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

#inputformat: 
#H_coeffs = [0.45, 0.34, 0.11, 0.62] #let's keep with the small one here and change to the big actual hamiltonian later.
#H_gates = [['Z2', 'X1'], ['Y2', 'X1', 'Z4'], ['Z3'], ['X2', 'Y1'] ]

H_coeffs = [-0.24274280513140462,-0.24274280513140462,-0.04207897647782276,0.1777128746513994,0.17771287465139946,0.12293305056183798,0.12293305056183798,0.1676831945771896,0.1676831945771896,0.17059738328801052,0.17627640804319591,-0.04475014401535161,-0.04475014401535161,0.04475014401535161,0.04475014401535161]
H_gates = [['Z3'],['Z4'],['I1'],['Z2'],['Z1'],['Z1', 'Z3'],['Z2','Z4'],['Z1','Z4'],['Z2','Z3'],['Z1','Z2'],['Z3', 'Z4'], ['Y1', 'Y2', 'X3', 'X4'],['X1', 'X2', 'Y3', 'Y4'], ['Y1', 'X2', 'X3', 'Y4'] ]
tot_wires = 4
Hamil_matrix = np.zeros(shape=(tot_wires*tot_wires, tot_wires*tot_wires), dtype = np.complex128)

X = [[0, 1], [1,0]]
I = [[1, 0], [0, 1]]
Z = [[1, 0], [0, -1]]
Y = [[0, -1j], [1j, 0]]


gates_dict = {"X":X, "Y":Y, "Z":Z, "I":I}

#matrices_on_wires = numpy.full((1, tot_wires), I)

def creating_sub_matrix(array):
    print("Test2")
    print(array[-2])
    St = np.kron(gates_dict.get(str(array[-2])), gates_dict.get(str(array[-1])))
    print("Test2.5")
    print(St)
    for i in range(len(array)-2):
        St = np.kron(gates_dict.get(str(array[-3-i])), St)
    print("Test3")
    return St

for i in range(len(H_gates)):
    matrices_on_wires = np.full((tot_wires), "I", dtype=str)
    for j in range(len(H_gates[i])):
        as_characters = list(H_gates[i][j])
        gate = as_characters[0]
        print("Test0")
        wire = int(as_characters[1])
        matrices_on_wires[wire-1] = gate
    print(matrices_on_wires)
    print("Test1")
    sub_mat = creating_sub_matrix(matrices_on_wires)
    Hamil_matrix = Hamil_matrix + sub_mat

eigvals, eigvecs = sp.linalg.eig(Hamil_matrix)
print(eigvals[np.argmin(eigvals)])

print(Hamil_matrix)

#to do here: 
    #checking that it makes the hamiltonian matrix
    #write function to get the (lowest eigenvalue out of this)
    #put it all into seperate functions