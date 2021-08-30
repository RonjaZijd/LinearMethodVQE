import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import LMLibrary2 as LM
import HamiltoniansLibrary as HML

np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

#inputformat: 

#Hamiltonian
#H_VQE_coeffs = [-0.24274280513140462,-0.24274280513140462,0.1777128746513994,0.17771287465139946,0.12293305056183798,0.12293305056183798,0.1676831945771896,0.1676831945771896,0.17059738328801052,0.17627640804319591,-0.04475014401535161,-0.04475014401535161,0.04475014401535161,0.04475014401535161]
#H_VQE_gates = [['Z3'],['Z4'],['Z2'],['Z1'],['Z1', 'Z3'],['Z2','Z4'],['Z1','Z4'],['Z2','Z3'],['Z1','Z2'],['Z3', 'Z4'], ['Y1', 'Y2', 'X3', 'X4'],['X1', 'X2', 'Y3', 'Y4'], ['Y1', 'X2', 'X3', 'Y4'], ['X1', 'Y2', 'Y3', 'X4'] ]
#H_coeffs = [(1-1j)/4, (1-1j)/4, (1-1j)/4, (3+1j)/2]
#H_gates = [['X1', 'X2'], ['Y1', 'Y2'], ['Z1', 'Z2'], ['I1', 'I2']]

#H_coeffs = [0.25, -0.25, -0.25, 0.25]
#H_gates = [['I1', 'I2'], ['I1', 'Z2'], ['Z1', 'I2'], ['Z1', 'Z2']]

#symbols = ["H", "H", "H", "H"]
#coordinates = np.array([0.0, 0.0, 0.371322, 0.0, 0.0, -0.371322, 0.0, 0.0, 0.7426644, 0.0, 0.0, -0.7426644])

# symbols = ['Li', 'H']
# coordinates = np.array([0.0, 0.0, 0.403635, 0.0, 0.0, -1.210905])
# H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
# print(qubits)
# H_VQE_coeffs = H.coeffs
# H_VQE_gates = LM.hamiltonian_into_gates(H)

# print(H_VQE_gates)
H_VQE_gates = HML.H_LiH_gates
H_VQE_coeffs = HML.H_LiH_coeffs
print(len(H_VQE_gates))
print(len(H_VQE_coeffs))

X = [[0, 1], [1,0]]
I = [[1, 0], [0, 1]]
Z = [[1, 0], [0, -1]]
Y = [[0, -1j], [1j, 0]]

gates_dict = {"X":X, "Y":Y, "Z":Z, "I":I}

def creating_sub_matrix(array):
    """Using Kronecker multiplication to multiply all the qubits out, starting at the end of the array."""
    St = np.kron(gates_dict.get(str(array[-2])), gates_dict.get(str(array[-1])))
    for i in range(len(array)-2):
        St = np.kron(gates_dict.get(str(array[-3-i])), St)
    return St

def exact_energ(H_gates, H_coeffs, tot_wires):
    """Calculating the exact energy of a hamiltonian"""
    Hamil_matrix = np.zeros(shape=(np.power(2, tot_wires), np.power(2, tot_wires)), dtype = np.complex128)
    for i in range(len(H_gates)):
        matrices_on_wires = np.full((tot_wires), "I", dtype=str) #preparing string of gates on each qubit, starting with ['I', 'I', etc..]
        for j in range(len(H_gates[i])):
            as_characters = list(H_gates[i][j])
            gate = as_characters[0]
            wire = int(as_characters[1])
            matrices_on_wires[wire-1] = gate #filling in gates from input: ['I', 'Z', 'I', ....]
        sub_mat = creating_sub_matrix(matrices_on_wires)
        Hamil_matrix += H_coeffs[i]*sub_mat #adding unto Hamiltonian Matrix

    eigvals, eigvecs = sp.linalg.eig(Hamil_matrix)
    lowest_energ = eigvals[np.argmin(eigvals)]
    print(Hamil_matrix)
    return(lowest_energ)

print(exact_energ(H_VQE_gates, H_VQE_coeffs, 4))
