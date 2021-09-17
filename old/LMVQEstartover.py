import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#prerequisites
I_mat = [[1,0], [0,1]]
np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

########################   INPUT      #################################################

#The circuit: 
U_gates = np.array([['RX', 'RY', 'RZ'], ['RX', 'RY', 'RZ'], ['RX', 'RY', 'RZ'], ['RX', 'RY', 'RZ']]) ##the U gates in order
U_gener = np.array([['CNOT', 'CY', 'CZ'], ['CNOT', 'CY', 'CZ'], ['CNOT', 'CY', 'CZ'], ['CNOT', 'CY', 'CZ']])
entangle_gates = np.array([[2,3], [2,0], [3,1]]) ###the entangled gates at the end

#The hamiltonian:
H_VQE_coeffs = [-0.24274280513140462,-0.24274280513140462,-0.04207897647782276,0.1777128746513994,0.17771287465139946,0.12293305056183798,0.12293305056183798,0.1676831945771896,0.1676831945771896,0.17059738328801052,0.17627640804319591,-0.04475014401535161,-0.04475014401535161,0.04475014401535161,0.04475014401535161]
H_VQE_gates = [['Z3'],['Z4'],['I1'],['Z2'],['Z1'],['Z1', 'Z3'],['Z2','Z4'],['Z1','Z4'],['Z2','Z3'],['Z1','Z2'],['Z3', 'Z4'], ['Y1', 'Y2', 'X3', 'X4'],['X1', 'X2', 'Y3', 'Y4'], ['Y1', 'X2', 'X3', 'Y4'] ]
Hamilt_written_outt = -0.24274280513140462*qml.PauliZ(wires=2) + -0.24274280513140462*qml.PauliZ(wires=3) +  0.1777128746513994*qml.PauliZ(wires=1)+0.17771287465139946*qml.PauliZ(wires=0)+0.12293305056183798*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))+0.12293305056183798*(qml.PauliZ(wires=1)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=0)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=1)@qml.PauliZ(wires=2)) +0.17059738328801052*(qml.PauliZ(wires=0)@qml.PauliZ(wires=1))+0.17627640804319591*(qml.PauliZ(wires=2)@qml.PauliZ(wires=3))+-0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliY(wires=1)@qml.PauliX(wires=2)@qml.PauliX(wires=3))+-0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliX(wires=1)@qml.PauliY(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliX(wires=1)@qml.PauliX(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliY(wires=1)@qml.PauliY(wires=2)@qml.PauliX(wires=3))
Hamilt_written_out = -0.2*qml.PauliZ(wires=2) + -0.56*qml.PauliZ(wires=3) + 0.122*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))

#other: 
no_of_wires =5
no_of_gates = len(U_gates)
Thets = np.random.normal(0, np.pi, (4,3))
matrix_length=Thets.size

####################################### Gates creators #################################
def gate_creator(string, thet, wir): ##a function to create the gates
    if string=='RX':
        return qml.RX(thet, wires=wir)
    if string=='RY':
        return qml.RY(thet, wires=wir)
    if string=='RZ':
        return qml.RZ(thet, wires=wir)
    if string=='CNOT' or string=='X':
        return qml.CNOT(wires=[0,wir])
    if string=='CY' or string=='Y':
        return qml.CY(wires=[0,wir])
    if string=='CZ' or string=='Z':
        return qml.CZ(wires=[0,wir])
    if string=='I':
        return qml.QubitUnitary(I_mat, wires=wir)

def entangler(wir1, wir2, plus_ancil):
    if plus_ancil == True:
        return qml.CNOT(wires=[wir1+1, wir2+1])
    else: 
        return qml.CNOT(wires=[wir1, wir2])

##################################### Circuits creators ###############################

def circ_creator(int1, int2, U_gates, U_gener, Thets): ##clean up later by putting it into one big numpy array
    i=0
    j=0 
    numbers_had=0               
    qml.Hadamard(wires=0) ##putting wire=0 into the +-state
    while i<len(U_gates) and numbers_had!=int1:
        while j<len(U_gates[i]) and numbers_had!=int1:
            gate_creator(U_gates[i][j], Thets[i][j], i+1)
            j=j+1
            numbers_had=numbers_had+1
        if j==len(U_gates[i]):
            i=i+1
            j=0 ##whenever we go to a new wire, we reset j
    qml.PauliX(wires=0)
    gate_creator(U_gener[i][j], 0, i+1)                          
    qml.PauliX(wires=0)
    while i<len(U_gates) and numbers_had!=int2:
        while j<len(U_gates[i]) and numbers_had!=int2:
            gate_creator(U_gates[i][j], Thets[i][j], i+1)
            j=j+1
            numbers_had=numbers_had+1
        if j==len(U_gates[i]):
            j=0
            i=i+1
    gate_creator(U_gener[i][j], 0, i+1)
    return i,j ##returns the i and j element where it was left off.

def up_to_un_circ(int2, i, j, int_max, U_gates, U_gener, Thets):  ##I and J is where we left off, if we put that in as 0 and 0, it will just go through all the gates
    numbers_had=int2
    while numbers_had<int_max:
        while j<len(U_gates[i]) and numbers_had!=int_max:
            gate_creator(U_gates[i][j], Thets[i][j], i+1)
            j=j+1
            numbers_had=numbers_had+1
        if j==len(U_gates[i]):
            j=0
            i=i+1

def final_entangled_gates_circ(entangle_gates):
    for i in range(len(entangle_gates)):
        entangler(entangle_gates[i][0], entangle_gates[i][1], True)

def c_notting_hamil(input_array): ##doesn't change when extending it to multiple wires.
    i=0
    while i<len(input_array):
        as_characters = list(input_array[i])
        gate = as_characters[0]
        wire = int(as_characters[1])
        gate_creator(gate, 0, wire)
        i=i+1

def circuit(params, wires): #standard circuit (directly from the PennylaneDemo)
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
    for i in wires:
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

###################################   Actual QNodes  ######################################
dev = qml.device('default.qubit', wires=5)
dev2 = qml.device('default.qubit', wires=4)  #dev2 used for directly calculating the energy

@qml.qnode(dev)  #assuming that these two are still correct
def real_circ_S(int1, int2, U_gates, U_gener, Thets):
    circ_creator(int1, int2, U_gates, U_gener, Thets)
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

@qml.qnode(dev)
def real_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, inp_array, entangle_gates):
    i, j = circ_creator(int1, int2, U_gates, U_gener, Thets)
    up_to_un_circ(int2, i, j, mat_len, U_gates, U_gener, Thets) 
    final_entangled_gates_circ(entangle_gates)
    c_notting_hamil(inp_array.numpy())
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev)
def imagin_circ_S(int1, int2, U_gates, U_gener, Thets): ##to get imaginary part: same circuit but then measure PauliY at the end
    circ_creator(int1, int2, U_gates, U_gener, Thets)
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliY(wires=0))
##################################  Calculations   ########################################

def S_alternative_way(U_gates, U_gener, Thets, matrix_lenth):
    S_matrix = np.zeros(shape=(matrix_length, matrix_length), dtype=np.complex128)
    for i in range(matrix_length):
        for n in range(matrix_length):
            if n==i:
                S_matrix[i][n] = 1 #setting diagonal elements to 1
            if n>i: 
                real_part = real_circ_S(i, n, U_gates, U_gener, Thets)
                S_matrix[i][n] = 2*real_part
                S_matrix[n][i] = 2*real_part
    return S_matrix

def total_ham_element(int1, int2, mat_len, U_gates, U_gener, Thets, hamiltonian_array, Hamil_coefs, entangle_gates):
    Ham = 0
    i=0
    while i<len(hamiltonian_array):
        small_h_real = real_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, hamiltonian_array[i], entangle_gates)
        small_h = small_h_real ##keeping it all real for the moment
        Ham = Ham + Hamil_coefs[i]*small_h  
        i=i+1
    
    return Ham

def H_alternative_way(U_gates, U_gener, Thets, matrix_length, Hamil_array, Hamil_coeffs, entangle_gates):
    H_matrix = np.zeros(shape=(matrix_length, matrix_length), dtype=np.complex128)
    for i in range(matrix_length):
        for n in range(matrix_length):
            if n==i:
                H = total_ham_element(i, n, matrix_length, U_gates, U_gener, Thets, Hamil_array, Hamil_coeffs, entangle_gates)
                H_matrix[i][n] = H
            if n>i:
                H = total_ham_element(i, n, matrix_length, U_gates, U_gener, Thets, Hamil_array, Hamil_coeffs, entangle_gates) 
                H_matrix[i][n] = 2*H
                H_matrix[n][i] = 2*H #twice the real part
    return H_matrix

def H_tilde_matrix(H_matrix, E_0, E_grad, k):  ##E_0 can be calculated using E_calc
    mat_len = len(H_matrix)+1 #the len thing might not work the way which I want it to work 
    H_tilde_matrix = np.empty(shape=(mat_len, mat_len), dtype=np.complex128)
    H_tilde_matrix[0][0] = E_0
    for j in range(E_grad.size):
        H_tilde_matrix[0][j+1] = E_grad[0][j]
        H_tilde_matrix[j+1][0] = E_grad[0][j]
    for i in range(len(H_matrix)):
        for j in range(len(H_matrix[i])):
            if i==j: ##so only diagonal elements 
                H_tilde_matrix[i+1][j+1] = H_matrix[i][j] + k       ##only to the diagonal elements
            else:
                H_tilde_matrix[i+1][j+1] = H_matrix[i][j]          
    return H_tilde_matrix

def S_tilde_matrix(S_matrix):
    mat_len = len(S_matrix)+1
    S_tilde_matrix = np.empty(shape=(mat_len, mat_len), dtype=np.complex128)
    S_tilde_matrix[0][0] = 1
    for i in range(mat_len-1):
        S_tilde_matrix[0][i+1] = 0
        S_tilde_matrix[i+1][0] = 0 
    for i in range(len(S_matrix)):
        for j in range(len(S_matrix[i])):
            S_tilde_matrix[i+1][j+1] = S_matrix[i][j]
    return S_tilde_matrix

def E_grad(Thets, Hamiltonian, circuit, device):
    energy_func = qml.ExpvalCost(circuit, Hamiltonian, device)
    grad_func = qml.grad(energy_func)
    E_gra = grad_func(Thets)
    E_gra = np.reshape(E_gra, (1, E_gra.size))
    return E_gra

def energy_calc(circuit, Hamilt_written_out, device, Thets):
    costy = qml.ExpvalCost(circuit, Hamilt_written_out, device)  ###for now we're simply going to do it with the specific case circuit
    energ = costy(Thets)
    return energ

def S_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length):
    S_matrix = np.zeros(shape=(matrix_length,matrix_length), dtype=np.complex128)
    i=0
    while i<matrix_length:
        n=0
        while n<matrix_length:
            if n>i or n==i:
                real_part = real_circ_S(i,n, U_gates, U_gener, Thets)
                #print(real_circ_S.draw())
                imaginary_part = imagin_circ_S(i,n,U_gates,U_gener, Thets)

                ###putting it into an S matrix: 
                S_matrix[i][n] = real_part+imaginary_part*1j
                S_matrix[n][i] = real_part-imaginary_part*1j ###conjugate and imaginary!


            n=n+1
        i=i+1

    return S_matrix
################################ Main trying out ###################################
H = H_alternative_way(U_gates, U_gener, Thets, matrix_length, H_VQE_gates, H_VQE_coeffs, entangle_gates)
S = S_alternative_way(U_gates, U_gener, Thets, matrix_length)

print("This is H: ")
print(H)
print()
print("This is S: ")
print(S)
print()

H_tilde = H_tilde_matrix(H, energy_calc(circuit, Hamilt_written_outt, dev2, Thets), E_grad(Thets, Hamilt_written_outt, circuit, dev2), 0) #the first k is going to be 0
S_tilde = S_tilde_matrix(S)

print("This is H_tilde: ")
print(H_tilde)
print()
print("This is S_tilde: ")
print(S_tilde)
print()

print("Is S positive definite: ")
print(np.all(np.linalg.eigvals(S)>0))
print()
print("Is S-tilde postive definite: ")
print(np.all(np.linalg.eigvals(S_tilde)>0))
print()


tol = 0.0000000000001  ###the Tolerance as computers aren't exact or perfect
print("Is H symmetric: ")
print(np.all(np.abs(H-H.T)<tol))
print()
print("Is H-tilde symmetric: ")
print(np.all(np.abs(H_tilde-H_tilde.T)<tol))
print()
print("Is S symmetric: ")
print(np.all(np.abs(S-S.T)<tol))
print()
print("Is S-tilde symmetric: ")
print(np.all(np.abs(S_tilde-S_tilde.T)<tol))
print()

eigvals, eigvecs = sp.linalg.eig(H_tilde, S_tilde)
print("These are the eigenvalues of the generalized equation: ")
print(eigvals)
print()
print("These are the eigenvalues of S: ")
print(np.linalg.eigvals(S))
print()
print("These are the eigenvalues of S-Tilde: ")
print(np.linalg.eigvals(S_tilde))
print()

eigvals_S, eigvecs_S = sp.linalg.eig(S)
eigvals_S_tilde, eigvecs_S_tilde = sp.linalg.eig(S_tilde)

print("This is the last eigenvector of S:")
print(eigvecs_S[-1])
print()

print("This is the second to last eigenvector of S-tilde: ")
print(eigvecs_S_tilde[-2])
print()
