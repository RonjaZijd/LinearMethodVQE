import pennylane as qml
from pennylane import numpy as np

#####To Do in Code:
    #start counting from 0 everywhere                       #done
    #make input accessible and check with different ones
    #potentially extend it to a small VQE case
    #put the U gates on wires other than 1
    #put in Identity C gate.

#######################    input information        ##############################################################
U_gates = np.array(['RX', 'RY', 'RZ', 'RX', 'RX', 'RY']) ##the U gates in order
no_of_gates = len(U_gates)
Thets = np.array([0.34, 0.21, 0.78, 0.45, 0.99, 0.21])



U_gener = np.array(['CNOT', 'CY', 'CZ', 'CNOT', 'CNOT', 'CY'])
H_coeffs = [0.45, 0.34, 0.11, 0.62]
H_gates = [['Z2', 'X1'], ['Y2', 'X1', 'X2'], ['Z3'], ['X2', 'Y1'] ]
no_of_wires = 4
matrix_length=6

######################      gates                    #############################################################
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

#######################      subcircuits        ###################################################################
def circ_creator(int1, int2, U_gates, U_gener, Thets): ##clean up later by putting it into one big numpy array
    i=0
    qml.Hadamard(wires=0) ##putting wire=0 into the +-state
    while i!=(int1):
        gate_creator(U_gates[i], Thets[i],1)
        i=i+1
    qml.PauliX(wires=0)
    gate_creator(U_gener[int1], 0, 1)
    qml.PauliX(wires=0)
    while i!=(int2):
        gate_creator(U_gates[i], Thets[i], 1)
        i=i+1
    gate_creator(U_gener[int2], 0, 1)
    
def up_to_un_circ(int2, int_max, U_gates, U_gener, Thets):
    i=int2
    while i<int_max+1:
        gate_creator(U_gates[i], Thets[i], 1)
        i=i+1

def c_notting_hamil(input_array):
    i=0
    while i<len(input_array):
        as_characters = list(input_array[i])
        gate = as_characters[0]
        wire = int(as_characters[1])
        gate_creator(gate, 0, wire)
        i=i+1


##############################################     Actual QNodes       #############################################
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def real_circ_S(int1, int2, U_gates, U_gener, Thets):
    circ_creator(int1, int2, U_gates, U_gener, Thets)
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

@qml.qnode(dev)
def imagin_circ_S(int1, int2, U_gates, U_gener, Thets): ##to get imaginary part: same circuit but then measure PauliY at the end
    circ_creator(int1, int2, U_gates, U_gener, Thets)
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliY(wires=0))

@qml.qnode(dev)
def real_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, inp_array):
    circ_creator(int1, int2, U_gates, U_gener, Thets)
    up_to_un_circ(int2, mat_len, U_gates, U_gener, Thets) 
    c_notting_hamil(inp_array.numpy())
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

@qml.qnode(dev)
def imagin_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, inp_array):
    circ_creator(int1, int2, U_gates, U_gener, Thets)
    up_to_un_circ(int2, mat_len, U_gates, U_gener, Thets) 
    c_notting_hamil(inp_array.numpy())
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliY(wires=0))


#################################    Final Calculations   ##########################################

def total_ham_element(int1, int2, mat_len, U_gates, U_gener, Thets, hamiltonian_array, Hamil_coefs):
    Ham = 0
    i=0
    while i<len(hamiltonian_array):
        small_h_real = real_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, hamiltonian_array[i])
        small_h_imag = imagin_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, hamiltonian_array[i])
        small_h = small_h_real + small_h_imag
        #Ham = Ham + small_h
        Ham = Ham + Hamil_coefs[i]*small_h     
        i=i+1

    return Ham

def H_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length, Hamil_array, Hamil_coeffs):
    H_matrix = np.empty(shape=(matrix_length, matrix_length))
    i=0
    while i<matrix_length:
        j=0
        while j<matrix_length:
            if j>i or j==i:
                H = total_ham_element(i, j, matrix_length, U_gates, U_gener, Thets, Hamil_array, Hamil_coeffs)
                H_matrix[i][j] = H
                H_matrix[j][i] = H
            j=j+1
        i=i+1
    
    return H_matrix

def S_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length):
    S_matrix = np.empty(shape=(matrix_length,matrix_length))
    i=0
    while i<matrix_length:
        j=0
        while j<matrix_length:
            if j>i or j==i:
                real_part = real_circ_S(i,j, U_gates, U_gener, Thets)
                #print(real_circ_S.draw())
                imaginary_part = imagin_circ_S(i,j,U_gates,U_gener, Thets)

                ###putting it into an S matrix: 
                S_matrix[i][j] = real_part+imaginary_part
                S_matrix[j][i] = real_part+imaginary_part

                ####printing it out to check the values
                #print("Real: ", real_part)
                #print("Imaginary: ", imaginary_part) 
                #print()
            j=j+1
        i=i+1

    return S_matrix



########################                     MAIN                     #################################

##example h-circuit
Inp = ['Z1', 'X2', 'Y3', 'X1']
real_circ_h(1, 5, 5, U_gates, U_gener, Thets, Inp)
print(real_circ_h.draw())
print(total_ham_element(1, 5, 5, U_gates, U_gener, Thets, H_gates, H_coeffs))

#H circuit
print("The H matrix: ")
print(H_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length-1, H_gates, H_coeffs))
print("The S matrix")
print(S_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length-1))