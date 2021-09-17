import pennylane as qml
from pennylane import numpy as np

#####To Do in Code:
    #start counting from 0 everywhere                       #done
    #make input accessible and check with different ones    #done
    #potentially extend it to a small VQE case              #maybe extend it to the example VQE problem
    #put the U gates on wires other than 1                  #done
    #do randomized input on the parameters
    #put in Identity C gate.

#######################    input information        ##############################################################
U_gates = np.array([['RX', 'RY', 'RZ', 'RY'], ['RX', 'RX'], ['RZ', 'RY', 'RZ', 'RY', 'RX'], ['RZ', 'RZ', 'RX']]) ##the U gates in order
no_of_gates = len(U_gates)
#write a short for loop:

#Thets = np.empty(shape=np.shape(U_gates))
Thets = []
temp_thet = []

for i in range(len(U_gates)):###Not figuring this out at the moment: problem = having trouble creating random entries which has the exact size as the U_gates array
    #do a save as file and go unto trying to do the small VQE case which is in the demo. 
    temp_thet=[]
    for j in range(len(U_gates[i])):
        p = np.random.normal(0, np.pi)
        print(i)
        print(j)
        print(p)
        #Thets[i] = np.append(Thets,p)
        temp_thet = np.append(temp_thet, p)
    Thets = np.vstack((Thets, temp_thet))
 
## Thets = np.array([[0.34, 0.21, 0.78, 0.44], [0.45, 0.99], [0.34, 0.51, 0.22, 0.84, 0.02], [0.69, 0.21, 0.5]])



U_gener = np.array([['CNOT', 'CY', 'CZ', 'CY'], ['CNOT', 'CNOT'], ['CZ', 'CY', 'CZ', 'CY', 'CNOT'], ['CZ', 'CZ', 'CNOT']])
H_coeffs = [0.45, 0.34, 0.11, 0.62]
H_gates = [['Z2', 'X1'], ['Y2', 'X1', 'Z4'], ['Z3'], ['X2', 'Y1'] ]
no_of_wires =5
matrix_length=13

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
     
def up_to_un_circ(int2, i, j, int_max, U_gates, U_gener, Thets):
    numbers_had=int2
    while numbers_had<int_max+1:
        while j<len(U_gates[i]) and numbers_had!=int_max+1:
            gate_creator(U_gates[i][j], Thets[i][j], i+1)
            j=j+1
            numbers_had=numbers_had+1
        if j==len(U_gates[i]):
            j=0
            i=i+1

def c_notting_hamil(input_array): ##doesn't change when extending it to multiple wires.
    i=0
    while i<len(input_array):
        as_characters = list(input_array[i])
        gate = as_characters[0]
        wire = int(as_characters[1])
        gate_creator(gate, 0, wire)
        i=i+1


##############################################     Actual QNodes       #############################################
dev = qml.device('default.qubit', wires=5)

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
    i, j = circ_creator(int1, int2, U_gates, U_gener, Thets)
    up_to_un_circ(int2, i, j, mat_len, U_gates, U_gener, Thets) 
    c_notting_hamil(inp_array.numpy())
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

@qml.qnode(dev)
def imagin_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, inp_array):
    i, j = circ_creator(int1, int2, U_gates, U_gener, Thets)
    up_to_un_circ(int2, i, j, mat_len, U_gates, U_gener, Thets) 
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
#Inp = ['Z1', 'X2', 'Y3', 'X1']
#real_circ_h(1, 5, 5, U_gates, U_gener, Thets, Inp)
#print(real_circ_h.draw())
#print(total_ham_element(1, 5, 5, U_gates, U_gener, Thets, H_gates, H_coeffs))

#real_circ_h(7, 12, 13, U_gates, U_gener, Thets, H_gates[1])
#print(real_circ_h.draw())



#H circuit
#print("The H matrix: ")
#print(H_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length-1, H_gates, H_coeffs))
#print("The S matrix")
#print(S_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length-1))

print(np.size(U_gates))
print(Thets)