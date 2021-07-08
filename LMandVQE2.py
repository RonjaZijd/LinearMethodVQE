import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cmath

####Needed for the code:
Identity_mat = [[1,0], [0,1]] #haven't gotten this working inside the Hamiltonian yet.  
#np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})


#######################    input information        ##############################################################
U_gates = np.array([['RX', 'RY', 'RZ'], ['RX', 'RY', 'RZ'], ['RX', 'RY', 'RZ'], ['RX', 'RY', 'RZ']]) ##the U gates in order
no_of_gates = len(U_gates)
Thets = np.random.normal(0, np.pi, (4,3))
U_gener = np.array([['CNOT', 'CY', 'CZ'], ['CNOT', 'CY', 'CZ'], ['CNOT', 'CY', 'CZ'], ['CNOT', 'CY', 'CZ']])
entangle_gates = np.array([[2,3], [2,0], [3,1]]) ###the entangled gates at the end

H_coeffs = [0.45, 0.34, 0.11, 0.62] #let's keep with the small one here and change to the big actual hamiltonian later.
H_gates = [['Z2', 'X1'], ['Y2', 'X1', 'Z4'], ['Z3'], ['X2', 'Y1'] ]

H_VQE_coeffs = [-0.24274280513140462,-0.24274280513140462,-0.04207897647782276,0.1777128746513994,0.17771287465139946,0.12293305056183798,0.12293305056183798,0.1676831945771896,0.1676831945771896,0.17059738328801052,0.17627640804319591,-0.04475014401535161,-0.04475014401535161,0.04475014401535161,0.04475014401535161]
H_VQE_gates = [['Z3'],['Z4'],['I1'],['Z2'],['Z1'],['Z1', 'Z3'],['Z2','Z4'],['Z1','Z4'],['Z2','Z3'],['Z1','Z2'],['Z3', 'Z4'], ['Y1', 'Y2', 'X3', 'X4'],['X1', 'X2', 'Y3', 'Y4'], ['Y1', 'X2', 'X3', 'Y4'] ]

I_mat = [[1,0], [0,1]]

Hamilt_written_outt = -0.24274280513140462*qml.PauliZ(wires=2) + -0.24274280513140462*qml.PauliZ(wires=3) +  0.1777128746513994*qml.PauliZ(wires=1)+0.17771287465139946*qml.PauliZ(wires=0)+0.12293305056183798*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))+0.12293305056183798*(qml.PauliZ(wires=1)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=0)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=1)@qml.PauliZ(wires=2)) +0.17059738328801052*(qml.PauliZ(wires=0)@qml.PauliZ(wires=1))+0.17627640804319591*(qml.PauliZ(wires=2)@qml.PauliZ(wires=3))+-0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliY(wires=1)@qml.PauliX(wires=2)@qml.PauliX(wires=3))+-0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliX(wires=1)@qml.PauliY(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliX(wires=1)@qml.PauliX(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliY(wires=1)@qml.PauliY(wires=2)@qml.PauliX(wires=3))
Hamilt_written_out = -0.2*qml.PauliZ(wires=2) + -0.56*qml.PauliZ(wires=3) + 0.122*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))
no_of_wires =5
matrix_length=12 ##starting from 1

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
    if string=='I':
        return qml.QubitUnitary(I_mat, wires=wir)

def entangler(wir1, wir2, plus_ancil):
    if plus_ancil == True:
        return qml.CNOT(wires=[wir1+1, wir2+1])
    else: 
        return qml.CNOT(wires=[wir1, wir2])

#######################      subcircuits        ###################################################################
def circ_creator(int1, int2, U_gates, U_gener, Thets): ##clean up later by putting it into one big numpy array
    i=0
    j=0 
    numbers_had=0 
   # print("test")               
    qml.Hadamard(wires=0) ##putting wire=0 into the +-state
    while i<len(U_gates) and numbers_had!=int1:
        #print("test0")
        while j<len(U_gates[i]) and numbers_had!=int1:
            gate_creator(U_gates[i][j], Thets[i][j], i+1)
            j=j+1
            numbers_had=numbers_had+1
            #print("test1")
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

def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
    for i in wires:
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

##############################################     Actual QNodes       #############################################
dev = qml.device('default.qubit', wires=5)
dev2 = qml.device('default.qubit', wires=4)

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
def real_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, inp_array, entangle_gates):
    i, j = circ_creator(int1, int2, U_gates, U_gener, Thets)
    up_to_un_circ(int2, i, j, mat_len, U_gates, U_gener, Thets) 
    final_entangled_gates_circ(entangle_gates)
    c_notting_hamil(inp_array.numpy())
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

@qml.qnode(dev)
def imagin_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, inp_array, entangle_gates):
    i, j = circ_creator(int1, int2, U_gates, U_gener, Thets)
    up_to_un_circ(int2, i, j, mat_len, U_gates, U_gener, Thets) 
    final_entangled_gates_circ(entangle_gates)
    c_notting_hamil(inp_array.numpy())
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliY(wires=0))


#################################    Final Calculations   ##########################################

def total_ham_element(int1, int2, mat_len, U_gates, U_gener, Thets, hamiltonian_array, Hamil_coefs, entangle_gates):
    Ham = 0
    HamC = 0
    i=0
    while i<len(hamiltonian_array):
        small_h_real = real_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, hamiltonian_array[i], entangle_gates)
        small_h_imag = imagin_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, hamiltonian_array[i], entangle_gates)
        small_h = small_h_real + small_h_imag*1j
        small_conj_h = small_h_real - small_h_imag*1j
        Ham = Ham + Hamil_coefs[i]*small_h  
        HamC = HamC + Hamil_coefs[i]*small_conj_h   
        i=i+1

    return Ham, HamC

def H_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length, Hamil_array, Hamil_coeffs, entangle_gates):
    H_matrix = np.empty(shape=(matrix_length, matrix_length), dtype=np.complex128)
   #print(np.shape(H_matrix))
    i=0
    while i<matrix_length:
        j=0
        while j<matrix_length:
            if j>i or j==i:
                H, Hc = total_ham_element(i, j, matrix_length, U_gates, U_gener, Thets, Hamil_array, Hamil_coeffs, entangle_gates)
                H_matrix[i][j] = H
                H_matrix[j][i] = Hc  ##and the complex conjugate
            j=j+1
        i=i+1
    
    return H_matrix

def S_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length):
    S_matrix = np.empty(shape=(matrix_length,matrix_length), dtype=np.complex128)
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

                ####printing it out to check the values
                #print("Real: ", real_part)
                #print("Imaginary: ", imaginary_part) 
                #print()
            n=n+1
        i=i+1

    return S_matrix

def E_grad(Thets, Hamiltonian, circuit, device):
    energy_func = qml.ExpvalCost(circuit, Hamiltonian, device)
    grad_func = qml.grad(energy_func)
    E_gra = grad_func(Thets)
    E_gra = np.reshape(E_gra, (1, E_gra.size))
    return E_gra

def H_tilde_matrix(H_matrix, E_0, E_grad):  ##E_0 can be calculated using E_calc
    mat_len = len(H_matrix)+1 #the len thing might not work the way which I want it to work 

    H_tilde_matrix = np.empty(shape=(mat_len, mat_len), dtype=np.complex128)
    H_tilde_matrix[0][0] = E_0
    print("This is E0: ", E_0)
    for j in range(E_grad.size):
        H_tilde_matrix[0][j+1] = E_grad[0][j]
        H_tilde_matrix[j+1][0] = E_grad[0][j]
    for i in range(len(H_matrix)):
        for j in range(len(H_matrix[i])):
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

def energy_calc(circuit, Hamilt_written_out, device, Thets):
    costy = qml.ExpvalCost(circuit, Hamilt_written_out, device)  ###for now we're simply going to do it with the specific case circuit
    energ = costy(Thets)
    return energ


########################                     MAIN                     #################################

#Calculating H and S for the first time so that it can be used
H = H_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length, H_gates, H_coeffs, entangle_gates)
S = S_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length)

H_tilde = H_tilde_matrix(H, energy_calc(circuit, Hamilt_written_outt, dev2, Thets), E_grad(Thets, Hamilt_written_outt, circuit, dev2))
S_tilde = S_tilde_matrix(S)

 #####################                     OPTIMIZATION                  ################################

#print(H_tilde)
#print(S_tilde)

def optimiz_tils(S_til, H_til, Thets, circuit, Hamil, device):
    eigvals_new=[]
    eigvals, eigvecs = sp.linalg.eig(H_til, S_til)
    for i in range(len(eigvecs)):
        eigvals_new = np.append(eigvals_new, eigvals[i]/eigvecs[i][0])
        eigvecs[i] = eigvecs[i]/eigvecs[i][0]
    new_e_matrix = []
    for i in range(len(eigvals_new)):
        ener = energy_calc(circuit, Hamil, device, new_thetsy(eigvecs[i], Thets))
        new_e_matrix = np.append(new_e_matrix, ener)
    print("-----------------------------------------------------------------------------")
    print(eigvals_new)
    print(new_e_matrix)
    print("Chosen ", np.argmin(new_e_matrix))
    print()
    return eigvecs[np.argmin(new_e_matrix)]

def new_thetsy(eigvec, Thets):
    #need to update Thets but without the second lement
    eigvec = np.delete(eigvec, 0) #taking off the first element: 
    thetsy = np.reshape(np.real(eigvec), (4,3))
    thets = Thets + thetsy
    return thets

#needed for updating:
energy_array = []
n_array = []
energy_old =0

for n in range(100): ##100 is atm the maximum value of iterations
    updaty = optimiz_tils(S_tilde, H_tilde, Thets, circuit, Hamilt_written_outt, dev2)
    #updaty = alternate_update_eigvecs(S, H, circuit, Hamilt_written_outt, dev2, Thets)
    #updaty = update_eigvec_w_smallest_tot(S, H)
    #updaty = update_eigvec_w_smallest_real(S,H, circuit, Hamilt_written_outt, dev2, Thets)
    Thets = new_thetsy(updaty, Thets)
    H = H_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length, H_gates, H_coeffs, entangle_gates)
    S = S_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length)

    eee = energy_calc(circuit, Hamilt_written_outt, dev2, Thets)
    H_tilde = H_tilde_matrix(H, eee, E_grad(Thets, Hamilt_written_outt, circuit, dev2))
    S_tilde = S_tilde_matrix(S)

    energy_array = np.append(energy_array, eee)
    print(n)
    print(eee)
    n_array = np.append(n_array, n)
    if np.abs(energy_old-energy_array[n])<0.0000001:  ###filling in how precise we want to have it
        print("Terminating early")
        break
    energy_old = energy_array[n]
 
print(energy_array)
plt.plot(n_array, energy_array)
plt.show()


"""
final_energy_array = []
i_had_array = []

for i in range(10):
    energy_array = []
    for n in range(20):
        updaty = alternate_update_eigvecs(S, H, circuit, Hamilt_written_outt, dev2, Thets)
        Thets = new_thets(updaty, Thets)
        H = H_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length, H_gates, H_coeffs, entangle_gates)
        S = S_Matrix_final_calc(U_gates, U_gener, Thets, matrix_length)
        eee = energy_calc(circuit, Hamilt_written_outt, dev2, Thets)
        energy_array = np.append(energy_array, eee)
    final_energy_array = np.append(final_energy_array, energy_array[-1])
    i_had_array = np.append(i_had_array, i)
    print(energy_array[-1])
    print(i)

"""



"""
extra stuff I might want to come back to later: 
def act_circ(U_gates, Thets, entangle_gates, wires):
    for i in range(len(U_gates)):
        for j in range(len(U_gates[i])):
            gate_creator(U_gates[i][j], Thets[i][j], i) #just i because we don't have an ancilliary bit
    final_entangled_gates_circ(entangle_gates)

def act_circ2(params, wires):   ###fix the circuits and make it work for the general case
    #print(params[0])
    for i in wires:  #######temporarryyyy!!!!!!!!!!!!!!!!
        print(i)
        print(len(params[0]))
        for j in range(3):
            print('test')
            gate_creator(params[0][i][j], params[1][i][j], i) #just i because we don't have an ancilliary bit
    final_entangled_gates_circ(params[2])




"""