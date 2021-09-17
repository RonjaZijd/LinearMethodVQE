import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cmath

####Needed for the code:
Identity_mat = [[1,0], [0,1]] #haven't gotten this working inside the Hamiltonian yet.  
np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})


#######################    input information        ##############################################################
U_gates = np.array([['ROT'], ['ROT'], ['ROT'], ['ROT']])
no_of_gates = len(U_gates)
Thets = np.random.normal(0, np.pi, (4,1,3))
U_gener = np.array([['CROT'], ['CROT'], ['CROT'], ['CROT']])
entangle_gates = np.array([[2,3], [2,0], [3,1]]) ###the entangled gates at the end

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
    if string=='ROT':
        print(thet)
        return qml.Rot(thet[0], thet[1], thet[2], wires=wir)
    if string== 'CROT':
        print("Doing a controlled C gate")
        return qml.CRot(*thet, wires=[0,wir])
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
print(Thets)
#######################      subcircuits        ###################################################################
def circ_creator(int1, int2, U_gates, U_gener, Thets): ##clean up later by putting it into one big numpy array
    i=0
    j=0 
    numbers_had=0 
    #print("test")               
    qml.Hadamard(wires=0) ##putting wire=0 into the +-state
    while i<len(U_gates) and numbers_had!=int1:
        #print("test0")
        while j<len(U_gates[i]) and numbers_had!=int1:
            print("This is what we're trying to put into the gate: ")
            print(Thets[i][j])
            gate_creator(U_gates[i][j], Thets[i][j], i+1)
            j=j+1
            numbers_had=numbers_had+1
            #print("test1")
        if j==len(U_gates[i]):
            i=i+1
            j=0 ##whenever we go to a new wire, we reset j
   # print("test2")
    qml.PauliX(wires=0)
    gate_creator(U_gener[i][j], [0,0,0], i+1)                          
    qml.PauliX(wires=0)
   # print("Test3")
    while i<len(U_gates) and numbers_had!=int2:
        while j<len(U_gates[i]) and numbers_had!=int2:
            gate_creator(U_gates[i][j], Thets[i][j], i+1)
            j=j+1
            numbers_had=numbers_had+1
           # print("Test4")
        if j==len(U_gates[i]):
            j=0
            i=i+1
    #print("Test5")
    gate_creator(U_gener[i][j], [0,0,0], i+1)
    #print("Test6")

    return i,j ##returns the i and j element where it was left off.

def up_to_un_circ(int2, i, j, int_max, U_gates, U_gener, Thets):  ##I and J is where we left off, if we put that in as 0 and 0, it will just go through all the gates
    numbers_had=int2
    #print("Test7")
    while numbers_had<int_max:
        while j<len(U_gates[i]) and numbers_had!=int_max:
            #print("Test8")
           # print("This is intmax: ")
            print(int_max)
            print(Thets[i][0])
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
    #print("This is where it goes wrong: ")
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
    for i in wires:
        #print(params[i])
        qml.Rot(*params[i][0], wires=i)
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
def real_circ_h(int1, int2, mat_len, U_gates, U_gener, Thets, inp_array, entangle_gates):
    i, j = circ_creator(int1, int2, U_gates, U_gener, Thets)
    up_to_un_circ(int2, i, j, mat_len, U_gates, U_gener, Thets) 
    final_entangled_gates_circ(entangle_gates)
    c_notting_hamil(inp_array.numpy())
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

@qml.qnode(dev)
def circuits(params):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=[0,1,2,3])   ##let's try taking this one away and see what it does
    for i in range(3):
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])
    return qml.expval(qml.PauliZ(wires=0))
#################################    Final Calculations   ##########################################

def total_ham_element(int1, int2, max_gates, U_gates, U_gener, Thets, hamiltonian_array, Hamil_coefs, entangle_gates):
    Ham = 0
    i=0
    while i<len(hamiltonian_array):
        small_h_real = real_circ_h(int1, int2, max_gates, U_gates, U_gener, Thets, hamiltonian_array[i], entangle_gates)
        small_h = small_h_real ##keeping it all real for the moment
        Ham = Ham + Hamil_coefs[i]*small_h  
        i=i+1

    return Ham #HamC

def S_alternative_way(U_gates, U_gener, Thets, matrix_lenth):
    S_matrix = np.zeros(shape=(matrix_length, matrix_length), dtype=np.complex128)
    #print("This is our matrix length: ")
    #print(matrix_length)
    for i in range(matrix_length):
        for n in range(matrix_length):
            #print("Now we're working in S on element: ")
           # print(i)
            #print(n)
            if n==i:
                S_matrix[i][n] = 1 #setting diagonal elements to 1
            if n>i: 
                #print("Trigggeerrr22")
                real_part = real_circ_S(i, n, U_gates, U_gener, Thets)
                S_matrix[i][n] = 2*real_part
                S_matrix[n][i] = 2*real_part
    return S_matrix

def H_alternative_way(Max_gates, U_gates, U_gener, Thets, matrix_length, Hamil_array, Hamil_coeffs, entangle_gates):
    H_matrix = np.zeros(shape=(matrix_length, matrix_length), dtype=np.complex128)
    for i in range(matrix_length):
        for n in range(matrix_length):
            print("This is the matrix element which we're working on: ")
            print(i)
            print(n)
            if n==i:
                H = total_ham_element(i, n, Max_gates, U_gates, U_gener, Thets, Hamil_array, Hamil_coeffs, entangle_gates)
                H_matrix[i][n] = H
            if n>i:
                H = total_ham_element(i, n, Max_gates, U_gates, U_gener, Thets, Hamil_array, Hamil_coeffs, entangle_gates) 
                H_matrix[i][n] = 2*H
                H_matrix[n][i] = 2*H #twice the real part
    return H_matrix

def E_grad(Thets, Hamiltonian, circuit, device):
    energy_func = qml.ExpvalCost(circuit, Hamiltonian, device)
    grad_func = qml.grad(energy_func)
    E_gra = grad_func(Thets)
    print("This is E_gra before reshaping: ")
    print(E_gra)
    E_gra = np.reshape(E_gra, (1, E_gra.size))
    return E_gra

def H_tilde_matrix(H_matrix, E_0, E_grad, k):  ##E_0 can be calculated using E_calc
    mat_len = len(H_matrix)+1 #the len thing might not work the way which I want it to work 
    #k is chosen by solving it for 3 different values and then choosing the best one. 
    H_tilde_matrix = np.empty(shape=(mat_len, mat_len), dtype=np.complex128)
    H_tilde_matrix[0][0] = E_0
    #print("This is E0: ", E_0)
    print("This is E_grad:")
    print(E_grad)
    for j in range(E_grad.size):
        H_tilde_matrix[0][j+1] = E_grad[0][j]
        H_tilde_matrix[j+1][0] = E_grad[0][j]
    for i in range(len(H_matrix)):
        for j in range(len(H_matrix[i])):
            if i==j: ##so only diagonal elements 
                H_tilde_matrix[i+1][j+1] = H_matrix[i][j] + k       ##only to the diagonal elements
            else:
                H_tilde_matrix[i+1][j+1] = H_matrix[i][j]             ###now actually with the regularization
    
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

def creating_S(circuit, Thets):
    grady = qml.grad(circuit)
    staty = grady(Thets)
    staty = np.reshape(staty, (staty.size)) 
    S_matrix = np.zeros((len(staty), len(staty)))
    for n in range(len(staty)):
        for m in range(len(staty)):
            if n==m: 
                S_matrix[n][m] = 1
            if n>m: 
                S_matrix[n][m] = staty[n]*staty[m]
                S_matrix[m][n] = staty[m]*staty[n]

    return S_matrix
########################                     MAIN                     #################################

#Calculating H and S for the first time so that it can be used
H = H_alternative_way(4, U_gates, U_gener, Thets, matrix_length, H_VQE_gates, H_VQE_coeffs, entangle_gates)
print(H)

S = S_alternative_way(U_gates, U_gener, Thets, matrix_length)
#S = S_alternative_way(circuits, Thets)
print(S)

H_tilde = H_tilde_matrix(H, energy_calc(circuit, Hamilt_written_outt, dev2, Thets), E_grad(Thets, Hamilt_written_outt, circuit, dev2), 0) #the first k is going to be 0
S_tilde = S_tilde_matrix(S)

 #####################                     OPTIMIZATION                  ################################

#print(E_grad(Thets, Hamilt_written_outt, circuit, dev2))
print(H_tilde)
print(S_tilde)

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
    return eigvecs[np.argmin(new_e_matrix)], np.argmin(new_e_matrix)

def smallest_real_optimiz(S_til, H_til):
    eigvals, eigvecs = sp.linalg.eig(H_til, S_til)
    return eigvecs[np.argmin(np.real(eigvals))]

def smallest_real_w_norm_optimiz(S_til, H_til):
    eigvals, eigvecs = sp.linalg.eig(H_til, S_til)
    print("These are the eigenvalues: ")
    print(eigvals)
    eigvec_wanted = eigvecs[np.argmin(np.real(eigvals))]
    eigvec_wanted_normed = eigvec_wanted / eigvec_wanted[0]
    return eigvec_wanted_normed

def new_thetsy(eigvec, Thets):
    #need to update Thets but without the second lement
    eigvec = np.delete(eigvec, 0) #taking off the first element: 
    thetsy = np.reshape(np.real(eigvec), (4,3))
    scaling_factor = 1
    thets = Thets + scaling_factor*thetsy
    return thets

def shake_of_thets(Thets):
    shake = np.random.uniform(-1,1,(4,3))
    return Thets+(shake/100)

def big_shake(Thets):
    shake = np.random.uniform(-1,1,(4,3))
    return Thets+(shake/10)


def actual_optimization(S_tilde, H_tilde, Thets, circuit, Hamilt_written_outt, dev2, U_gates, U_gener, matrix_length, H_gates, H_coeffs, entangel_gates):
    energy_array = []
    n_array = []
    energy_old =0
    shakey = []
    eigvals_args = []
    times_shaken = 0
    for n in range(50): ##100 is atm the maximum value of iterations 
        H = H_alternative_way(U_gates, U_gener, Thets, matrix_length, H_gates, H_coeffs, entangel_gates)
        S = S_alternative_way(circuits, Thets)
        S_tilde = S_tilde_matrix(S)

        temp_thets_ar = []
        temp_energ_ar = []
        non_temp_k_ar = [100, 10, 1, 0.1, 0.001]
        e_before = energy_calc(circuit, Hamilt_written_outt, dev2, Thets)
        for k in non_temp_k_ar: 
            H_tilde = H_tilde_matrix(H, e_before, E_grad(Thets, Hamilt_written_outt, circuit, dev2), k)
            update = smallest_real_w_norm_optimiz(H_tilde, S_tilde)
            Thets_temp = new_thetsy(update, Thets)
            Energ_temp = energy_calc(circuit, Hamilt_written_outt, dev2, Thets_temp)
            temp_thets_ar = np.append(temp_thets_ar, Thets_temp)
            temp_energ_ar = np.append(temp_energ_ar, Energ_temp)
        print("---------------------------------------------------------------------------------------------------")
        print("Iteration: ", n)
        print("This is temp_energ_ar: ")
        print(temp_energ_ar)
        print(np.argmin(temp_energ_ar))
        temp_thets_ar = np.reshape(temp_thets_ar, (len(non_temp_k_ar), Thets.size))
        Thets = np.reshape(temp_thets_ar[np.argmin(temp_energ_ar)], Thets.shape) ##choose the new theta's of the lowest energy
        eee = temp_energ_ar[np.argmin(temp_energ_ar)] ###pick the lowest energy. 
       # eigvals_args = np.append(eigvals_args, eigvaly)
        energy_array = np.append(energy_array, eee)
        print(eee)
        n_array = np.append(n_array, n)
        print("These are the paramters: ")  #don't want to print the theta's for now
        print(Thets % (2*np.pi))
        if energy_array[n]<(-1.07):
            print("Terminating early wrt absolute value")
            break
        #if np.abs(energy_old-energy_array[n])<((1-(np.abs(energy_array[n])/1.07))*0.01):   #trying out a new shaking condition
        if np.abs(energy_old-energy_array[n])<(np.abs(energy_array[n]-1.07)*0.0001):
            #probably a local minimum so shake up the Theta's
            print("Shakingg as the needed difference is: ")
            print(np.abs(energy_array[n]-1.07)*0.0001)
            Thets = shake_of_thets(Thets)
            times_shaken = times_shaken+1
            shakey = np.append(shakey, 1)
        else: #so when it doesn't get shaken
            times_shaken = 0
            #shakey = np.append(shakey, 0)
        print("Times shaken is: ", times_shaken)
        if times_shaken>3:
            Thets = big_shake(Thets)
            print("Big Shake")
        energy_old = energy_array[n] #energy old is now simply the previous value so it gets shaken quicker
        
 
    print(energy_array)

    #fig, (ax1, ax2) = plt.subplots(2)
    #ax1.scatter(n_array, shakey)
    #ax2.plot(n_array, energy_array)
    plt.plot(n_array, energy_array)
    plt.title("Smallest normalized real w/regularization of 100 - 0.001 w/ shaking and alt S")
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.show()

    return 0

#actual_optimization(S_tilde, H_tilde, Thets, circuit, Hamilt_written_outt, dev2, U_gates, U_gener, matrix_length, H_VQE_gates, H_VQE_coeffs, entangle_gates)


