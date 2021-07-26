import pennylane as qml
from pennylane import numpy as np
import scipy as sp

I_mat = [[1,0], [0,1]]

######################      gates                    #############################################################

def gate_creator(string, thet, wir): ##a function to create the gates
    if string=='RX':
        return qml.RX(thet, wires=wir)
    if string=='RY':
        return qml.RY(thet, wires=wir)
    if string=='RZ':
        return qml.RZ(thet, wires=wir)

def generator_creator(string, wir):
    if string=='RX' or string=='X':
        return qml.CNOT(wires=[0, wir])
    if string=='RY' or string=='Y':
        return qml.CY(wires=[0,wir])
    if string=='RZ' or string=='Z':
        return qml.CZ(wires=[0, wir])
    if string=='I':
        return qml.QubitUnitary(I_mat, wires=wir)

def entangler(wir1, wir2, plus_ancil):
    if plus_ancil == True:
        return qml.CNOT(wires=[wir1+1, wir2+1])
    else: 
        return qml.CNOT(wires=[wir1, wir2])

def numb_to_wire(num, U_gates): #works!
    wire = 0
    local_pos = 0
    iter = 0
    loop_breaker = False
    for i in range(len(U_gates)):
        for j in range(len(U_gates[i])):
            if iter==num:
                local_pos = j
                loop_breaker =True
                break; #break out of both loops
            else: 
                iter = iter+1
        if (loop_breaker):
            break
        wire = wire+1
    
    return wire, local_pos

#######################      subcircuits        ###################################################################

def special_circ_creator(int1, int2, U_gates, Thets):
    qml.Hadamard(wires=0)
    ##under the assumption that num1 is smaller than num2, which we can design it to be, as they're symmetric matrices. 
    wire1, pos1 = numb_to_wire(int1, U_gates)
    wire2, pos2 = numb_to_wire(int2, U_gates)
    if wire1==wire2:
       # print("We're gonna do some same wire stuff: ")
        n=0
        while n<len(U_gates[wire1]):
            if n==pos1:
                break
            else:
                gate_creator(U_gates[wire1][n], Thets[wire1][n], 1)
            n=n+1
        qml.PauliX(wires=0)
        generator_creator(U_gates[wire1][pos1], 1)                          
        qml.PauliX(wires=0)
        while n<len(U_gates[wire1]):
            if n==pos2:
                break
            else:
                gate_creator(U_gates[wire1][n], Thets[wire1][n], 1)
            n=n+1
        generator_creator(U_gates[wire1][pos2], 1)

    else: 
        i=0
        while i<len(U_gates[wire1]):
            if i==pos1:
                break
            else:
               gate_creator(U_gates[wire1][i], Thets[wire1][i], 1)
               i=i+1
         
        qml.PauliX(wires=0)
        generator_creator(U_gates[wire1][pos1], 1)                          
        qml.PauliX(wires=0)

        while i<len(U_gates[wire1]):
            gate_creator(U_gates[wire1][i], Thets[wire1][i], 1)
            i=i+1

        for j in range(len(U_gates[wire2])):
            if j==pos2:
                break
            else:
                gate_creator(U_gates[wire2][j], Thets[wire2][j], 2)

        generator_creator(U_gates[wire2][pos2],2)

def circ_creator(int1, int2, U_gates, Thets): ##clean up later by putting it into one big numpy array
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
    generator_creator(U_gates[i][j], i+1)                          
    qml.PauliX(wires=0)

    while i<len(U_gates) and numbers_had!=int2:
        while j<len(U_gates[i]) and numbers_had!=int2:
            gate_creator(U_gates[i][j], Thets[i][j], i+1)
            j=j+1
            numbers_had=numbers_had+1
        if j==len(U_gates[i]):
            j=0
            i=i+1
    generator_creator(U_gates[i][j], i+1)

    return i,j ##returns the i and j element where it was left off.

def up_to_un_circ(int2, i, j, int_max, U_gates, Thets):  ##I and J is where we left off, if we put that in as 0 and 0, it will just go through all the gates
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
        generator_creator(gate, wire)
        i=i+1

def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
    for i in wires:
        qml.Rot(params[i][2], params[i][1], params[i][0], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

###########################################      Devices      ###########################################################

dev = qml.device('default.qubit', wires=5)
dev2 = qml.device('default.qubit', wires=4)
dev3 = qml.device('default.qubit', wires=3)

@qml.qnode(dev3)
def real_circ_S_newy(int1, int2, U_gates, Thets):
    special_circ_creator(int1, int2, U_gates, Thets)
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

@qml.qnode(dev3)
def imagin_circ_S_newy(int1, int2, U_gates, Thets):
    special_circ_creator(int1, int2, U_gates, Thets)
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliY(wires=0))

@qml.qnode(dev)
def real_circ_S(int1, int2, U_gates, Thets):
    circ_creator(int1, int2, U_gates, Thets)
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

@qml.qnode(dev)
def imagin_circ_S(int1, int2, U_gates, Thets): ##to get imaginary part: same circuit but then measure PauliY at the end
    circ_creator(int1, int2, U_gates, Thets)
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliY(wires=0))

@qml.qnode(dev)
def real_circ_h(int1, int2, U_gates, Thets, inp_array, entangle_gates):
    mat_len = Thets.size
    i, j = circ_creator(int1, int2, U_gates, Thets)
    up_to_un_circ(int2, i, j, mat_len, U_gates, Thets) 
    final_entangled_gates_circ(entangle_gates)
    c_notting_hamil(inp_array.numpy())
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

@qml.qnode(dev)
def imagin_circ_h(int1, int2, U_gates, Thets, inp_array, entangle_gates):
    mat_len = Thets.size
    i, j = circ_creator(int1, int2, U_gates, Thets)
    up_to_un_circ(int2, i, j, mat_len, U_gates, Thets) 
    final_entangled_gates_circ(entangle_gates)
    c_notting_hamil(inp_array.numpy())
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliY(wires=0))

####################   Needed for Matrix calculations  #####################################################################

def total_ham_element(int1, int2, U_gates, Thets, hamiltonian_array, Hamil_coefs, entangle_gates):
    Ham = 0
    HamC = 0
    i=0
    while i<len(hamiltonian_array):
        small_h_real = real_circ_h(int1, int2, U_gates, Thets, hamiltonian_array[i], entangle_gates)
        small_h_imag = imagin_circ_h(int1, int2, U_gates, Thets, hamiltonian_array[i], entangle_gates)
        small_h = small_h_real + small_h_imag*1j
        #small_h = small_h_real ##keeping it all real for the moment
        small_conj_h = small_h_real - small_h_imag*1j
        Ham = Ham + Hamil_coefs[i]*small_h  
        HamC = HamC + Hamil_coefs[i]*small_conj_h   
        i=i+1

    return Ham, HamC

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

#############################    Matrix calculations    #################################################################


def H_Matrix_final_calc(U_gates, Thets, Hamil_array, Hamil_coeffs, entangle_gates):
    matrix_length = Thets.size
    H_matrix = np.zeros(shape=(matrix_length, matrix_length), dtype=np.complex128)
   #print(np.shape(H_matrix))
    i=0
    while i<matrix_length:
        j=0
        while j<matrix_length:
            if j>i or j==i:
                H, Hc = total_ham_element(i, j, U_gates, Thets, Hamil_array, Hamil_coeffs, entangle_gates)
                H_matrix[i][j] = (1/4)*H
                H_matrix[j][i] = (1/4)*Hc  ##and the complex conjugate
            j=j+1
        i=i+1
    
    return H_matrix

def S_Matrix_final_calc(U_gates, Thets):
    matrix_length = Thets.size
    S_matrix = np.zeros(shape=(matrix_length,matrix_length), dtype=np.complex128)
    i=0
    while i<matrix_length:
        n=0
        while n<matrix_length:
            if n>i or n==i:
                real_part = real_circ_S(i,n, U_gates, Thets)
                #print(real_circ_S.draw())
                imaginary_part = imagin_circ_S(i,n,U_gates, Thets)

                ###putting it into an S matrix: 
                S_matrix[i][n] = (1/4)*(real_part+imaginary_part*1j)
                S_matrix[n][i] = (1/4)*(real_part-imaginary_part*1j) ###conjugate and imaginary!
            n=n+1
        i=i+1

    return S_matrix

def S_Matrix_final_calc_newy(U_gates, Thets):
    matrix_length = Thets.size
    S_matrix = np.zeros(shape=(matrix_length,matrix_length), dtype=np.complex128)
    i=0
    while i<matrix_length:
        n=0
        while n<matrix_length:
            if n>i or n==i:
                real_part = real_circ_S_newy(i,n, U_gates, Thets)
                #print(real_circ_S.draw())
                imaginary_part = imagin_circ_S_newy(i,n,U_gates, Thets)
                ###putting it into an S matrix: 
                S_matrix[i][n] = (1/4)*(real_part+imaginary_part*1j)
                S_matrix[n][i] = (1/4)*(real_part-imaginary_part*1j) ###conjugate and imaginary!
            n=n+1
        i=i+1

    return S_matrix

def S_alternative_way(U_gates, Thets):
    matrix_length = Thets.size
    S_matrix = np.zeros(shape=(matrix_length, matrix_length), dtype=np.complex128)
    for i in range(matrix_length):
        for n in range(matrix_length):
            if n==i:
                S_matrix[i][n] = 1/4 #setting diagonal elements to 1
            if n>i: 
                real_part = real_circ_S(i, n, U_gates, Thets)
                S_matrix[i][n] = (1/2)*real_part
                S_matrix[n][i] = (1/2)*real_part
    return S_matrix

def H_alternative_way(U_gates, Thets, Hamil_array, Hamil_coeffs, entangle_gates):
    matrix_length = Thets.size
    H_matrix = np.zeros(shape=(matrix_length, matrix_length), dtype=np.complex128)
    for i in range(matrix_length):
        for n in range(matrix_length):
            if n==i:
                H, Hc = total_ham_element(i, n, U_gates, Thets, Hamil_array, Hamil_coeffs, entangle_gates)
                H_matrix[i][n] = H
            if n>i:
                H, Hc = total_ham_element(i, n, U_gates, Thets, Hamil_array, Hamil_coeffs, entangle_gates) 
                H_matrix[i][n] = 2*H
                H_matrix[n][i] = 2*H #twice the real part
    return H_matrix

def H_tilde_matrix(H_matrix, E_0, E_grad, k):  ##E_0 can be calculated using E_calc
    mat_len = len(H_matrix)+1 #the len thing might not work the way which I want it to work 
    #k is chosen by solving it for 3 different values and then choosing the best one. 
    H_tilde_matrix = np.empty(shape=(mat_len, mat_len), dtype=np.complex128)
    H_tilde_matrix[0][0] = E_0
    #print("This is E0: ", E_0)
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

######################################          Functions for the optimization         ##################################

def smallest_real_w_norm_optimiz(S_til, H_til):
    eigvals, eigvecs = sp.linalg.eig(H_til, S_til)
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


##########################   Functions to check properties    ####################################################

def is_real(Array):
    tolerance = 0.0000000000001
    print(np.min(np.imag(Array)))
    return np.all(np.imag(Array)>-tolerance) & np.all(np.imag(Array)<tolerance)
  
def is_postive_definite(Matrix):
    eigvals = sp.linalg.eigvalsh(Matrix)
    print("The lowest eigenvalue is: ")
    print(eigvals[0])
    return (eigvals[0]+0.00000000000001)>0

def standard_deviation(Array, new_val, tolerance):
    std = 0
    for i in range(3):
        std = std + (Array[-2-i]-new_val)*(Array[-2-i]-new_val)
    std = np.sqrt(std/4)
    if std<tolerance:
        return True
    else:
        return False 
