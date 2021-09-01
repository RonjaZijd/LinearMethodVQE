import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import itertools as it

I_mat = [[1,0], [0,1]]
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0], [0,-1]])
shapy = (4,3)
num_wires = 4

######################      gates                    #############################################################

gate_dict = {k: eval(f"qml.{k}") for k in ("RX", "RY", "RZ")}
gate_creator = lambda k, theta, wire: gate_dict[str(k)](theta, wires=wire)

generator_dict = {
    "RX": lambda wire: qml.CNOT(wires=[0, wire]),
    "X": lambda wire: qml.CNOT(wires=[0, wire]),
    "RY": lambda wire: qml.CY(wires=[0, wire]),
    "Y": lambda wire: qml.CY(wires=[0, wire]),
    "RZ": lambda wire: qml.CZ(wires=[0, wire]),
    "Z": lambda wire: qml.CZ(wires=[0, wire]),
    "I": lambda wire: qml.QubitUnitary(np.eye(2), wires=wire),
}
generator_creator = lambda k, wire: generator_dict[str(k)](wire)

def c_not_hamil_matrix(term, num_wires):
    P0 = np.array([[1, 0],[0, 0]])
    P1 = np.array([[0, 0],[0, 1]])
    do_nothing = np.kron(P0, np.eye(2**(num_wires-1)))
    gates = {}
    for op in term:
        gates[int(op[1])] = eval(f"qml.Pauli{op[0]}({op[1]}).matrix")
        
    to_do = np.eye(1, dtype=complex)
    for i in range(1, num_wires):
        to_do = np.kron(to_do, gates.get(i, np.eye(2)))
    do_something = np.kron(P1, to_do)

    return do_nothing+do_something


entangler = lambda wires, plus_aux: qml.CNOT(wires=np.array(wires)+plus_aux) ##where is this used? in the main file? 

def numb_to_wire(num, U_gates):
    _num = num
    wire = 0
    for wire, _gates in enumerate(U_gates):
        if len(_gates)<=_num:
            _num -= len(_gates)
        else:
            break

    return wire, _num

name_to_index = {
    'Identity': "I",
    'PauliX': "X",
    'PauliY': "Y",
    'PauliZ': "Z",
}

def hamiltonian_into_gates(H):
    return [
        [name_to_index[op_name]+str(wire) for wire, op_name in zip(op.wires, op.name)]
        for op in H.ops
    ]
        
#######################      subcircuits        ###################################################################

def special_circ_creator(int1, int2, U_gates, Thets):
    qml.Hadamard(wires=0)
    ##under the assumption that num1 is smaller than num2, which we can design it to be, as they're symmetric matrices. 
    wire1, pos1 = numb_to_wire(int1, U_gates)
    wire2, pos2 = numb_to_wire(int2, U_gates)
    if wire1==wire2:
        for n, (gate, theta) in enumerate(zip(U_gates[wire1], Thets[wire1])):
            if n==pos1:
                qml.PauliX(wires=0)
                generator_creator(gate, 1)                          
                qml.PauliX(wires=0)
            if n==pos2:
                generator_creator(gate, 1)
                break
            gate_creator(gate, theta, 1)
    else: 
        for i, (wire, pos) in enumerate(zip([wire1, wire2], [pos1, pos2])):
            for n, (gate, theta) in enumerate(zip(U_gates[wire], Thets[wire])):
                if n==pos:
                    if i==0:
                        qml.PauliX(wires=0)
                    generator_creator(gate, 1+i)
                    if i==0:
                        qml.PauliX(wires=0)
                    else:
                        break
                gate_creator(gate, theta, 1+i)

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
        
###########################################      Devices      ###########################################################
dev = qml.device('default.qubit', wires=num_wires+1)
#dev2 = qml.device('default.qubit', wires=num_wires+1) ##don't completely see why we need multiple different devices here?
#dev3 = qml.device('default.qubit', wires=num_wires+1)

_aux_op = np.kron(X-1j*Y, np.eye(2**num_wires))

def S_newy(int1, int2, U_gates, Thets):
    @qml.qnode(dev)
    def circ_S_newy(int1, int2, U_gates, Thets):
        special_circ_creator(int1, int2, U_gates, Thets)
        return qml.state()
    state = circ_S_newy(int1, int2, U_gates, Thets)
    S = np.vdot(state, _aux_op @ state)
    return S

def total_ham_element(int1, int2, U_gates, Thets, inp_arrays, Hamil_coeffs, entangle_wires):
    @qml.qnode(dev)
    def circ_h(int1, int2, U_gates, Thets, entangle_wires):
        mat_len = Thets.size
        i, j = circ_creator(int1, int2, U_gates, Thets)
        up_to_un_circ(int2, i, j, mat_len, U_gates, Thets) 
        [qml.CNOT(wires=[ent_wires[0]+1, ent_wires[1]+1]) for ent_wires in entangle_wires] 
        return qml.state()

    state = circ_h(int1, int2, U_gates, Thets, entangle_wires)
    H = 0.0+1j*0.0
    for term, coeff in zip(inp_arrays, Hamil_coeffs):
        _state = c_not_hamil_matrix(term, num_wires+1) @ state 
        H += coeff * np.vdot(_state, _aux_op @ _state) #

    return H

#############################    Matrix calculations    #################################################################


def H_Matrix_final_calc(U_gates, Thets, Hamil_array, Hamil_coeffs, entangle_gates):
    n = Thets.size
    H_matrix = np.zeros((n,n), dtype=complex)
    for i, j in it.combinations_with_replacement(range(n), 2): #it goes over all combinations (which is what we want)
        #the combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
        H = total_ham_element(i, j, U_gates, Thets, Hamil_array, Hamil_coeffs, entangle_gates)
        H_matrix[i,j] = 0.25*H
        H_matrix[j,i] = 0.25*np.conj(H)
    
    return H_matrix

def S_Matrix_final_calc_newy(U_gates, Thets):
    n = Thets.size
    S_matrix = np.zeros((n, n), dtype=complex)
    for i, j in it.combinations_with_replacement(range(n), 2):
        S = S_newy(i, j, U_gates, Thets)
        S_matrix[i,j] = 0.25*S
        S_matrix[j,i] = 0.25*np.conj(S)

    return S_matrix

def H_tilde_matrix(H, E, grad, k): ##understand thteesee
    n = grad.size
    H_tilde = np.zeros((n+1,n+1), dtype=complex)
    H_tilde[0,0] = E
    H_tilde[0,1:] = H_tilde[1:,0] = 0.5 * grad.reshape(n)
    H_tilde[1:, 1:] = H + np.eye(n) * k
    return H_tilde

def S_tilde_matrix(S, k):
    n = len(S)
    S_tilde = np.zeros((n+1, n+1), dtype=complex)
    S_tilde[0, 0] = 1.0
    S_tilde[1:, 1:] = S + np.eye(n) * k
    return S_tilde

######################################          Functions for the optimization         ##################################

def my_gen_solve(matrixA, matrixB, size):  ##nothing really got changed below this line. 
    eigvalsB_mat = np.zeros((size, size), dtype=np.complex128)
    eigvalsB_mat_special = np.zeros((size, size), dtype = np.complex128)

    eigvals_B, eigvecs_B = np.linalg.eig(matrixB)
    eigvecs_B = np.matrix.transpose(eigvecs_B) #to have the columns be the eigenvectors

    for i in range(size):
        for j in range(size):
            if i==j: #only along the diagonal
                eigvalsB_mat[i][j] = eigvals_B[i]
                if eigvals_B[i] == 0: 
                    eigvalsB_mat_special[i][j] = 0
                else:  
                    eigvalsB_mat_special[i][j] = 1/(np.sqrt(eigvals_B[i]))

    eigvecB_tilde = np.matmul(eigvecs_B, eigvalsB_mat_special) #we don't have to transpose B_tilde, because it's already made with the correct things
    A1 = np.matmul(matrixA, eigvecB_tilde)
    matrixA_tilde = np.matmul((np.matrix.transpose(eigvecB_tilde)), A1) #seems to work
    eigvalsA, eigvecsA = np.linalg.eig(matrixA_tilde)
    eigvecsA = np.matrix.transpose(eigvecsA)
    final_eigvec = np.matmul(eigvecB_tilde, eigvecsA)

    return eigvalsA, final_eigvec

def smallest_real_w_norm_optimiz_eig(H_til, S_til):  ##keep the option in, in case I want to go back to this later. 
    eigvals, eigvecs = my_gen_solve(H_til, S_til, len(H_til))
    eigvec_wanted = eigvecs[np.argmin(np.real(eigvals))]
    eigvec_wanted_normed = eigvec_wanted / eigvec_wanted[0]
    return eigvec_wanted_normed

def smallest_real_w_norm_optimiz(H_til, S_til):
    eigvals, eigvecs = sp.linalg.eig(H_til, S_til)
    #eigvals, eigvecs = my_gen_solve(H_til, S_til, len(H_til))
    eigvec_wanted = eigvecs[np.argmin(np.real(eigvals))]
    eigvec_wanted_normed = eigvec_wanted / eigvec_wanted[0]
    return eigvec_wanted_normed

def new_thetsy(eigvec, Thets):
    #need to update Thets but without the second lement
    eigvec = np.delete(eigvec, 0) #taking off the first element: 
    thetsy = np.reshape(np.real(eigvec), shapy)
    thets = Thets + thetsy
    return thets
