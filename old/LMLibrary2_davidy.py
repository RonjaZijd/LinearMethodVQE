import pennylane as qml
from pennylane import numpy as np
from scipy import linalg as ln
import scipy as sp
import itertools as it

I_mat = [[1,0], [0,1]]
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0], [0,-1]])
shapy = (4,3)


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


entangler = lambda wires, plus_aux: qml.CNOT(wires=np.array(wires)+plus_aux)

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

def c_notting_hamil(input_array): ##doesn't change when extending it to multiple wires.
    i=0
    while i<len(input_array):
        as_characters = list(input_array[i])
        gate = as_characters[0]
        wire = int(as_characters[1])
        generator_creator(gate, wire)
        i=i+1
        

def circuit(params, wires):
    print("We are trying to enter this:")
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
    for i in wires:
        qml.Rot(params[i][2], params[i][1], params[i][0], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

###########################################      Devices      ###########################################################

num_wires = 8
dev = qml.device('default.qubit', wires=num_wires+1, shots=None)
dev2 = qml.device('default.qubit', wires=num_wires+1, shots=None)
dev3 = qml.device('default.qubit', wires=num_wires+1, shots=None)

_aux_op = np.kron(X-1j*Y, np.eye(2**num_wires))
def S_newy(int1, int2, U_gates, Thets):
    @qml.qnode(dev3)
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
        H += coeff * np.vdot(_state, _aux_op @ _state)

    return H

#############################    Matrix calculations    #################################################################


def H_Matrix_final_calc(U_gates, Thets, Hamil_array, Hamil_coeffs, entangle_gates):
    n = Thets.size
    H_matrix = np.zeros((n,n), dtype=complex)
    for i, j in it.combinations_with_replacement(range(n), 2):
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

def H_tilde_matrix(H, E, grad, k):
    n = grad.size
    H_tilde = np.zeros((n+1,n+1), dtype=complex)
    H_tilde[0,0] = E
    H_tilde[0,1:] = H_tilde[1:,0] = 0.5 * grad.reshape(n)
    H_tilde[1:, 1:] = H + np.eye(n) * k
    #print("H device excs: ", dev.num_executions)
    return H_tilde

def S_tilde_matrix(S, k):
    n = len(S)
    S_tilde = np.zeros((n+1, n+1), dtype=complex)
    S_tilde[0, 0] = 1.0
    S_tilde[1:, 1:] = S + np.eye(n) * k
    devies = dev3.num_executions
    #print(devies)
    return S_tilde

######################################          Functions for the optimization         ##################################

def my_gen_solve(matrixA, matrixB, size):
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

def gen_eigh(A, B):
    """Solve the generalized eigenvalue problem Av = lambda Bv."""

    # Some tests that there is no wrong inputs
    #  - Test for Hermiticity of input matrices
    assert np.allclose(B, B.T.conj()) and np.allclose(A, A.T.conj())
    #  - Test for positivity of B
    assert np.min(ln.eigvalsh(B))>0.0
    print("Using this one!!")
    # Solve eigenvalue problem for B
    Lambda_B, Phi_B = ln.eig(B)
    # Define the modified eigenvectors of B (@ is np.matmul)
    Phi_B_tilde = Phi_B @ np.diag(Lambda_B**(-1/2))
    # Define transformed A matrix
    A_tilde = Phi_B_tilde.T @ A @ Phi_B_tilde
    # Solve eigenvalue problem for transformed A
    Lambda_A, Phi_A = ln.eig(A_tilde)
    # The eigenvalues of transformed A are the generalized eigenvalues
    Lambda = Lambda_A
    # The backtransformed eigenvectors of the transformed A are the gen. eigenvectors
    Phi = Phi_B_tilde @ Phi_A
    # Bonus: Normalize the columns (i.e. the eigenvectors) to 1
    Phi /= ln.norm(Phi, 2, axis=0)

    return Lambda, Phi

def smallest_real_w_norm_optimiz_eig(H_til, S_til):
    eigvals, eigvecs = gen_eigh(H_til, S_til)
    #print("Eigenvalues using own solver: ", eigvals)
    eigvec_wanted = eigvecs[np.argmin(np.real(eigvals))]
    eigvec_wanted_normed = eigvec_wanted / eigvec_wanted[0]
    return eigvec_wanted_normed

def smallest_real_w_norm_optimiz(H_til, S_til):
    eigvals, eigvecs = sp.linalg.eig(H_til, S_til)
    #print("Eigenvalues using SciPy: ", eigvals)
    #eigvals, eigvecs = my_gen_solve(H_til, S_til, len(H_til))
    eigvec_wanted = eigvecs[np.argmin(np.real(eigvals))]
    eigvec_wanted_normed = eigvec_wanted / eigvec_wanted[0]
    return eigvec_wanted_normed

def new_thetsy(eigvec, Thets):
    #need to update Thets but without the second lement
    eigvec = np.delete(eigvec, 0) #taking off the first element: 
    thetsy = np.reshape(np.real(eigvec), shapy)

    scaling_factor = 1
    thets = Thets + scaling_factor*thetsy
    return thets

def shake_of_thets(Thets):
    shake = np.random.uniform(-1,1,shapy)
    return Thets+(shake/100)

def big_shake(Thets):
    shake = np.random.uniform(-1,1,shapy)
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

def different_regularization(Array, tolerance):

    #Input array is the array that contains the energies of the different regularizations
    #Output is the index which I want to choose
    #If there is a big energy difference, choose the one with the lowest energy
    #If there is a small energy difference <0.001 between the lowest and some others
    #identify which others also have this energy and then choose the one with 
    #the lowest regularization 
    lowest_energies = []

    sorted_array = np.argsort(Array)
    #print(sorted_array)
    for i in range(len(sorted_array)):
        #print("Comparing these numbers for ", i)
        #print(np.abs(Array[sorted_array[i]]-Array[sorted_array[0]]))
        if np.abs(Array[sorted_array[i]]-Array[sorted_array[0]])<tolerance:
            lowest_energies = np.append(lowest_energies, sorted_array[i])
    #print(lowest_energies)
    return int(np.amax(lowest_energies))

def finding_start_of_tail(array, k_array, tol):
    compare_val = array[-1]
    print("The compare value is: ", compare_val)
    for i in range(len(k_array)):
        k_max = k_array[-i]
        #print("The difference with k", k_array[i])
        if np.abs(compare_val-array[-1-i])>tol:
            if i>5:
                print("Tail ends at: ", k_array[-i+4])
                k_max = k_array[-i+4] #so that it doesn't completely cut off the tail
                break
            else:
                print("Tail ends at: ", k_max)
    return k_max

def is_sparse(array):
    n_of_elements = array.size
    tol = 0.000000001
    zero_count  = 0
    for i in range(len(array)):
        for j in range(len(array[i])):
            if np.abs(array[i][j])<tol: 
                zero_count = zero_count+1
    return zero_count>(n_of_elements/2)
