"""
name: Most recent version LM and VQE algorithm
last edited: 20/07

latest changes: 
    -changed the gates to [RZ RY RZ] so as to capture the correct rotation
    -made the shaking sensitive to the global shaking
    -try to not implement gates when not needed
    -putting all my functions into a library

"""
import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import LMLibrary as LM
#import LMandVQE21July as old
import copy as cp
import cmath

print("Running this fileee!!!!")

####Needed for the code:
Identity_mat = [[1,0], [0,1]] #haven't gotten this working inside the Hamiltonian yet.  
np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

#######################    input information        ##############################################################
U_gates = np.array([['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ']]) ##the U gates in order
no_of_gates = len(U_gates)
Thets = np.random.normal(0, np.pi, (4,3))
Thets_start = cp.copy(Thets)
#U_gener = np.array([['CZ', 'CY', 'CZ'], ['CZ', 'CY', 'CZ'], ['CZ', 'CY', 'CZ'], ['CZ', 'CY', 'CZ']])
entangle_gates = np.array([[2,3], [2,0], [3,1]]) ###the entangled gates at the end

H_VQE_coeffs = [-0.24274280513140462,-0.24274280513140462,-0.04207897647782276,0.1777128746513994,0.17771287465139946,0.12293305056183798,0.12293305056183798,0.1676831945771896,0.1676831945771896,0.17059738328801052,0.17627640804319591,-0.04475014401535161,-0.04475014401535161,0.04475014401535161,0.04475014401535161]
H_VQE_gates = [['Z3'],['Z4'],['I1'],['Z2'],['Z1'],['Z1', 'Z3'],['Z2','Z4'],['Z1','Z4'],['Z2','Z3'],['Z1','Z2'],['Z3', 'Z4'], ['Y1', 'Y2', 'X3', 'X4'],['X1', 'X2', 'Y3', 'Y4'], ['Y1', 'X2', 'X3', 'Y4'] ]

I_mat = [[1,0], [0,1]]

Hamilt_written_outt = -0.24274280513140462*qml.PauliZ(wires=2) + -0.24274280513140462*qml.PauliZ(wires=3) +  0.1777128746513994*qml.PauliZ(wires=1)+0.17771287465139946*qml.PauliZ(wires=0)+0.12293305056183798*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))+0.12293305056183798*(qml.PauliZ(wires=1)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=0)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=1)@qml.PauliZ(wires=2)) +0.17059738328801052*(qml.PauliZ(wires=0)@qml.PauliZ(wires=1))+0.17627640804319591*(qml.PauliZ(wires=2)@qml.PauliZ(wires=3))+-0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliY(wires=1)@qml.PauliX(wires=2)@qml.PauliX(wires=3))+-0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliX(wires=1)@qml.PauliY(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliX(wires=1)@qml.PauliX(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliY(wires=1)@qml.PauliY(wires=2)@qml.PauliX(wires=3))
Hamilt_written_out = -0.2*qml.PauliZ(wires=2) + -0.56*qml.PauliZ(wires=3) + 0.122*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))
no_of_wires =5
matrix_length=12 ##starting from 1
dev2 = qml.device('default.qubit', wires=4)

#########################   MAIN           ############################

def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
    for i in wires:
        qml.Rot(params[i][2], params[i][1], params[i][0], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])


energy_array = []
n_array = []
max_iterations = 200
energy_old =0
times_shaken = 0

def energy_calc(circuit, Hamilt_written_out, device, Thets):
    costy = qml.ExpvalCost(circuit, Hamilt_written_out, device)  ###for now we're simply going to do it with the specific case circuit
    energ = costy(Thets)
    return energ



###################################           Scipy BFGS Method        #######################################################
def cost_func_scipy(Thets):
    Thets = np.reshape(Thets, (4,3))
    cost_function_scipy = qml.ExpvalCost(circuit, Hamilt_written_outt, dev2)
    return cost_function_scipy(Thets)

Thets = np.reshape(Thets, (12,1))
res = sp.optimize.minimize(cost_func_scipy, Thets, method = 'BFGS')
iterations_scipy = res.get('nfev')
energy_scipy = res.get('fun')

#print(res)

###################################         Adam Method             #########################################################

energy_array_adam = []
n_array_adam = []
Thets=Thets_start
cost_function_adam = qml.ExpvalCost(circuit, Hamilt_written_outt, dev2)
opt = qml.AdamOptimizer(stepsize=0.1)
for n in range(300):
    Thets, Prev_energ = opt.step_and_cost(cost_function_adam, Thets)
    if n>2:
        if (np.abs(Prev_energ-energy_array_adam[-1])<0.00001) & (n>1):
            print()
            print("Reached convergence!")
            break
    energy_array_adam = np.append(energy_array_adam, Prev_energ)
    n_array_adam = np.append(n_array_adam, n)

###############################         Gradient Descent Method    ########################################################
Thets=Thets_start
energy_array_grad = []
n_array_grad = []
cost_function_grad_descent = qml.ExpvalCost(circuit, Hamilt_written_outt, dev2)
opt = qml.GradientDescentOptimizer(stepsize=0.4)
for n in range(700):
    Thets, Prev_energ = opt.step_and_cost(cost_function_grad_descent, Thets)
    if n>2:
        if (np.abs(Prev_energ-energy_array_grad[-1])<0.00001) & (n>1):
            #print()
            print("Reached convergence!")
            break
    energy_array_grad = np.append(energy_array_grad, Prev_energ)
    n_array_grad = np.append(n_array_grad, n)
    
 
##################################           Linear Method        ###########################################
n_array = []
energy_array_LM = []
Thets=Thets_start
eee=0

for n in range(3):
    H = LM.H_Matrix_final_calc(U_gates, Thets, H_VQE_gates, H_VQE_coeffs, entangle_gates)
    S = LM.S_Matrix_final_calc_newy(U_gates, Thets)
    S_tilde = LM.S_tilde_matrix(S)

    temp_thets_ar = []
    temp_energ_ar = []
    non_temp_k_ar = [100, 10, 1, 0.1]
    
    for k in non_temp_k_ar: 
        H_tilde = LM.H_tilde_matrix(H, eee, LM.E_grad(Thets, Hamilt_written_outt, circuit, dev2), k)
        update = LM.smallest_real_w_norm_optimiz(H_tilde, S_tilde)
        Thets_temp = LM.new_thetsy(update, Thets)
        Energ_temp = LM.energy_calc(circuit, Hamilt_written_outt, dev2, Thets_temp)
        temp_thets_ar = np.append(temp_thets_ar, Thets_temp)
        temp_energ_ar = np.append(temp_energ_ar, Energ_temp)

    temp_thets_ar = np.reshape(temp_thets_ar, (len(non_temp_k_ar), Thets.size))
    Thets = np.reshape(temp_thets_ar[np.argmin(temp_energ_ar)], Thets.shape) ##choose the new theta's of the lowest energy
    eee = temp_energ_ar[np.argmin(temp_energ_ar)] ###pick the lowest energy. 
    energy_array_LM = np.append(energy_array_LM, eee)
    n_array = np.append(n_array, n)

    print("---------------------------------------------------------------------------------------------------")
    print("Iteration: ", n)
    print("This is temp_energ_ar: ")
    print(temp_energ_ar)
    print(np.argmin(temp_energ_ar))
    print(eee)
    # print("These are the paramters: ")  #don't want to print the theta's for now
    # print(Thets % (2*np.pi))

    if energy_array_LM[n]<(-1.07):
            print("Terminating early wrt absolute value")
            #break
    if np.abs(energy_old-energy_array_LM[n])<0.001:
        print("Shakingg as the needed difference is: ")
        print(np.abs(energy_array_LM[n]-1.07)*0.0001)
        Thets = LM.shake_of_thets(Thets)
        times_shaken = times_shaken+1
    else: 
        times_shaken = 0
    print("Times shaken is: ", times_shaken)
    if times_shaken>3:
        Thets = LM.big_shake(Thets)
        print("Big Shake")
    energy_old = energy_array_LM[n]


#n_array2, energy_array = old.actual_optimization(Thets, circuit, Hamilt_written_outt, dev2, U_gates, H_VQE_gates, H_VQE_coeffs, entangle_gates)


print(energy_array)


fig, ax = plt.subplots(2,2)
ax[0,0].plot(n_array, energy_array_LM, label='Cleaned up')
#ax[0,0].plot(n_array2, energy_array_LM, label='Old vers')
ax[0,0].legend()
ax[0,0].set_title('Linear Method')
ax[0,1].plot(n_array_adam, energy_array_adam)
ax[0,1].set_title('Adam')
ax[1,0].plot(n_array_grad, energy_array_grad)
ax[1,0].set_title('Grad')
ax[1,1].scatter(iterations_scipy, energy_scipy)
ax[1,1].set_title('Scipy')

for axi in ax.flat:
    axi.set(xlabel='Iterations', ylabel='Energy')


plt.show()

# plt.plot(n_array, energy_array, label='Linear Method')
# plt.plot(n_array, energy_array_adam, label='Adam')
# plt.plot(n_array, energy_array_grad, label='Gradient Descent')
# plt.title("Optimization Methods for VQE")
# plt.legend()
# plt.xlabel("Iterations")
# plt.ylabel("Energy")
# plt.show()

# print(res)


