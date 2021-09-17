import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import LMLibrary2 as LM  ##contains all self-made functions needed for Linear method optimization
import copy as cp
import time
import pandas as pd
from scipy.interpolate import interp1d

np.set_printoptions(suppress=True, precision=5, formatter={'float_kind':'{:0.4f}'.format})

###################################################################################################

U_gates = np.array([['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ']]) ##the U gates in order
entangle_gates = np.array([[2,3], [2,0], [3,1]]) ###the entangled gates at the end
Thets = np.random.normal(0, np.pi, (4,3))
Thets_start = cp.copy(Thets)

H_VQE_coeffs = [-0.24274280513140462,-0.24274280513140462,-0.04207897647782276,0.1777128746513994,0.17771287465139946,0.12293305056183798,0.12293305056183798,0.1676831945771896,0.1676831945771896,0.17059738328801052,0.17627640804319591,-0.04475014401535161,-0.04475014401535161,0.04475014401535161,0.04475014401535161]
H_VQE_gates = [['Z3'],['Z4'],['I1'],['Z2'],['Z1'],['Z1', 'Z3'],['Z2','Z4'],['Z1','Z4'],['Z2','Z3'],['Z1','Z2'],['Z3', 'Z4'], ['Y1', 'Y2', 'X3', 'X4'],['X1', 'X2', 'Y3', 'Y4'], ['Y1', 'X2', 'X3', 'Y4'] ]
Hamilt_written_outt = -0.24274280513140462*qml.PauliZ(wires=2) + -0.24274280513140462*qml.PauliZ(wires=3) +  0.1777128746513994*qml.PauliZ(wires=1)+0.17771287465139946*qml.PauliZ(wires=0)+0.12293305056183798*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))+0.12293305056183798*(qml.PauliZ(wires=1)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=0)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=1)@qml.PauliZ(wires=2)) +0.17059738328801052*(qml.PauliZ(wires=0)@qml.PauliZ(wires=1))+0.17627640804319591*(qml.PauliZ(wires=2)@qml.PauliZ(wires=3))+-0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliY(wires=1)@qml.PauliX(wires=2)@qml.PauliX(wires=3))+-0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliX(wires=1)@qml.PauliY(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliX(wires=1)@qml.PauliX(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliY(wires=1)@qml.PauliY(wires=2)@qml.PauliX(wires=3))
Hamilt_written_out = -0.2*qml.PauliZ(wires=2) + -0.56*qml.PauliZ(wires=3) + 0.122*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))

#Other information
no_of_gates = len(U_gates)
no_of_wires =5
matrix_length=12 ##starting from 1

dev_lm = qml.device('default.qubit', wires=4)
dev2 = qml.device('default.qubit', wires=4)

######################################################################################################

def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
    for i in wires:
        qml.Rot(params[i][2], params[i][1], params[i][0], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

def energy_calc(circuit, Hamilt_written_out, device, Thets):
    costy = qml.ExpvalCost(circuit, Hamilt_written_out, device)  ###for now we're simply going to do it with the specific case circuit
    energ = costy(Thets)
    return energ

########################################################################################################

n_array = []
energy_array_LM = []
#Thets_start = np.array([[5.4994, 2.7966, 5.0794], [2.3249, 1.6795, 0.7422], [3.9375, 1.2165, 3.7741], [3.7267, 3.8355, 0.0653]])
Thets=Thets_start
Thets_hist = np.reshape(Thets, Thets.size)

All_Thets = np.column_stack((Thets_hist, Thets_hist))


print(All_Thets)

eee=0
Regularization = 0
max_k =1
print("I've started running!")

for n in range(15) :
    H = LM.H_Matrix_final_calc(U_gates, Thets, H_VQE_gates, H_VQE_coeffs, entangle_gates)
    S = LM.S_Matrix_final_calc_newy(U_gates, Thets)
    S_tilde = LM.S_tilde_matrix(S, Regularization)

    temp_thets_ar = []
    temp_energ_ar = []
    #non_temp_k_ar = [0.404476, 0.404477]
    non_temp_k_ar = np.linspace(0, max_k, 100, endpoint=True)
    
    for k in non_temp_k_ar: 
        H_tilde = LM.H_tilde_matrix(H, eee, LM.E_grad(Thets, Hamilt_written_outt, circuit, dev_lm), k)
        update = LM.smallest_real_w_norm_optimiz(H_tilde, S_tilde)
        Thets_temp = LM.new_thetsy(update, Thets)
        Energ_temp = LM.energy_calc(circuit, Hamilt_written_outt, dev_lm, Thets_temp)
        temp_thets_ar = np.append(temp_thets_ar, Thets_temp)
        temp_energ_ar = np.append(temp_energ_ar, Energ_temp)

    full_temp_energ_ar = []
    temp_thets_ar = np.reshape(temp_thets_ar, (len(non_temp_k_ar), Thets.size))
    # #arg_chosen = LM.different_regularization(temp_energ_ar, 0.000001)
    arg_chosen = np.argmin(temp_energ_ar)
    #arg_chosen=-27  ###aka choose k is equal to 1
    step_thets = np.reshape(temp_thets_ar[arg_chosen], Thets.shape)-Thets
    Thets = np.reshape(temp_thets_ar[arg_chosen], Thets.shape) ##choose the new theta's of the lowest energy
    Thets_hist = np.reshape(Thets, Thets.size)
    All_Thets = np.column_stack((All_Thets, Thets_hist))
    eee = temp_energ_ar[arg_chosen] ###pick the lowest energy. 

    max_k = LM.finding_start_of_tail(temp_energ_ar, non_temp_k_ar, 0.02)
    energy_array_LM = np.append(energy_array_LM, eee)
    n_array = np.append(n_array, n)
    
    #################################  All the printing and plotting  ###########################################
    print("---------------------------------------------------------------------------------------------------") #printing things to see what the program is doing
    print("Iteration: ", n)
    print("This is temp_energ_ar: ")
    print(temp_energ_ar)
    print(np.argmin(temp_energ_ar))
    #print("K chosen using interpolation", k_chosen)
    print("Actual k chosen", non_temp_k_ar[arg_chosen])
    print("Energy chosen: ", eee)
    #print("Energy w/ interpolation: ", e_e)
    #plt.plot(non_temp_k_ar, temp_energ_ar, 'o', knew, f(knew), '-', knew, f2(knew), '--', knew, f3(knew), '-.')
    #plt.legend(['data', 'linear', 'cubic', 'quadratic'], loc='best')
    #plt.scatter(non_temp_k_ar, temp_energ_ar)
    #plt.xlabel('k-value')
    #plt.ylabel('Energy')
    #plt.title('K-cutoff point:', max_k)
    #plt.plot(non_temp_k_ar, temp_energ_ar)
    #plt.show()
    print("These are the paramters: ")  #don't want to print the theta's for now
    print(Thets % (2*np.pi))
    print("This is the step which has just been taken: ")
    print(step_thets)

    if energy_array_LM[n]<(-1.095):
            print("Terminating early wrt absolute value")
            break

print(All_Thets[0].size)
n_array = np.insert(n_array, 0, -1)
n_array = np.insert(n_array, 0, -1)
print(n_array.size)
print(n_array)
x_array=np.array([0,1,2])
All_Thets = All_Thets % (2*np.pi)
for i in range(Thets.size):
    plt.plot(n_array, All_Thets[i], label=f'Theta {i}')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Theta value')
plt.show()
