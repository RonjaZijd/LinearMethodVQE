"""
name: Most recent version LM and VQE algorithm, plus Adam, Scipy and Grad
last edited: 20/07

To do programming wise: 
    -make a function to strip the actual hamiltonian into the needed arrays
"""
import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import LMLibrary2 as LM  ##contains all self-made functions needed for Linear method optimization
import copy as cp
import time
import pandas as pd
from scipy.interpolate import interp1d

#np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

#######################    input information        ##############################################################
#np.random.seed(333)
#The variational circuit
U_gates = np.array([['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ']]) ##the U gates in order
entangle_gates = np.array([[2,3], [2,0], [3,1]]) ###the entangled gates at the end
Thets = np.random.normal(0, np.pi, (4,3))
print(Thets)
Thets_start = cp.copy(Thets)

#Hamiltonian
H_VQE_coeffs = [-0.24274280513140462,-0.24274280513140462,0.1777128746513994,0.17771287465139946,0.12293305056183798,0.12293305056183798,0.1676831945771896,0.1676831945771896,0.17059738328801052,0.17627640804319591,-0.04475014401535161,-0.04475014401535161,0.04475014401535161,0.04475014401535161]
H_VQE_gates = [['Z3'],['Z4'],['Z2'],['Z1'],['Z1', 'Z3'],['Z2','Z4'],['Z1','Z4'],['Z2','Z3'],['Z1','Z2'],['Z3', 'Z4'], ['Y1', 'Y2', 'X3', 'X4'],['X1', 'X2', 'Y3', 'Y4'], ['Y1', 'X2', 'X3', 'Y4'] ]
Hamilt_written_outt = -0.24274280513140462*qml.PauliZ(wires=2) + -0.24274280513140462*qml.PauliZ(wires=3) +  0.1777128746513994*qml.PauliZ(wires=1)+0.17771287465139946*qml.PauliZ(wires=0)+0.12293305056183798*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))+0.12293305056183798*(qml.PauliZ(wires=1)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=0)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=1)@qml.PauliZ(wires=2)) +0.17059738328801052*(qml.PauliZ(wires=0)@qml.PauliZ(wires=1))+0.17627640804319591*(qml.PauliZ(wires=2)@qml.PauliZ(wires=3))+-0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliY(wires=1)@qml.PauliX(wires=2)@qml.PauliX(wires=3))+-0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliX(wires=1)@qml.PauliY(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliX(wires=1)@qml.PauliX(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliY(wires=1)@qml.PauliY(wires=2)@qml.PauliX(wires=3))
Hamilt_written_out = -0.2*qml.PauliZ(wires=2) + -0.56*qml.PauliZ(wires=3) + 0.122*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))

#Other information
no_of_gates = len(U_gates)
no_of_wires =5
matrix_length=12 ##starting from 1

name_csv_file = "EnergyTry1.csv"

##################################            Devices          ##########################################################

dev_grad = qml.device('default.qubit', wires=4)
dev_adam = qml.device('default.qubit', wires=4)
dev_scipy = qml.device('default.qubit', wires=4)
dev_lm = qml.device('default.qubit', wires=4)
dev2 = qml.device('default.qubit', wires=4)

#########################   MAIN           ############################

def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
    for i in wires:
        qml.Rot(params[i][2], params[i][1], params[i][0], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

max_iterations = 200

def energy_calc(circuit, Hamilt_written_out, device, Thets):
    costy = qml.ExpvalCost(circuit, Hamilt_written_out, device)  ###for now we're simply going to do it with the specific case circuit
    energ = costy(Thets)
    return energ

def energy_grad(Thets):
    global circuit
    global Hamilt_written_outt
    global dev2
    Thets = np.reshape(Thets, (4,3))
    energy_func = qml.ExpvalCost(circuit, Hamilt_written_outt, dev2)
    grad_func = qml.grad(energy_func)
    E_gra = grad_func(Thets)
    #E_gra = np.reshape(E_gra, (E_gra.size, 1))
    return E_gra.flatten()

##################################           Scipy BFGS Method        #######################################################
energy_array_scip = []
n_array_scipy = []

def cost_func_scipy(Thets):
    global energy_array_scip #all global, needed to track the progress of SciPy function
    global circuit
    global Hamilt_written_outt
    global dev2
    Thets = np.reshape(Thets, (4,3))
    cost_function_scipy = qml.ExpvalCost(circuit, Hamilt_written_outt, dev_scipy)
    energy_array_scip = np.append(energy_array_scip, energy_calc(circuit, Hamilt_written_outt, dev2, Thets))
    return cost_function_scipy(Thets)

t_0_scipy = time.process_time()
Thets = np.reshape(Thets, (12,1))
res = sp.optimize.minimize(cost_func_scipy, Thets, method = 'BFGS', jac=energy_grad)
iterations_scipy = res.get('nfev')
energy_scipy = res.get('fun')
t_1_scipy = time.process_time()

for n in range(iterations_scipy):
    n_array_scipy = np.append(n_array_scipy, n)

print(res)

#################################         Adam Method             #########################################################

t_0_adam = time.process_time()
#initializing
energy_array_adam = []
n_array_adam = []
Thets=Thets_start

cost_function_adam = qml.ExpvalCost(circuit, Hamilt_written_outt, dev_adam)  #functions on which optimization is performed
opt = qml.AdamOptimizer(stepsize=0.1)

for n in range(300):
    Thets, Prev_energ = opt.step_and_cost(cost_function_adam, Thets)
    if n>2:
        if (np.abs(Prev_energ-energy_array_adam[-1])<0.000001) & (n>1):
            print("Reached convergence!")
            break
    energy_array_adam = np.append(energy_array_adam, Prev_energ)
    n_array_adam = np.append(n_array_adam, n)
t_1_adam = time.process_time()

###############################         Gradient Descent Method    ########################################################

t_0_grad = time.process_time()
#initializing
Thets=Thets_start
energy_array_grad = []
n_array_grad = []

cost_function_grad_descent = qml.ExpvalCost(circuit, Hamilt_written_outt, dev_grad)
opt = qml.GradientDescentOptimizer(stepsize=0.4)

for n in range(700):
    Thets, Prev_energ = opt.step_and_cost(cost_function_grad_descent, Thets)
    if n>2:
        if (np.abs(Prev_energ-energy_array_grad[-1])<0.000001) & (n>1):
            print("Reached convergence!")
            break
    energy_array_grad = np.append(energy_array_grad, Prev_energ)
    n_array_grad = np.append(n_array_grad, n)
t_1_grad = time.process_time()  
 
#################################           Linear Method        ###########################################

t_0_lm = time.process_time()
#Initializing
n_array = []
energy_array_LM = []
#Thets_start = np.array([[2.69, 1.39, 2.10], [3.07, 1.12, 4.79], [0.45, 2.18, 3.93], [1.24, 0.62, 6.85]])
#Thets_start = np.array([[ 6.32638897, -6.12331649, -4.49510873], [ 0.99456863,  0.12559029,  0.26026123], [-3.55112765, -6.02068374 , 0.24326378],[-7.56792862 , 6.3901332 ,  0.36616908]])
Thets=Thets_start
eee= LM.energy_calc(circuit, Hamilt_written_outt, dev_lm, Thets)
n_array = np.append(n_array, 0)
energy_array_LM = np.append(energy_array_LM, eee)
condition_array_H = []
###For the naming: 
Regularization = 0.01

print("I've started running!")
print("Very initial parameters: ")
print(Thets)
print("Starting energy associated with these parameters: ")
print(eee)
prev_parameters = Thets


for n in range(10) :
    H = LM.H_Matrix_final_calc(U_gates, Thets, H_VQE_gates, H_VQE_coeffs, entangle_gates)
    S = LM.S_Matrix_final_calc_newy(U_gates, Thets)
    S_tilde = LM.S_tilde_matrix(S, Regularization)
    temp_thets_ar = []
    temp_energ_ar = []
    condition_array_H = []
    n_of_times_not_converged = 0
    non_temp_k_ar = np.linspace(0.01,  1, 100, endpoint=True) ##disabling the tail
    
    for k in non_temp_k_ar: 
        H_tilde = LM.H_tilde_matrix(H, eee, LM.E_grad(Thets, Hamilt_written_outt, circuit, dev_lm), k)
        condition_array_H = np.append(condition_array_H, np.linalg.cond(H_tilde))
        try:
            update = LM.smallest_real_w_norm_optimiz(H_tilde, S_tilde)
            Thets_temp = LM.new_thetsy(update, Thets)
            Energ_temp = LM.energy_calc(circuit, Hamilt_written_outt, dev_lm, Thets_temp)
            temp_thets_ar = np.append(temp_thets_ar, Thets_temp)
            temp_energ_ar = np.append(temp_energ_ar, Energ_temp)
        except:
            print("Could not converge")
            n_of_times_not_converged += 1
            non_temp_k_ar = np.delete(non_temp_k_ar, np.where(non_temp_k_ar==k))
            condition_array_H = np.delete(condition_array_H, np.where(condition_array_H==np.linalg.cond(H_tilde)))
    full_temp_energ_ar = []
    temp_thets_ar = np.reshape(temp_thets_ar, (len(non_temp_k_ar), Thets.size))
    arg_chosen = np.argmin(temp_energ_ar)
    Thets = np.reshape(temp_thets_ar[arg_chosen], Thets.shape) ##choose the new theta's of the lowest energy
    eee = temp_energ_ar[arg_chosen] ###pick the lowest energy. 
    max_k = LM.finding_start_of_tail(temp_energ_ar, non_temp_k_ar, 0.001)
    energy_array_LM = np.append(energy_array_LM, eee)
    n_array = np.append(n_array, n+1)
    
    print("---------------------------------------------------------------------------------------------------") #printing things to see what the program is doing
    print("Iteration: ", n)
    print("Number of times not converged: ", n_of_times_not_converged)
    print()
    print("This is temp_energ_ar: ")
    print(temp_energ_ar)
    print(np.argmin(temp_energ_ar))
    print("Actual k chosen", non_temp_k_ar[arg_chosen])
    print("Energy chosen after update step: ", eee)


    ############plottingg########################################
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.scatter(non_temp_k_ar, temp_energ_ar)
    #plt.xlabel('k-value')
    #plt.ylabel('Energy')
    #plt.title('K-cutoff point:', max_k)
    ax1.plot(non_temp_k_ar, temp_energ_ar, label = f'Condition number S: {np.linalg.cond(S_tilde)}, not converged: {n_of_times_not_converged}')
    ax1.legend()
    ax2.set(xlabel='K-value', ylabel = 'Condition number H')
    ax1.set(ylabel='Energy')
    ax2.scatter(non_temp_k_ar, condition_array_H, label=f'Condition number S: {np.linalg.cond(S_tilde)}')
    ax2.legend()
    plt.show()

    print("These are the paramters: ")  #don't want to print the theta's for now
    print(Thets)
    print("This is the difference in parameters: ")
    print(Thets-prev_parameters)
    prev_parameters = Thets

    if energy_array_LM[n]<(-1.095): ###########################stop condition
            print("Terminating early wrt absolute value")
            break

t_1_lm = time.process_time()
lm_scaling = lambda n, k, H_len : ((n*n+n)*H_len + (n*n-n)) #lambda function to calculate how many times circ is done to get H and S

#######################     Plotting     ############################################
fig, ax = plt.subplots(2,2)
ax[0,0].plot(n_array, energy_array_LM, label='S{}, {:.2f} seconds and {} executions + hs'.format(Regularization, t_1_lm-t_0_lm, dev_lm.num_executions+lm_scaling(Thets.size, len(non_temp_k_ar), len(H_VQE_coeffs))))
#ax[0,0].plot(n_array2, energy_array_LM2, label='S0.1, Alt method')
#ax[0,0].set_yscale('log')
ax[0,0].legend()
ax[0,0].set_title(f'Linear Method w/ final value: {eee}')
ax[0,1].plot(n_array_adam, energy_array_adam, label='{:.2f} seconds and {} executions'.format(t_1_adam-t_0_adam, dev_adam.num_executions))
ax[0,1].set_title('Adam')
ax[0,1].legend()
ax[1,0].plot(n_array_grad, energy_array_grad, label='{:.2f} seconds and {} executions'.format(t_1_grad-t_0_grad, dev_grad.num_executions))
#ax[1,0].plot(n_array2, energy_array_LM2, label='{:.2f} seconds and {} executions'.format(t_1_lm2-t_0_lm2, 0) )
ax[1,0].set_title('Grad')
#ax[1,0].set_title('LM alternative_way')
ax[1,0].legend()
ax[1,1].plot(n_array_scipy, energy_array_scip, label='{:.2f} seconds and {} executions'.format(t_1_scipy-t_0_scipy, dev_scipy.num_executions))
#ax[1,1].scatter(iterations_scipy, energy_scipy, label='{:.2f} seconds'.format(t_1_scipy-t_0_scipy))
ax[1,1].set_title('Scipy')
ax[1,1].legend()

for axi in ax.flat:
    axi.set(xlabel='Iterations', ylabel='Energy')

plt.show()
