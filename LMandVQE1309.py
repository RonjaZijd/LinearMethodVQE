"""
Some good comments

"""
import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import LMLibrary3 as LM
import HamiltoniansLibrary as HML
import copy as cp
import time
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns

np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

# Configuration
plot_lm = True
num_steps_lm = 25
num_steps_adam = 300
num_steps_grad = 700
Regularization = 0.01
stop_condition = -1.095 #if the global minimum is known

#######################    input information        ##############################################################

#The variational circuit
U_gates = np.array([['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ']]) 
entangle_gates = np.array([[2,3], [2,0], [3,1]]) 

Thets = np.random.normal(0, np.pi, (4,3))
Thets_start = cp.copy(Thets)

#Hamiltonian
H_VQE_coeffs = HML.H_HH_coeffs
H_VQE_gates = HML.H_HH_gates
Hamilt_written_out = HML.creating_written_out_ham(H_VQE_coeffs, H_VQE_gates)




#Other information
no_of_gates = len(U_gates)
no_of_wires = 5
matrix_length = Thets.size

##################################            Devices          ##########################################################

dev_grad = qml.device('default.qubit', wires=no_of_wires-1, shots=None)
dev_adam = qml.device('default.qubit', wires=no_of_wires-1, shots=None)
dev_scipy = qml.device('default.qubit', wires=no_of_wires-1, shots=None)
dev_lm = qml.device('default.qubit', wires=no_of_wires-1, shots=None)

#########################   MAIN           ############################

def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)
    for i in wires:
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])
    
get_energy = lambda device: qml.ExpvalCost(circuit, Hamilt_written_out, device, optimize=True)

#################################           Linear Method        ###########################################
t_0_lm = time.process_time()
Thets = Thets_start.copy()

# Initial point
energy = get_energy(dev_lm)
energy_grad = qml.grad(energy)
E_start = energy(Thets)

#Initialize memory
iterations_lm = [0]
energies_lm = [E_start]
k_max = 1

for n in range(num_steps_lm):
    H = LM.H_Matrix_final_calc(U_gates, Thets, H_VQE_gates, H_VQE_coeffs, entangle_gates)
    S = LM.S_Matrix_final_calc_newy(U_gates, Thets)
    S_tilde = LM.S_tilde_matrix(S, Regularization)
    _thetas = []
    _energies = []
    _energiesh = []
    condition_numbers_H = []
    n_not_converged = 0
    gradient = energy_grad(Thets)
    regularizations = np.linspace(0.01,  1, 100, endpoint=True)
    np.savetxt('matrixS', S_tilde, delimiter=',') 
    
    for k_ind, k in enumerate(regularizations): 
        H_tilde = LM.H_tilde_matrix(H, energies_lm[-1], gradient, k)
        C = (np.linalg.inv(H_tilde)*S_tilde)
        try:
            #update = LM.smallest_real_w_norm_optimizz(C)
            update = LM.smallest_real_w_norm_optimiz(H_tilde, S_tilde)
            update_own = LM.smallest_real_w_norm_optimiz_eigh(H_tilde, S_tilde)
            _Thets = LM.new_thetsy(update, Thets)
            _E = energy(_Thets)
            _Eh = energy(LM.new_thetsy(update_own, Thets))
            _thetas.append(_Thets)
            _energies.append(_E)
            _energiesh.append(_Eh)
            condition_numbers_H.append(np.linalg.cond(H_tilde))
            
            np.savetxt(f'matrixH_k{k}', H_tilde, delimiter=',')

        except:
            print("Eigensolver did not converge")
            n_not_converged +=1
            regularizations = np.delete(regularizations, np.where(regularizations==k))

    _thetas = np.array(_thetas).reshape((len(_thetas), Thets.size))
    arg_chosen = np.argmin(_energies)
    Thets = np.reshape(_thetas[arg_chosen], Thets.shape) 
    energies_lm.append(_energies[arg_chosen])
    iterations_lm.append(n+1)
    
    print("-"*100) #printing things to see what the program is doing
    print("Iteration: ", n)
    print("Number of times not converged: ", n_not_converged)
    print("Chosen minimal index: ", arg_chosen)
    print("Actual k chosen", regularizations[arg_chosen])
    print("Energy chosen after update step: ", energies_lm[-1])
    print("These are the parameters: ")  #don't want to print the theta's for now
    print(Thets)

    if plot_lm:
        ################### Plot energy vs  regularization  ##########################
        plt.scatter(regularizations, _energies, label = 'sp.linalg.eig()')
        plt.scatter(regularizations, _energiesh, label = 'sp.linalg.eigh()')
        plt.legend()
        plt.xlabel('Regularization on H')
        plt.ylabel('Energy')
        plt.show()

    if energies_lm[n]<(stop_condition): ## stop condtion
        print("Terminating early wrt absolute value")
        break

t_1_lm = time.process_time()
lm_scaling = lambda n : n*2*78 #lambda function to calculate how many times circ is done to get H and S, specific to HH (will have to test the other systems seperately.)

##################################           Scipy BFGS Method        #######################################################
t_0_scipy = time.process_time()

energy = get_energy(dev_scipy)
energy_grad = qml.grad(energy)
energy_flat = lambda param: energy(param.reshape(Thets_start.shape))
energy_grad_flat = lambda param: energy_grad(param.reshape(Thets_start.shape)).reshape(Thets.size)
Thets_flat = Thets_start.copy().reshape(Thets_start.size)
E_start = energy_flat(Thets_flat)
energies_scipy = [E_start]

def cost_func_scipy(Thets):
    E = energy_flat(Thets)
    energies_scipy.append(E)
    return E

res = sp.optimize.minimize(cost_func_scipy, Thets_flat, method = 'BFGS', jac=energy_grad_flat)
iterations_scipy = np.arange(0, res.get('nfev')+1)

t_1_scipy = time.process_time()

#################################         Adam Method             #########################################################

t_0_adam = time.process_time()
energy_grad = qml.grad(get_energy(dev_adam))
energy = get_energy(dev_adam)
#initializing
Thets = Thets_start.copy()
energies_adam = []
iterations_adam = []

opt = qml.AdamOptimizer(stepsize=0.1)

for n in range(num_steps_adam):
    Thets, Prev_energ = opt.step_and_cost(energy, Thets, grad_fn=energy_grad)
    if n>2 and (np.abs(Prev_energ-energies_adam[-1])<1e-6):
        print("Reached convergence!")
        break
    energies_adam.append(Prev_energ)
    iterations_adam.append(n)
t_1_adam = time.process_time()

###############################         Gradient Descent Method    ########################################################

t_0_grad = time.process_time()
energy_grad = qml.grad(get_energy(dev_grad))
energy = get_energy(dev_grad)
#initializing
Thets = Thets_start.copy()
energies_grad = []
iterations_grad = []

opt = qml.GradientDescentOptimizer(stepsize=0.4)

for n in range(num_steps_grad):
    Thets, Prev_energ = opt.step_and_cost(energy, Thets, grad_fn=energy_grad)
    if n>2 and (np.abs(Prev_energ-energies_grad[-1])<1e-6):
        print("Reached convergence!")
        break
    energies_grad.append(Prev_energ)
    iterations_grad.append(n)
t_1_grad = time.process_time()


#######################     Plotting     ############################################
fig, ax = plt.subplots(1,1,figsize=(9,6))
all_energies = [energies_lm, energies_scipy, energies_grad, energies_adam]
all_final_energies = [energies_lm[-1], energies_scipy[-1], energies_grad[-1], energies_adam[-1]]
all_iterations = [iterations_lm, iterations_scipy, iterations_grad, iterations_adam]
all_execution_counts = [
    dev_lm.num_executions + lm_scaling(num_steps_lm),
    dev_scipy.num_executions,
    dev_grad.num_executions,
    dev_adam.num_executions,
]
labels = ["Linear Method Eig", "Linear Method Eigh", "BFGS", "Gradient Descent", "ADAM"]
start_times = [t_0_lm, t_0_scipy, t_0_grad, t_0_adam]
end_times = [t_1_lm,  t_1_scipy, t_1_grad, t_1_adam]

for K in range(4):
    _time = end_times[K] - start_times[K]
    _label = labels[K]+f" {_time:.1f} seconds"
    ax.plot(all_iterations[K], all_energies[K], label=_label)

ax.set(xlabel='Iterations', ylabel='Energy')
ax.legend(bbox_to_anchor=(0.0,1.0), loc="lower left", ncol=2)
print("Final energies:")
for K in range(4):
    print(f"{labels[K]}: {all_energies[K][-1]}")

plt.show()
