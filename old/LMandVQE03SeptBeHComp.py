"""
name: Most recent version LM and VQE algorithm, plus Adam, Scipy and Grad
last edited: 25/08

"""
import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#import LMLibrary2_david as LM  ##contains all self-made functions needed for Linear method optimization
import LMLibrary3BeH as LM
import LMLibrary2 as LM_R  
import HamiltoniansLibrary as HML
import copy as cp
import time
import pandas as pd
from scipy.interpolate import interp1d
from jax.config import config

config.update("jax_enable_x64", True)
import jax
np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

# Configuration
plot_lm = False
num_steps_lm = 50
num_steps_adam = 300
num_steps_grad = 700

#######################    input information        ##############################################################
#np.random.seed(542)
#The variational circuit
U_gates = np.array([['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'],['RZ', 'RY', 'RZ']]) ##the U gates in order
entangle_gates = np.array([[2,3], [2,0], [3,1]]) ###the entangled gates at the end

#U_gates = np.array([['RY', 'RZ'], ['RY', 'RZ'], ['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ']]) ##the U gates in order
#entangle_gates = np.array([[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7]]) ###the entangled gates at the end
Thets = np.random.normal(0, np.pi, (6,3))
print(Thets)
Thets_start = cp.copy(Thets)

#Hamiltonian
H_VQE_coeffs = HML.H_BeH2_coeffs
H_VQE_gates = HML.H_BeH2_terms
print(H_VQE_coeffs)
print(H_VQE_gates)
Hamilt_written_out = HML.creating_written_out_ham(H_VQE_coeffs, H_VQE_gates)

#Other information
no_of_gates = len(U_gates)
no_of_wires =7
matrix_length = Thets.size
execs_per_H = 171
name_csv_file = "BeHSystem50ItsReg0.01.csv"
name_csv_file_execs = "BeHSystem50ItsReg0.01ExecsFile"

##################################            Devices          ##########################################################

dev_grad = qml.device('default.qubit', wires=no_of_wires-1, shots=None)
dev_adam = qml.device('default.qubit', wires=no_of_wires-1, shots=None)
dev_scipy = qml.device('default.qubit', wires=no_of_wires-1, shots=None)
dev_lm = qml.device('default.qubit', wires=no_of_wires-1, shots=None)

#########################   MAIN           ############################

def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0, 0, 0], requires_grad=False), wires=wires)
    for i in wires:
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[3,4])
    qml.CNOT(wires=[4,5])
    
#def circuit(params, wires):
#    print("Going into this circuit")
#    qml.BasisState(np.array([0,0,0,0,0,0,0,0,0], requires_grad=False), wires=wires)
#    for i in wires:
#        qml.RY(params[i][0], wires=i)
#        qml.RZ(params[i][1], wires=i)
#    qml.CNOT(wires=[0,1])
#    qml.CNOT(wires=[1,2])
#    qml.CNOT(wires=[2,3])
#    qml.CNOT(wires=[3,4])
#    qml.CNOT(wires=[4,5])
#    qml.CNOT(wires=[5,6])
#    qml.CNOT(wires=[6,7])


max_iterations = 200

# These functions take the parameters as unflattened shape - and energy_grad outputs an unflattened gradient
# optimize=True makes the function a bit cheaper
get_energy = lambda device: qml.ExpvalCost(circuit, Hamilt_written_out, device, optimize=True)
get_energy_jit = lambda device: jax.jit(qml.ExpvalCost(circuit, Hamilt_written_out, device, optimize=True,interface='jax'))



#################################           Linear Method        ###########################################
# The linear method will use the unflattened functions

t_0_lm = time.process_time()

# copy initial parameters
Thets = Thets_start.copy()

# Initial point
energy = get_energy(dev_lm)
energy_grad = qml.grad(energy)
energy_jit = get_energy_jit(dev_lm)
E_start = energy_jit(Thets)
#Initialize memory
iterations_lm = [0]
energies_lm = [E_start]

###For the naming: 
Regularization = 0.01

print(f"Very initial parameters:\n{Thets}")
print(f"Starting energy associated with these parameters:\n{E_start}")
prev_parameters = Thets

for n in range(num_steps_lm):
    H = LM.H_Matrix_final_calc(U_gates, Thets, H_VQE_gates, H_VQE_coeffs, entangle_gates)
    S = LM.S_Matrix_final_calc_newy(U_gates, Thets)
    S_tilde = LM.S_tilde_matrix(S, Regularization)
    _thetas = []
    _energies = []
    condition_numbers_H = []
    n_not_converged = 0
    gradient = energy_grad(Thets)
    regularizations = np.linspace(0.01,  1, 200, endpoint=True) ##disabling the tail
    
    for k_ind, k in enumerate(regularizations): 
        H_tilde = LM.H_tilde_matrix(H, energies_lm[-1], gradient, k)
        try:
            update = LM.smallest_real_w_norm_optimiz(H_tilde, S_tilde)
            #update_own = LM.smallest_real_w_norm_optimiz_eig(H_tilde, S_tilde)
            _Thets = LM.new_thetsy(update, Thets)
            #Energ_temp = LM.energy_calc(circuit, Hamilt_written_out, dev_lm, Thets_temp)
            _E = energy_jit(_Thets)
            _thetas.append(_Thets)
            _energies.append(_E)
            condition_numbers_H.append(np.linalg.cond(H_tilde))
        except:
            n_not_converged +=1
            regularizations = np.delete(regularizations, np.where(regularizations==k))

        #except:
         #   print("Could not converge")
         #   print(f"Condition number H: {np.linalg.cond(H_tilde)}")
         #   print(f"Condition number S: {np.linalg.cond(S_tilde)}\n")
         #   print(S_tilde)
         #   n_not_converged += 1

    _thetas = np.array(_thetas).reshape((len(_thetas), Thets.size))
    arg_chosen = np.argmin(_energies)
    Thets = np.reshape(_thetas[arg_chosen], Thets.shape) ##choose the new theta's of the lowest energy
    energies_lm.append(_energies[arg_chosen])###pick the lowest energy. 
    #max_k = LM.finding_start_of_tail(_energies, regularizations, 0.001)
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
        ############plottingg########################################
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.scatter(regularizations, _energies)
        #plt.xlabel('k-value')
        #plt.ylabel('Energy')
        #plt.title('K-cutoff point:', max_k)
        ax1.plot(regularizations, _energies, label = f'Condition number S: {np.linalg.cond(S_tilde)}, not converged: {n_not_converged}')
        ax1.legend()
        ax1.set(ylabel='Energy')
        ax2.set(xlabel='K-value', ylabel = 'Condition number H')
        ax2.scatter(regularizations, condition_numbers_H, label=f'Condition number S: {np.linalg.cond(S_tilde)}')
        ax2.legend()
        plt.show()

        print("These are the parameters: ")  #don't want to print the theta's for now
        print(Thets)
        print("This is the difference in parameters: ")
        print(Thets-prev_parameters)
        prev_parameters = Thets

    #if energies_lm[n]<(-1.095): ###########################stop condition
    #    print("Terminating early wrt absolute value")
    #    break

t_1_lm = time.process_time()
lm_scaling = lambda n : n*2*171 #lambda function to calculate how many times circ is done to get H and S, specific to HH (will have to test the other systems seperately.)

##################################           Scipy BFGS Method        #######################################################
t_0_scipy = time.process_time()

energy = get_energy(dev_scipy)
energy_grad = qml.grad(energy)
# Convert to flat cost function and gradient function
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

#print(res)

#################################         Adam Method             #########################################################

t_0_adam = time.process_time()
energy_grad = qml.grad(get_energy(dev_adam))
energy_jit = get_energy_jit(dev_adam)
#initializing
Thets = Thets_start.copy()
energies_adam = []
iterations_adam = []

opt = qml.AdamOptimizer(stepsize=0.1)

for n in range(num_steps_adam):
    Thets, Prev_energ = opt.step_and_cost(energy_jit, Thets, grad_fn=energy_grad)
    if n>2 and (np.abs(Prev_energ-energies_adam[-1])<1e-6):
        print("Reached convergence!")
        break
    energies_adam.append(Prev_energ)
    iterations_adam.append(n)
t_1_adam = time.process_time()

###############################         Gradient Descent Method    ########################################################

t_0_grad = time.process_time()
energy_grad = qml.grad(get_energy(dev_grad))
energy_jit = get_energy_jit(dev_grad)
#initializing
Thets = Thets_start.copy()
energies_grad = []
iterations_grad = []

opt = qml.GradientDescentOptimizer(stepsize=0.4)

for n in range(num_steps_grad):
    Thets, Prev_energ = opt.step_and_cost(energy_jit, Thets, grad_fn=energy_grad)
    if n>2 and (np.abs(Prev_energ-energies_grad[-1])<1e-6):
        print("Reached convergence!")
        break
    energies_grad.append(Prev_energ)
    iterations_grad.append(n)
t_1_grad = time.process_time()

####################################    Panda-ing  ##########################################################################






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
labels = ["Linear Method", "BFGS", "Gradient Descent", "ADAM"]
start_times = [t_0_lm, t_0_scipy, t_0_grad, t_0_adam]
end_times = [t_1_lm, t_1_scipy, t_1_grad, t_1_adam]
zipsies = zip(labels, all_final_energies)
energies_and_name = dict(zipsies)
zipsies_execs = zip(labels, all_execution_counts)
execs_and_name = dict(zipsies_execs)
print(energies_and_name)

#df = pd.DataFrame(energies_and_name, index = [0])
df = pd.read_csv(name_csv_file)
df = df.append(energies_and_name, ignore_index = True)
df.drop(columns = df.columns[0], axis=1, inplace=True)
print(df)
df.to_csv(name_csv_file)

#df2 = pd.DataFrame(execs_and_name, index = [0])
df2 = pd.read_csv(name_csv_file_execs)
df2 = df2.append(execs_and_name, ignore_index = True)
df2.drop(columns = df2.columns[0], axis=1, inplace=True)
print(df2)
df2.to_csv(name_csv_file_execs)







for K in range(4):
    _time = end_times[K] - start_times[K]
    _label = labels[K]+f" {_time:.1f} sec and {all_execution_counts[K]} execs"
    ax.plot(all_iterations[K], all_energies[K], label=_label)

ax.set(xlabel='Iterations', ylabel='Energy')
ax.legend(bbox_to_anchor=(0.0,1.0), loc="lower left", ncol=2)
print("Final energies:")
for K in range(4):
    print(f"{labels[K]}: {all_energies[K][-1]}")

plt.show()
