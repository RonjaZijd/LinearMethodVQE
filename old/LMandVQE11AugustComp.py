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
#from IPython import get_ipython

np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})
name_opt_energ_file = "100ks15itsstats.csv"

#######################    input information        ##############################################################

#The variational circuit
U_gates = np.array([['RY', 'RZ'], ['RY', 'RZ'], ['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ'],['RY', 'RZ'] ]) ##the U gates in order
entangle_gates = np.array([[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9], [9,10], [10, 11]]) ###the entangled gates at the end
Thets = np.random.normal(0, np.pi, (12,2))
Thets_start = cp.copy(Thets)

#Hamiltonian

symbols = ["Li", "H"]
coordinates = np.array([0.0,0.0, 0.403635, 0.0,0.0, -1.210905])

H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
H_VQE_coeffs = H.coeffs
H_VQE_gates = LM.hamiltonian_into_gates(H)
Hamilt_written_outt = H


shapy = (12,2)


#Other information
no_of_gates = len(U_gates)
no_of_wires =13
matrix_length=24 ##starting from 1

#name_csv_file = "EnergyTry1.csv"

##################################            Devices          ##########################################################

dev_grad = qml.device('default.qubit', wires=12)
dev_adam = qml.device('default.qubit', wires=12)
dev_scipy = qml.device('default.qubit', wires=12)
dev_lm = qml.device('default.qubit', wires=12)
dev2 = qml.device('default.qubit', wires=12)

#########################   MAIN           ############################

#def circuit(params, wires):
#    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
#    for i in wires:
#        qml.Rot(params[i][2], params[i][1], params[i][0], wires=i)
#    qml.CNOT(wires=[2, 3])
#    qml.CNOT(wires=[2, 0])
#    qml.CNOT(wires=[3, 1])

def circuit(params, wires):
    qml.BasisState(np.array([0,0,0,0,0,0,0,0,0,0,0,0], requires_grad=False), wires=wires)
    for i in wires:
        qml.RY(params[i][0], wires=i)
        qml.RZ(params[i][1], wires=i)
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[3,4])
    qml.CNOT(wires=[4,5])
    qml.CNOT(wires=[5,6])
    qml.CNOT(wires=[6,7])
    qml.CNOT(wires=[7,8])
    qml.CNOT(wires=[8,9])
    qml.CNOT(wires=[9,10])
    qml.CNOT(wires=[10,11])


max_iterations = 200

def energy_calc(circuit, Hamilt_written_out, device, Thets):
    costy = qml.ExpvalCost(circuit, Hamilt_written_out, device)  ###for now we're simply going to do it with the specific case circuit
    energ = costy(Thets)
    return energ

def energy_grad(Thets):
    global circuit
    global Hamilt_written_outt
    global dev2
    Thets = np.reshape(Thets, shapy)
    energy_func = qml.ExpvalCost(circuit, Hamilt_written_outt, dev2)
    grad_func = qml.grad(energy_func)
    E_gra = grad_func(Thets)
    #E_gra = np.reshape(E_gra, (E_gra.size, 1))
    return E_gra.flatten()

"""
##################################           Scipy BFGS Method        #######################################################
energy_array_scip = []
n_array_scipy = []

def cost_func_scipy(Thets):
    global energy_array_scip #all global, needed to track the progress of SciPy function
    global circuit
    global Hamilt_written_outt
    global dev2
    Thets = np.reshape(Thets, shapy)
    cost_function_scipy = qml.ExpvalCost(circuit, Hamilt_written_outt, dev_scipy)
    energy_array_scip = np.append(energy_array_scip, energy_calc(circuit, Hamilt_written_outt, dev2, Thets))
    return cost_function_scipy(Thets)

t_0_scipy = time.process_time()
Thets = np.reshape(Thets, (matrix_length,1))
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
""" 

#################################           Linear Method        ###########################################

t_0_lm = time.process_time()
#Initializing
n_array = []
energy_array_LM = []
Thets_start = np.array([[3.28, 4.61], [5.71, 2.27], [2.75, 2.63], [4.56, 4.29],[6.05, 1.82], [1.88, 4.95], [0.13, 0.60], [2.00, 1.43], [4.72, 6.18], [1.78, 4.98], [3.33, 1.05], [2.84, 0.85]])
Thets=Thets_start
eee=0
energy_old =0
times_shaken = 0

###For the naming: 
Regularization = 0
K_max = 109
name_run = "R01K100"

max_k =1

def finding_start_of_tail(array, k_array, tol):
    compare_val = array[-1]
    print("The compare value is: ", compare_val)
    for i in range(len(k_array)):
        k_max = k_array[-i]
        #print("The difference with k", k_array[i])
        if np.abs(compare_val-array[-1-i])>tol:
            print("Tail ends at: ", k_max)
            break
    return k_max

print("I've started running!")

for n in range(5) :
    H = LM.H_Matrix_final_calc(U_gates, Thets, H_VQE_gates, H_VQE_coeffs, entangle_gates)
    S = LM.S_Matrix_final_calc_newy(U_gates, Thets)
    
    S_tilde = LM.S_tilde_matrix(S, Regularization)

    temp_thets_ar = []
    temp_energ_ar = []
    #non_temp_k_ar = [1, 0.2, 0.4, 0.6, 0.8, 0]
    non_temp_k_ar = np.linspace(0, max_k, 100, endpoint=True)
    
    for k in non_temp_k_ar: 
        H_tilde = LM.H_tilde_matrix(H, eee, LM.E_grad(Thets, Hamilt_written_outt, circuit, dev_lm), k)
        update = LM.smallest_real_w_norm_optimiz(H_tilde, S_tilde)
        Thets_temp = LM.new_thetsy(update, Thets)
        Energ_temp = LM.energy_calc(circuit, Hamilt_written_outt, dev_lm, Thets_temp)
        print("Just calculated the energy with ", k)
        temp_thets_ar = np.append(temp_thets_ar, Thets_temp)
        temp_energ_ar = np.append(temp_energ_ar, Energ_temp)
    full_temp_energ_ar = []
    #using interpolation to choose the appropriate k value and using that to update:
    # f2 = interp1d(non_temp_k_ar, temp_energ_ar, kind='cubic')
    # f = interp1d(non_temp_k_ar,  temp_energ_ar)
    # f3 = interp1d(non_temp_k_ar,  temp_energ_ar, kind='quadratic')
    # knew = xnew = np.linspace(np.min(non_temp_k_ar), np.max(non_temp_k_ar), num=41, endpoint=True)
    # for k in knew: 
    #     full_temp_energ_ar = np.append(full_temp_energ_ar, f2(k))
    # k_chosen = knew[np.argmin(full_temp_energ_ar)]
    # H_tilde = LM.H_tilde_matrix(H, eee, LM.E_grad(Thets, Hamilt_written_outt, circuit, dev_lm), k_chosen)
    # update = LM.smallest_real_w_norm_optimiz(H_tilde, S_tilde)
    # Thets_very_temp = LM.new_thetsy(update, Thets)
    # e_e = LM.energy_calc(circuit, Hamilt_written_outt, dev_lm, Thets_very_temp)

    temp_thets_ar = np.reshape(temp_thets_ar, (len(non_temp_k_ar), Thets.size))
    # #arg_chosen = LM.different_regularization(temp_energ_ar, 0.000001)
    arg_chosen = np.argmin(temp_energ_ar)
    Thets = np.reshape(temp_thets_ar[arg_chosen], Thets.shape) ##choose the new theta's of the lowest energy
    eee = temp_energ_ar[arg_chosen] ###pick the lowest energy. 

    max_k = finding_start_of_tail(temp_energ_ar, non_temp_k_ar, 0.02)


    energy_array_LM = np.append(energy_array_LM, eee)
    n_array = np.append(n_array, n)
    
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
    plt.scatter(non_temp_k_ar, temp_energ_ar)
    plt.xlabel('k-value')
    #lt.ylabel('Energy')
    plt.plot(non_temp_k_ar, temp_energ_ar)
    #get_ipython().run_line_magic('matplotlib', 'inline')
    plt.show()
    print("These are the paramters: ")  #don't want to print the theta's for now
    print(Thets % (2*np.pi)) #not surewhy thisd didntwoork just now

    #if energy_array_LM[n]<(-1.095):
    #        print("Terminating early wrt absolute value")
    #        break
    #if n>7:
    #    if np.abs(energy_old-energy_array_LM[n])<0.0001:
        #if LM.standard_deviation(energy_array_LM, energy_array_LM[n], 0.001): #condition on shaking
    #        Thets = LM.shake_of_thets(Thets)
     #       times_shaken = times_shaken+1
     #   else: 
     #       times_shaken = 0
    #print("Times shaken is: ", times_shaken)
    #if times_shaken>3:
     #   Thets = LM.big_shake(Thets)
     #   print("Big Shake")
    #energy_old = energy_array_LM[n]

t_1_lm = time.process_time()

lm_scaling = lambda n, k, H_len : ((n*n+n)*H_len + (n*n-n)) #lambda function to calculate how many times circ is done to get H and S


######printing information related to scaling
print("Executions used per device: ")
print("For scipy: ", dev_scipy.num_executions)
print("For Adam: ", dev_adam.num_executions)
print("For Grad: ", dev_grad.num_executions)
print("For LM (the energy executions): ", dev_lm.num_executions)
print("For LM (The H and S calculations): ", lm_scaling(Thets.size, len(non_temp_k_ar), len(H_VQE_coeffs)))



################For the stats

#energy_lm = energy_array_LM[-1]
#energy_adam = energy_array_adam[-1]
#energy_grad = energy_array_grad[-1]
#energy_scipy = energy_array_scip[-1]

#energy_dic = {"LM":[energy_lm], "Adam": [energy_adam], "Grad": [energy_grad], "SciPy": [energy_scipy]}
#energy_dic2 = {"LM":energy_lm, "Adam": energy_adam, "Grad": energy_grad, "SciPy": energy_scipy}
#df_temp_en_opt = pd.DataFrame(energy_dic)
#df_temp_en_opt.to_csv(name_opt_energ_file) #only unselect the first time running it:

##df_en_opt = pd.read_csv(name_opt_energ_file)
#df_en_opt = df_en_opt.append(energy_dic2, ignore_index=True, sort=False)
#df_en_opt.drop(columns=df_en_opt.columns[0], axis=1, inplace=True)
#df_en_opt.to_csv(name_opt_energ_file)




#####Adding it to my Panda file: 
#df = pd.read_csv(name_csv_file)
#df_temp = pd.DataFrame(energy_array_LM, columns=[name_run])
#df = pd.concat([df, df_temp], axis=1)
#df.to_csv(name_csv_file)



"""

######Plotting
fig, ax = plt.subplots(2,2)
ax[0,0].plot(n_array, energy_array_LM, label='S0.01, {:.2f} seconds and {} executions + hs'.format(t_1_lm-t_0_lm, dev_lm.num_executions+lm_scaling(Thets.size, len(non_temp_k_ar), len(H_VQE_coeffs))))
#ax[0,0].plot(n_array2, energy_array_LM2, label='S0.1, Alt method')
ax[0,0].legend()
ax[0,0].set_title('Linear Method')
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

#get_ipython().run_line_magic('matplotlib', 'tk')
plt.show()
"""
