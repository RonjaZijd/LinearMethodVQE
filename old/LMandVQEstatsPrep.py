import pennylane as qml
from pennylane import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import LMLibrary2 as LM  ##contains all self-made functions needed for Linear method optimization
import copy as cp
import time
import pandas as pd

np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

######################

run=2
name_LM_file = "S0tr.csv"
name_opt_energ_file = "S0energ.csv"
name_opt_time_file = "S0time.csv"
name_opt_exec_file = "S0ex.csv"
name_opt_iter_file = "S0it.csv"

#####################  Input information    ####################################################

#The variational circuit
U_gates = np.array([['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ']]) ##the U gates in order
entangle_gates = np.array([[2,3], [2,0], [3,1]]) ###the entangled gates at the end
Thets = np.random.normal(0, np.pi, (4,3))
Thets_start = cp.copy(Thets)
#Hamiltonian
H_VQE_coeffs = [-0.24274280513140462,-0.24274280513140462,-0.04207897647782276,0.1777128746513994,0.17771287465139946,0.12293305056183798,0.12293305056183798,0.1676831945771896,0.1676831945771896,0.17059738328801052,0.17627640804319591,-0.04475014401535161,-0.04475014401535161,0.04475014401535161,0.04475014401535161]
H_VQE_gates = [['Z3'],['Z4'],['I1'],['Z2'],['Z1'],['Z1', 'Z3'],['Z2','Z4'],['Z1','Z4'],['Z2','Z3'],['Z1','Z2'],['Z3', 'Z4'], ['Y1', 'Y2', 'X3', 'X4'],['X1', 'X2', 'Y3', 'Y4'], ['Y1', 'X2', 'X3', 'Y4'] ]
Hamilt_written_outt = -0.24274280513140462*qml.PauliZ(wires=2) + -0.24274280513140462*qml.PauliZ(wires=3) +  0.1777128746513994*qml.PauliZ(wires=1)+0.17771287465139946*qml.PauliZ(wires=0)+0.12293305056183798*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))+0.12293305056183798*(qml.PauliZ(wires=1)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=0)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=1)@qml.PauliZ(wires=2)) +0.17059738328801052*(qml.PauliZ(wires=0)@qml.PauliZ(wires=1))+0.17627640804319591*(qml.PauliZ(wires=2)@qml.PauliZ(wires=3))+-0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliY(wires=1)@qml.PauliX(wires=2)@qml.PauliX(wires=3))+-0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliX(wires=1)@qml.PauliY(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliX(wires=1)@qml.PauliX(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliY(wires=1)@qml.PauliY(wires=2)@qml.PauliX(wires=3))
Hamilt_written_out = -0.2*qml.PauliZ(wires=2) + -0.56*qml.PauliZ(wires=3) + 0.122*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))
#Other information
no_of_gates = len(U_gates)
no_of_wires =5
matrix_length=12 ##starting from 1

####################

dev_grad = qml.device('default.qubit', wires=4)
dev_adam = qml.device('default.qubit', wires=4)
dev_scipy = qml.device('default.qubit', wires=4)
dev_lm = qml.device('default.qubit', wires=4)
dev2 = qml.device('default.qubit', wires=4)

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

####################### SCIPY

def cost_func_scipy(Thets):
    Thets = np.reshape(Thets, (4,3))
    cost_function_scipy = qml.ExpvalCost(circuit, Hamilt_written_outt, dev_scipy)
    return cost_function_scipy(Thets)

t_0_scipy = time.process_time()
Thets = np.reshape(Thets, (12,1))
res = sp.optimize.minimize(cost_func_scipy, Thets, method = 'BFGS', jac=energy_grad)

t_1_scipy = time.process_time()

iterations_scipy = res.get('nfev')
energy_scipy = res.get('fun')
time_scipy = t_1_scipy-t_0_scipy
exec_scipy = dev_scipy.num_executions

#######################  ADAM

t_0_adam = time.process_time()
Thets=Thets_start
energy_array_adam = []

cost_function_adam = qml.ExpvalCost(circuit, Hamilt_written_outt, dev_adam)  #functions on which optimization is performed
opt = qml.AdamOptimizer(stepsize=0.1)

for n in range(300):
    Thets, Prev_energ = opt.step_and_cost(cost_function_adam, Thets)
    if n>2:
        if (np.abs(Prev_energ-energy_array_adam[-1])<0.000001) & (n>1):
            print("Reached convergence!")
            iterations_adam = n
            break
    energy_array_adam = np.append(energy_array_adam, Prev_energ)
    
t_1_adam = time.process_time()

energy_adam = energy_array_adam[-1]
time_adam = t_1_adam-t_0_adam
exec_adam = dev_adam.num_executions

##################### GRAD

t_0_grad = time.process_time()
Thets=Thets_start
energy_array_grad = []

cost_function_grad_descent = qml.ExpvalCost(circuit, Hamilt_written_outt, dev_grad)
opt = qml.GradientDescentOptimizer(stepsize=0.4)

for n in range(700):
    Thets, Prev_energ = opt.step_and_cost(cost_function_grad_descent, Thets)
    if n>2:
        if (np.abs(Prev_energ-energy_array_grad[-1])<0.000001) & (n>1):
            print("Reached convergence!")
            iterations_grad = n
            break
    energy_array_grad = np.append(energy_array_grad, Prev_energ)
   
t_1_grad = time.process_time()  

time_grad = t_1_grad-t_0_grad
energy_grad = energy_array_grad[-1]
exec_grad = dev_grad.num_executions

#####################  LM

t_0_lm = time.process_time()

n_array = []
energy_array_LM = []
Thets=Thets_start
eee=LM.energy_calc(circuit, Hamilt_written_outt, dev_lm, Thets)
energy_old =0
times_shaken = 0

for n in range(100) :
    H = LM.H_Matrix_final_calc(U_gates, Thets, H_VQE_gates, H_VQE_coeffs, entangle_gates)
    S = LM.S_Matrix_final_calc_newy(U_gates, Thets)
    
    S_tilde = LM.S_tilde_matrix(S, 0)

    temp_thets_ar = []
    temp_energ_ar = []
    non_temp_k_ar = [100, 10, 1, 0.1]
    
    for k in non_temp_k_ar: 
        H_tilde = LM.H_tilde_matrix(H, eee, LM.E_grad(Thets, Hamilt_written_outt, circuit, dev_lm), k)
        update = LM.smallest_real_w_norm_optimiz(H_tilde, S_tilde)
        Thets_temp = LM.new_thetsy(update, Thets)
        Energ_temp = LM.energy_calc(circuit, Hamilt_written_outt, dev_lm, Thets_temp)
        temp_thets_ar = np.append(temp_thets_ar, Thets_temp)
        temp_energ_ar = np.append(temp_energ_ar, Energ_temp)

    temp_thets_ar = np.reshape(temp_thets_ar, (len(non_temp_k_ar), Thets.size))
    arg_chosen = LM.different_regularization(temp_energ_ar, 0.00001)
    #arg_chosen = np.argmin(temp_energ_ar)
    Thets = np.reshape(temp_thets_ar[arg_chosen], Thets.shape) ##choose the new theta's of the lowest energy
    eee = temp_energ_ar[arg_chosen] ###pick the lowest energy. 
    energy_array_LM = np.append(energy_array_LM, eee)

    print("At iteration: ", n, "with energy: ", eee)

    if n>5:
        #if np.abs(energy_old-energy_array_LM[n])<0.0001:
        if LM.standard_deviation(energy_array_LM, energy_array_LM[n], 0.001): #condition on shaking
            Thets = LM.shake_of_thets(Thets)
            times_shaken = times_shaken+1
        else: 
            times_shaken = 0
    if times_shaken>3:
        Thets = LM.big_shake(Thets)
        print("Big Shake")
    energy_old = energy_array_LM[n]

t_1_lm = time.process_time()

lm_scaling = lambda n, k, H_len : ((n*n+n)*H_len + (n*n-n)) #lambda function to calculate how many times circ is done to get H and S

energy_lm = energy_array_LM[-1]
iterations_lm = 100
exec_lm = dev_lm.num_executions
time_lm = t_1_lm-t_0_lm

###########################################  Panda's:

##LM dataframe

df_temp = pd.DataFrame(energy_array_LM, columns=[run])
df_temp2 = pd.Series(energy_array_LM)
#df_temp.to_csv(name_LM_file)

df_lm = pd.read_csv(name_LM_file)
df_lm = pd.concat([df_lm, df_temp2], axis=1)
df_lm.drop(columns=df_lm.columns[0], axis=1, inplace=True)
df_lm.to_csv(name_LM_file)

##Energy opt dataframe
energy_dic = {"LM":[energy_lm], "Adam": [energy_adam], "Grad": [energy_grad], "SciPy": [energy_scipy]}
energy_dic2 = {"LM":energy_lm, "Adam": energy_adam, "Grad": energy_grad, "SciPy": energy_scipy}
df_temp_en_opt = pd.DataFrame(energy_dic)
#df_temp_en_opt.to_csv(name_opt_energ_file) #only unselect the first time running it:

df_en_opt = pd.read_csv(name_opt_energ_file)
df_en_opt = df_en_opt.append(energy_dic2, ignore_index=True, sort=False)
df_en_opt.drop(columns=df_en_opt.columns[0], axis=1, inplace=True)
df_en_opt.to_csv(name_opt_energ_file)

#Iterations opt dataframe
iterations_dic = {"LM":[iterations_lm], "Adam": [iterations_adam], "Grad": [iterations_grad], "SciPy": [iterations_scipy]}
iterations_dic2 = {"LM":iterations_lm, "Adam": iterations_adam, "Grad": iterations_grad, "SciPy": iterations_scipy}
df_temp_it_opt = pd.DataFrame(iterations_dic)
#df_temp_it_opt.to_csv(name_opt_iter_file)

df_it_opt = pd.read_csv(name_opt_iter_file)
df_it_opt = df_it_opt.append(iterations_dic2, ignore_index=True, sort=False)
df_it_opt.drop(columns=df_it_opt.columns[0], axis=1, inplace=True)
df_it_opt.to_csv(name_opt_iter_file)

#Executions opt dataframe
execs_dic = {"LM":[exec_lm], "Adam": [exec_adam], "Grad": [exec_grad], "SciPy": [exec_scipy]}
execs_dic2 = {"LM":exec_lm, "Adam": exec_adam, "Grad": exec_grad, "SciPy": exec_scipy}
df_temp_exec_opt = pd.DataFrame(execs_dic)
#df_temp_exec_opt.to_csv(name_opt_exec_file)

df_exec_opt = pd.read_csv(name_opt_exec_file)
df_exec_opt = df_exec_opt.append(execs_dic2, ignore_index=True, sort=False)
df_exec_opt.drop(columns=df_exec_opt.columns[0], axis=1, inplace=True)
df_exec_opt.to_csv(name_opt_exec_file)

#Times opt dataframe
times_dic = {"LM":[time_lm], "Adam": [time_adam], "Grad": [time_grad], "SciPy": [time_scipy]}
times_dic2 = {"LM":time_lm, "Adam": time_adam, "Grad": time_grad, "SciPy": time_scipy}
df_temp_time_opt = pd.DataFrame(times_dic)
#df_temp_time_opt.to_csv(name_opt_time_file)

df_time_opt = pd.read_csv(name_opt_time_file)
df_time_opt = df_time_opt.append(times_dic2, ignore_index=True, sort=False)
df_time_opt.drop(columns=df_time_opt.columns[0], axis=1, inplace=True)
df_time_opt.to_csv(name_opt_time_file)