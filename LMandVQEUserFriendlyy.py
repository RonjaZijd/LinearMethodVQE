import pennylane as qml
from pennylane import numpy as np
import LMLibrary as LM
import HamiltoniansLibrary as HML
import matplotlib.pyplot as plt
import scipy as sp
import tkinter as tk
import copy as cp
import time

np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

# Configuration
plot_lm = False
num_steps_lm = 10
num_steps_adam = 300
num_steps_grad = 700
Regularization = 0.01
stop_condition = -1.095 #if the global minimum is known

###########################################  Hamiltonian input #####################################################

#The variational circuit
U_gates = np.array([['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ']]) 
entangle_gates = np.array([[2,3], [2,0], [3,1]]) 



#Hamiltonian
H_VQE_coeffs = HML.H_HH_coeffs
H_VQE_gates = HML.H_HH_gates
Hamilt_written_out = HML.creating_written_out_ham(H_VQE_coeffs, H_VQE_gates)

#Other information
no_of_gates = len(U_gates)
no_of_wires = 5 #including the auxiliary qubit

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

############################################  GUI for input of LM ###################################################

### functions needed ####
def finished(): 
    global gui_running
    gui_running = False

def run_input_gui():
    global gui_running
    inp_gui = tk.Tk()
    inp_gui.title('GUI for settings Linear Method')

    title_frame = tk.Frame(inp_gui)
    labels_frame = tk.Frame(inp_gui)
    inputs_frame = tk.Frame(inp_gui)
    finished_frame = tk.Frame(inp_gui)

    title_frame.pack()
    labels_frame.pack(side='left')
    inputs_frame.pack(side='right')
    finished_frame.pack()

    title = tk.Label(title_frame, text = "Settings for linear Method", anchor = 'w', font=("Courier", 16))  
    title.pack()

    label_1 = tk.Label(labels_frame, text = "Regularization on S: ")
    label_2 = tk.Label(labels_frame, text = "Minimum K: ")
    label_3 = tk.Label(labels_frame, text = "Maximum K: ")  
    label_4 = tk.Label(labels_frame, text = "Number of K values: ")
    label_5 = tk.Label(labels_frame, text = "Number of Iterations: ")
    label_6 = tk.Label(labels_frame, text = "Seed number: ")

    label_1.pack()
    label_2.pack()
    label_3.pack()
    label_4.pack()
    label_5.pack()
    label_6.pack()

    reg_entry = tk.Entry(inputs_frame)
    min_k_entry = tk.Entry(inputs_frame)
    max_k_entry = tk.Entry(inputs_frame)
    n_of_k_entry = tk.Entry(inputs_frame)
    n_of_its_entry = tk.Entry(inputs_frame)
    seed_n_entry = tk.Entry(inputs_frame)

    reg_entry.pack()
    min_k_entry.pack()
    max_k_entry.pack()
    n_of_k_entry.pack()
    n_of_its_entry.pack()
    seed_n_entry.pack()

    check_2 = tk.IntVar()
    check_3 = tk.IntVar()
    check_button_2 = tk.Checkbutton(finished_frame, text = "Plot the energy in k-space.", variable=check_2)
    check_button_3 = tk.Checkbutton(finished_frame, text = "Print condition number of H in k-space", variable=check_3)
    finish_but = tk.Button(finished_frame, text = "Filled everything in!", command = finished)
    
    check_button_2.pack()
    check_button_3.pack()
    finish_but.pack()

    gui_running = True
    while gui_running:
        inp_gui.update()

    try:
        reg = float(reg_entry.get())      
        min_k = float(min_k_entry.get())
        max_k = float(max_k_entry.get())
        if max_k<min_k:
            print(3/0)  ##probably a very bad way to throw an error
        n_of_k = int(n_of_k_entry.get())
        n_of_its = int(n_of_its_entry.get())
        seed_n = int(seed_n_entry.get())
    except:             ##if any of the above throw an error, default settings are chosen
        reg = 0.01
        min_k = 0.001
        max_k = 1
        n_of_k = 100
        n_of_its = 10
        seed_n = 102

    inp_gui.destroy()

    return reg, min_k, max_k, n_of_k, n_of_its, seed_n, check_2.get(), check_3.get()

reg, min_k, max_k, n_of_k, n_of_its, seed_n, plot_lm, plot_c = run_input_gui()

###########################################  Running the program ###################################################
np.random.seed(seed_n)
Thets = np.random.normal(0, np.pi, (4,3)) ##the shape of the parameters
Thets_start = cp.copy(Thets)
matrix_length = Thets.size

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


### LM ###
t_0_lm = time.process_time()
Thets = Thets_start.copy()

# Initial point
energy = get_energy(dev_lm)
energy_grad = qml.grad(energy)
E_start = energy(Thets)

#Initialize memory
iterations_lm = [0]
energies_lm = [E_start]

for n in range(n_of_its):
    H = LM.H_Matrix_final_calc(U_gates, Thets, H_VQE_gates, H_VQE_coeffs, entangle_gates)
    S = LM.S_Matrix_final_calc_newy(U_gates, Thets)
    S_tilde = LM.S_tilde_matrix(S, Regularization)
    _thetas = []
    _energies = []
    condition_numbers_H = []
    n_not_converged = 0
    gradient = energy_grad(Thets)
    regularizations = np.linspace(min_k,  max_k, n_of_k, endpoint=True) 
    
    for k_ind, k in enumerate(regularizations): 
        H_tilde = LM.H_tilde_matrix(H, energies_lm[-1], gradient, k)
        try:
            eigvalys, update = LM.smallest_real_w_norm_optimiz_eigh(H_tilde, S_tilde)
            _Thets = LM.new_thetsy(update, Thets)
            _E = energy(_Thets)
            _thetas.append(_Thets)
            _energies.append(_E)
            condition_numbers_H.append(np.linalg.cond(H_tilde))
        except:
            print("Eigensolver did not converge")
            n_not_converged +=1
            regularizations = np.delete(regularizations, np.where(regularizations==k))

    _thetas = np.array(_thetas).reshape((len(_thetas), Thets.size))
    arg_chosen = np.argmin(_energies)
    Thets = np.reshape(_thetas[arg_chosen], Thets.shape) 
    energies_lm.append(_energies[arg_chosen])
    iterations_lm.append(n+1)
    
    print("-"*100) #printing things to see state of the program while it's running
    print("Iteration: ", n)
    print("Number of times not converged: ", n_not_converged)
    print("Chosen minimal index: ", arg_chosen)
    print("Actual k chosen", regularizations[arg_chosen])
    print("Energy chosen after update step: ", energies_lm[-1])
    print("These are the parameters: ")  
    print(Thets)

    if plot_lm:
        ################### Plot energy vs  regularization  ##########################
        plt.scatter(regularizations, _energies, label = 'sp.linalg.eig()')
        plt.legend()
        plt.xlabel('Regularization on H')
        plt.ylabel('Energy')
        plt.show()

    if energies_lm[n]<(stop_condition): ## stop condtion
        print("Terminating early wrt absolute value")
        break

t_1_lm = time.process_time()
lm_scaling = lambda n : n*2*78 ##78 is specific to HH system and shows how many device executions per H/S matrix were done. 


############################################ Plotting functions ########################################################

energy_array_LM_log_prep = energies_lm + np.abs(stop_condition) ##step should only be included if minimum is negative
energy_array_adam_log_prep = energies_adam + np.abs(stop_condition)
energy_array_grad_log_prep = energies_grad + np.abs(stop_condition)
energy_array_scipy_log_prep = energies_scipy + np.abs(stop_condition)

def plot_lm():
    fig = plt.figure()
    ploty = fig.add_subplot(111)
    ploty.set(xlabel='Iterations', ylabel = 'Energy', title='Linear Method') 
    if check_1.get()==1:
        ploty.plot(iterations_lm, energy_array_LM_log_prep)
        ploty.set_yscale('log')
    else:
        ploty.plot(iterations_lm, energies_lm)
    fig.show()

def plot_scipy():
    fig = plt.figure()
    ploty = fig.add_subplot(111)
    ploty.set(xlabel='Iterations', ylabel = 'Energy', title='SciPy BFGS')
    if check_1.get()==1:
        ploty.plot(iterations_scipy, energy_array_scipy_log_prep)
        ploty.set_yscale('log')
    else:
        ploty.plot(iterations_scipy, energies_scipy)
    fig.show()

def plot_adam():
    fig = plt.figure()
    ploty = fig.add_subplot(111)
    ploty.set(xlabel='Iterations', ylabel = 'Energy', title='ADAM')
    if check_1.get()==1:
        ploty.plot(iterations_adam, energy_array_adam_log_prep)
        ploty.set_yscale('log')
    else:
        ploty.plot(iterations_adam, energies_adam)
    fig.show()

def plot_grad():
    fig = plt.figure()
    ploty = fig.add_subplot(111)
    ploty.set(xlabel='Iterations', ylabel = 'Energy', title='Gradient Descent')
    if check_1.get()==1:
        ploty.plot(iterations_grad, energy_array_grad_log_prep)
        ploty.set_yscale('log')
    else:
        ploty.plot(iterations_grad, energies_grad)
    fig.show()

def plot_all():
    fig, ax = plt.subplots(2,2)

    if check_1.get()==1:
        ax[0,0].plot(iterations_lm, energy_array_LM_log_prep)
        ax[0,1].plot(iterations_adam, energy_array_adam_log_prep)
        ax[1,1].plot(iterations_scipy, energy_array_scipy_log_prep)
        ax[1,0].plot(iterations_grad, energy_array_grad_log_prep)
    else: 
        ax[0,0].plot(iterations_lm, energies_lm)
        ax[0,1].plot(iterations_adam, energies_adam)
        ax[1,0].plot(iterations_grad, energies_grad)
        ax[1,1].plot(iterations_scipy, energies_scipy)

    ax[0,0].set_title('Linear Method')
    ax[0,1].set_title('Adam')
    ax[1,0].set_title('Grad')
    ax[1,1].set_title('Scipy')

    for axi in ax.flat:
        axi.set(xlabel='Iterations', ylabel='Energy')

    plt.show()

def plot_in_one_graph():
    fig = plt.figure()
    ploty = fig.add_subplot(111)
    if check_1.get()==1:
        ploty.plot(iterations_lm, energy_array_LM_log_prep, label = 'Linear Method')
        ploty.plot(iterations_grad, energy_array_grad_log_prep, label = 'Grad')
        ploty.plot(iterations_scipy, energy_array_scipy_log_prep, label = 'SciPy')
        ploty.plot(iterations_adam, energy_array_adam_log_prep, label = 'Adam')
        ploty.set_yscale('log')
    else:
        ploty.plot(iterations_grad, energies_grad, label = 'Grad')
        ploty.plot(iterations_adam, energies_adam, label = 'Adam')
        ploty.plot(iterations_scipy, energies_scipy, label = 'SciPy')
        ploty.plot(iterations_lm, energies_lm, label = 'Linear Method')
    ploty.legend()
    ploty.set(xlabel='Iterations', ylabel = 'Energy')
    fig.show()

############################################# GUI for plotting results ###############################################


plot_gui = tk.Tk()
plot_gui.title("GUI to plot the results")

title_frame2 = tk.Frame(plot_gui)
left_frame = tk.Frame(plot_gui)
right_frame = tk.Frame(plot_gui)

title_frame2.pack(fill = 'x')
left_frame.pack(side = 'left')
right_frame.pack(side = 'right')


title_1 = tk.Label(title_frame2, text = "Plots", anchor = 'w', font=("Courier", 16))   
title_1.pack()


title_2 = tk.Label(left_frame, text = "Individual" )
button_2 = tk.Button(left_frame, text = "Show the SciPy plot", command=plot_scipy)
button_3 = tk.Button(left_frame, text = "Show the Adam plot", command=plot_adam)
button_4 = tk.Button(left_frame, text = "Show the Grad plot", command=plot_grad)
button_5 = tk.Button(left_frame, text = "Show the LM plot", command=plot_lm)

title_2.pack()
button_2.pack()
button_3.pack()
button_4.pack()
button_5.pack()

check_1 = tk.IntVar()
check_button_1 = tk.Checkbutton(right_frame, text = "Show the y-axis on log scale.", variable=check_1)
title_3 = tk.Label(right_frame, text = "Together")
button_6 = tk.Button(right_frame, text = "Show all plots together", command = plot_all)
button_8 = tk.Button(right_frame, text = "Show all optimizers in one graph", command = plot_in_one_graph)
button_7 = tk.Button(right_frame, text = "Done here", command = finished)

title_3.pack()
button_6.pack()
button_8.pack()
check_button_1.pack()
button_7.pack()

gui_running = True
while gui_running: 
    plot_gui.update()

plot_gui.destroy()



