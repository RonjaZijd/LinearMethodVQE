import pennylane as qml
from pennylane import numpy as np
import LMLibrary2 as LM
import matplotlib.pyplot as plt
import scipy as sp
import tkinter as tk
import copy as cp
import time

np.set_printoptions(suppress=True, precision=3, formatter={'float_kind':'{:0.2f}'.format})

###########################################  Hamiltonian input #####################################################

###  Variational Circuit: ###
U_gates = np.array([['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ'], ['RZ', 'RY', 'RZ']]) ##the U gates in order
entangle_gates = np.array([[2,3], [2,0], [3,1]]) ###the entangled gates at the end
shape = (4,3)


#Hamiltonian
H_VQE_coeffs = [-0.24274280513140462,-0.24274280513140462,0.1777128746513994,0.17771287465139946,0.12293305056183798,0.12293305056183798,0.1676831945771896,0.1676831945771896,0.17059738328801052,0.17627640804319591,-0.04475014401535161,-0.04475014401535161,0.04475014401535161,0.04475014401535161]
H_VQE_gates = [['Z3'],['Z4'],['Z2'],['Z1'],['Z1', 'Z3'],['Z2','Z4'],['Z1','Z4'],['Z2','Z3'],['Z1','Z2'],['Z3', 'Z4'], ['Y1', 'Y2', 'X3', 'X4'],['X1', 'X2', 'Y3', 'Y4'], ['Y1', 'X2', 'X3', 'Y4'], ['X1', 'Y2', 'Y3', 'X4'] ]
Hamilt_written_out = -0.24274280513140462*qml.PauliZ(wires=2) + -0.24274280513140462*qml.PauliZ(wires=3) +  0.1777128746513994*qml.PauliZ(wires=1)+0.17771287465139946*qml.PauliZ(wires=0)+0.12293305056183798*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))+0.12293305056183798*(qml.PauliZ(wires=1)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=0)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=1)@qml.PauliZ(wires=2)) +0.17059738328801052*(qml.PauliZ(wires=0)@qml.PauliZ(wires=1))+0.17627640804319591*(qml.PauliZ(wires=2)@qml.PauliZ(wires=3))+-0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliY(wires=1)@qml.PauliX(wires=2)@qml.PauliX(wires=3))+-0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliX(wires=1)@qml.PauliY(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliX(wires=1)@qml.PauliX(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliY(wires=1)@qml.PauliY(wires=2)@qml.PauliX(wires=3))

### Basic Circuit in Pennylane form ###
def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
    for i in wires:
        qml.Rot(params[i][2], params[i][1], params[i][0], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

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

    finish_but = tk.Button(finished_frame, text = "Filled everything in!", command = finished)
    finish_but.pack()

    gui_running = True
    while gui_running:
        inp_gui.update()

    try:
        reg = int(reg_entry.get())       ### Currently no input validation!!
        min_k = int(min_k_entry.get())
        max_k = int(max_k_entry.get())
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

    return reg, min_k, max_k, n_of_k, n_of_its, seed_n

reg, min_k, max_k, n_of_k, n_of_its, seed_n = run_input_gui()

###########################################  Running the program ###################################################

### Initializing ###
Thets = np.random.normal(0, np.pi, shape)
Thets_start = cp.copy(Thets)
no_of_wires  = len(U_gates)+1 #plus 1 because of the ancillary bit
n_of_qubits = len(U_gates)
matrix_length = Thets.size

### Decives ###
dev_grad = qml.device('default.qubit', wires=n_of_qubits)
dev_adam = qml.device('default.qubit', wires=n_of_qubits)
dev_scipy = qml.device('default.qubit', wires=n_of_qubits)
dev_lm = qml.device('default.qubit', wires=n_of_qubits)
dev_extra = qml.device('default.qubit', wires=n_of_qubits)

### Functions needed in all ###
def energy_grad(Thets):
    global circuit
    global Hamilt_written_out
    global dev_extra
    Thets = np.reshape(Thets, shape)
    energy_func = qml.ExpvalCost(circuit, Hamilt_written_out, dev_extra)
    grad_func = qml.grad(energy_func)
    E_gra = grad_func(Thets)
    return E_gra.flatten()
def energy_calc(circuit, Hamilt_written_out, device, Thets):
    costy = qml.ExpvalCost(circuit, Hamilt_written_out, device)  ###for now we're simply going to do it with the specific case circuit
    energ = costy(Thets)
    return energ

### SCIPY  ###
t_0_scipy = time.process_time()  ##put SciPy into a function later
energy_array_scip = []
n_array_scipy = []
def cost_func_scipy(Thets):
    global energy_array_scip #all global, needed to track the progress of SciPy function
    global circuit
    global Hamilt_written_outt
    global dev2
    Thets = np.reshape(Thets, shape)
    cost_function_scipy = qml.ExpvalCost(circuit, Hamilt_written_out, dev_scipy)
    energy_array_scip = np.append(energy_array_scip, energy_calc(circuit, Hamilt_written_out, dev_extra, Thets))
    return cost_function_scipy(Thets)
Thets = np.reshape(Thets, (matrix_length,1))
res = sp.optimize.minimize(cost_func_scipy, Thets, method = 'BFGS', jac=energy_grad) #results of scipy
iterations_scipy = res.get('nfev')
energy_scipy = res.get('fun')
for n in range(iterations_scipy):
    n_array_scipy = np.append(n_array_scipy, n)
t_1_scipy = time.process_time()

### ADAM ###
def adam_optimizer(circuit, hamiltonian, device, thets, max_iters, tolerance):
    t_0 = time.process_time()
    energy_array = []
    n_array = []

    cost_function = qml.ExpvalCost(circuit, hamiltonian, device)  #functions on which optimization is performed
    opt = qml.AdamOptimizer(stepsize=0.1)

    for n in range(max_iters):
        thets, prev_energ = opt.step_and_cost(cost_function, thets)
        if n>2:
            if (np.abs(prev_energ-energy_array[-1])<tolerance) & (n>1):
                print("Reached convergence!")
                break
        energy_array = np.append(energy_array, prev_energ)
        n_array = np.append(n_array, n)

    t_1 = time.process_time()
    time_taken = t_1 - t_0
    return time_taken, energy_array, n_array
time_adam, energy_array_adam, n_array_adam = adam_optimizer(circuit, Hamilt_written_out, dev_adam, Thets_start, 300, 0.000001)

### GRAD ###
def grad_optimizer(circuit, hamiltonian, device, thets, max_iters, tolerance):
    t_0 = time.process_time()
    energy_array = []
    n_array = []

    cost_function = qml.ExpvalCost(circuit, hamiltonian, device)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    for n in range(max_iters):
        thets, prev_energ = opt.step_and_cost(cost_function, thets)
        if n>2:
            if (np.abs(prev_energ-energy_array[-1])<tolerance) & (n>1):
                print("Reached convergence!")
                break
        energy_array = np.append(energy_array, prev_energ)
        n_array = np.append(n_array, n)

    t_1 = time.process_time()  
    time_taken = t_1 - t_0
    
    return time_taken, energy_array, n_array

time_grad, energy_array_grad, n_array_grad = grad_optimizer(circuit, Hamilt_written_out, dev_grad, Thets_start, 700, 0.000001)

### LM ###
t_0_lm = time.process_time()
n_array_LM = []
energy_array_LM = []
Thets = Thets_start
eee= LM.energy_calc(circuit, Hamilt_written_out, dev_lm, Thets)
n_array_LM = np.append(n_array_LM, 0)
energy_array_LM = np.append(energy_array_LM, eee)

for n in range(n_of_its) :
    H = LM.H_Matrix_final_calc(U_gates, Thets, H_VQE_gates, H_VQE_coeffs, entangle_gates)
    S = LM.S_Matrix_final_calc_newy(U_gates, Thets)
    S_tilde = LM.S_tilde_matrix(S, reg)
    temp_thets_ar = []
    temp_energ_ar = []
    non_temp_k_ar = np.linspace(min_k,  max_k, n_of_k, endpoint=True) ##disabling the tail
    
    for k in non_temp_k_ar: 
        H_tilde = LM.H_tilde_matrix(H, eee, LM.E_grad(Thets, Hamilt_written_out, circuit, dev_lm), k)
        try:
            update = LM.smallest_real_w_norm_optimiz(H_tilde, S_tilde)
            Thets_temp = LM.new_thetsy(update, Thets)
            Energ_temp = LM.energy_calc(circuit, Hamilt_written_out, dev_lm, Thets_temp)
            temp_thets_ar = np.append(temp_thets_ar, Thets_temp)
            temp_energ_ar = np.append(temp_energ_ar, Energ_temp)
        except:
            print("Could not converge")
            non_temp_k_ar = np.delete(non_temp_k_ar, np.where(non_temp_k_ar==k))
    temp_thets_ar = np.reshape(temp_thets_ar, (len(non_temp_k_ar), Thets.size))
    arg_chosen = np.argmin(temp_energ_ar)
    Thets = np.reshape(temp_thets_ar[arg_chosen], Thets.shape) ##choose the new theta's of the lowest energy
    eee = temp_energ_ar[arg_chosen] ###pick the lowest energy. 
    energy_array_LM = np.append(energy_array_LM, eee)
    n_array_LM = np.append(n_array_LM, n+1)
    
    print("---------------------------------------------------------------------------------------------------") #printing things to see what the program is doing
    print("Iteration: ", n)
    print("Energy chosen after update step: ", eee)

############################################ Plotting functions ########################################################

def plot_lm():
    fig = plt.figure()
    ploty = fig.add_subplot(111)
    ploty.plot(n_array_LM, energy_array_LM)
    ploty.set(xlabel='Iterations', ylabel = 'Energy')
    fig.show()

def plot_scipy():
    fig = plt.figure()
    ploty = fig.add_subplot(111)
    ploty.plot(n_array_scipy, energy_array_scip)
    ploty.set(xlabel='Iterations', ylabel = 'Energy')
    fig.show()

def plot_adam():
    fig = plt.figure()
    ploty = fig.add_subplot(111)
    ploty.plot(n_array_adam, energy_array_adam)
    ploty.set(xlabel='Iterations', ylabel = 'Energy')
    fig.show()

def plot_grad():
    fig = plt.figure()
    ploty = fig.add_subplot(111)
    ploty.plot(n_array_grad, energy_array_grad)
    ploty.set(xlabel='Iterations', ylabel = 'Energy')
    fig.show()

def plot_all():
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(n_array_LM, energy_array_LM)
    ax[0,0].set_title(f'Linear Method w/ final value: {eee}')
    ax[0,1].plot(n_array_adam, energy_array_adam)
    ax[0,1].set_title('Adam')
    ax[1,0].plot(n_array_grad, energy_array_grad)
    ax[1,0].set_title('Grad')
    ax[1,1].plot(n_array_scipy, energy_array_scip)
    ax[1,1].set_title('Scipy')

    for axi in ax.flat:
        axi.set(xlabel='Iterations', ylabel='Energy')

    plt.show()


############################################# GUI for plotting results ###############################################


plot_gui = tk.Tk()
plot_gui.title("GUI to plot the results")

title_frame2 = tk.Frame(plot_gui)
first_frame = tk.Frame(plot_gui)

title_frame2.pack(fill = 'x')
first_frame.pack()

title_1 = tk.Label(title_frame2, text = "Plots", anchor = 'w', font=("Courier", 16))   
title_1.pack()
print("Got to this poinT!")
button_2 = tk.Button(first_frame, text = "Show the SciPy plot", command=plot_scipy)
button_3 = tk.Button(first_frame, text = "Show the Adam plot", command=plot_adam)
button_4 = tk.Button(first_frame, text = "Show the Grad plot", command=plot_grad)
button_5 = tk.Button(first_frame, text = "Show the LM plot", command=plot_lm)
button_6 = tk.Button(first_frame, text = "Show all plots together", command = plot_all)
button_7 = tk.Button(first_frame, text = "Done here", command = finished)
#packing buttons into the first frame
button_2.pack(side='left')
button_3.pack(side = 'left')
button_4.pack(side = 'left')
button_5.pack(side = 'left')
button_6.pack()
button_7.pack()

gui_running = True
while gui_running: 
    plot_gui.update()

plot_gui.destroy()



