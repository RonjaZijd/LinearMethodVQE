
import pennylane as qml
from pennylane import numpy as np
import copy as cp
import TryOutLibrary as TL
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import tkinter as tk


I_mat = [[1,0], [0,1]]
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0], [0,-1]])
shapy = (4,3)
num_wires = 4
_aux_op = np.kron(X-1j*Y, np.eye(2**num_wires))
print(_aux_op)



































# ############################################ Functions  ###############################################################################
# ar_1 = [3,23,4,4]
# ar_2 = [1,2 ,4,5]


# def plot_scipy(): 
#     print("This is the plotting Scipy function.")
#     figs = plt.figure()
#     ploty = figs.add_subplot(111)
#     ploty.plot(ar_1, ar_2)
#     figs.show() 

# def plot_adam():
#     print("This is the plotting Adam function.")

# def plot_grad():
#     print("This is the plotting grad function.")

# def plot_lm():
#     print("This is the plotting lm function.")

# def plot_all():
#     print("This is the plotting all function.")
#     print("The reg chosen was: ", reg_entry.get())

# def finished():
#     global first_gui
#     #root0.destroy()
#     first_gui =False

# def finished2():
#     global second_gui
#     second_gui = False

# ############################################# Setting up initial GUI  #######################################################
# root0 = tk.Tk()
# root0.title("Very first GUI")

# tit_frame = tk.Frame(root0)
# tit_frame.pack()
# title_0 = tk.Label(tit_frame, text = "Plots", anchor = 'w', font=("Courier", 16)) 
# buttonny = tk.Button(tit_frame, text = "Done here. ", command=finished)
# title_0.pack()
# buttonny.pack()

# first_gui = True
# input_quest = True
# while first_gui:
#     root0.update()
#     #user_inp = input("Print 'exit' to exit this GUI")
#    # if first_gui==False :
#     #    root0.destroy()
#      #   input_quest = False
# root0.destroy()  
        
# ############################################ Setting up the GUI  #######################################################################

# root = tk.Tk()
# root.title("Extra things GUI")

# title_frame = tk.Frame(root)
# first_frame = tk.Frame(root)
# second_frame = tk.Frame(root)

# title_frame.pack(fill = 'x')
# first_frame.pack()
# second_frame.pack()

# ############################################### Title frame  #######################################################################
# #title items in the title_frame
# title_1 = tk.Label(title_frame, text = "Plots", anchor = 'w', font=("Courier", 16))   
# title_2 = tk.Label(title_frame, text = "This interface can be used to display various plots or data as wished.", anchor = 'w')          
# #packing the title frame
# title_1.pack()
# title_2.pack()



# ######################################     First frame      ################################################
# #buttons inside the first frame
# button_2 = tk.Button(first_frame, text = "Show the SciPy plot", command=plot_scipy)
# button_3 = tk.Button(first_frame, text = "Show the Adam plot", command=plot_adam)
# button_4 = tk.Button(first_frame, text = "Show the Grad plot", command=plot_grad)
# button_5 = tk.Button(first_frame, text = "Show the LM plot", command=plot_lm)
# button_6 = tk.Button(first_frame, text = "Show all plots together", command = plot_all)
# button_7 = tk.Button(first_frame, text = "Done here", command = finished2)
# #packing buttons into the first frame
# button_2.pack(side='left')
# button_3.pack(side = 'left')
# button_4.pack(side = 'left')
# button_5.pack(side = 'left')
# button_6.pack()
# button_7.pack()

# ######################################        Second frame        ##############################################

# reg_entry = tk.Entry(second_frame)
# k_max_entry = tk.Entry(second_frame)

# reg_entry.pack()
# k_max_entry.pack()
# second_gui=True
# plot_loop = True
# while plot_loop:
#     root.update()
#     if second_gui==False:
#         plot_loop = False
#         root.quit()


# #root.mainloop()#to keep the interface showing




































# #Energy values H-H-H-H:

# energ_array = np.array([11.1777, 10.42222, 9.6746, 9.1919, 9.16635, 9.10336, 9.55514, 9.29165, 9.17156, 9.174596, 9.08, 9.0797, 9.7186, 9.19707, 9.81206, 9.14539])
# iter_array = np.arange(0, 16, 1)
# print(len(energ_array))
# print(len(iter_array))
# plt.scatter(iter_array, energ_array)
# plt.plot(iter_array, energ_array)
# plt.xlabel('Iteration')
# plt.ylabel('Energy')
# plt.title('LMVQE on H-H-H-H')
# plt.show()



















































"""
# energ_ar = np.array([-3.19, -4.16, -4.20, -2.07, -4.27, -3.89, -4.19, -4.19, -4.35, -4.19, -4.18, -4.28,
#  -4.20, -4.21, -4.19, -4.37, -4.21, -4.38,-4.30,-3.16, -3.61, -4.25, -3.47, -3.06,
#  -3.93, -3.25, -3.87, -3.99, -4.35, -4.52, -4.53, -4.13, -2.05, -1.85, -2.74, -2.39,
#  -3.97, -3.46, -3.69, -3.81 ,-3.23, -3.97, -5.47, -4.87, -3.55, -4.22, -4.04, -4.06,
#  -2.16, -4.20, -3.21, -5.28 ,-4.98, -5.11, -4.15, -3.70, -4.17, -4.54, -3.29, -4.44,
#  -4.14, -3.38, -4.31, -4.49, -3.13, -4.25, -4.01, -4.58, -4.49, -4.29, -4.38, -4.03,
#  -4.36, -4.34, -3.98, -4.25, -3.66, -4.10, -4.24, -4.22, -2.33, -4.33, -4.30, -4.31,
#  -3.70, -4.19, -4.29, -4.31, -3.90, -2.38, -3.92, -4.45, -3.69, -4.60, -4.49, -2.78,
#  -5.43, -1.68, -4.32, -4.26, -5.08, -3.80, -4.64, -3.24, -4.34, -4.06, -4.02, -4.42,
#  -3.48, -2.72, -2.52, -4.22, -4.05, -4.22, -3.75, -3.53, -3.16, -3.66, -3.94, -4.35,
#  -3.79, -3.97, -4.42, -3.18, -4.00, -4.33, -4.12, -3.87, -4.16, -4.14, -3.20, -4.16,
#  -4.22, -3.69, -4.57, -4.40, -4.19, -4.29, -4.23, -4.25, -4.17, -4.52,-4.50, -4.58,
#  -4.33, -4.30, -4.53, -4.33, -2.71, -4.36, -4.39, -4.29, -4.16, -4.50, -4.69, -3.04,
#  -3.39, -3.89, -3.55, -4.78, -4.29, -4.62, -2.75, -4.17, -4.25, -4.05, -4.06, -4.78,
#  -4.24, -4.84, -4.70, -3.96, -4.16, -4.40, -4.66, -4.33, -4.12, -2.97, -4.73, -3.74,
#  -4.55, -4.27, -3.95, -4.73, -2.89, -4.28, -3.52, -4.40, -3.40,-4.71, -3.10, -4.10,
#  -4.41, -4.43, -4.03, -3.95, -4.22, -4.35, -4.32, -4.26, -3.47,-4.08, -4.65, -4.31,
#  -4.69, -2.89, -4.25, -4.39, -4.29, -4.53, -4.11, -4.18, -3.75, -4.18, -3.70, -3.93,
#  -4.56, -3.85, -3.42, -3.34, -3.33, -3.30, -4.34, -3.46, -3.39, -3.43, -3.30, -3.38,
#  -2.90, -3.05, -5.83, -3.45, -3.81, -3.33, -2.26, -4.11, -3.67, -2.18, -5.55, -3.41,
#  -4.15, -4.07, -4.04, -3.77, -3.64, -3.54, -4.21, -4.16, -3.87, -4.07])
energ_ar = np.array([-5.79, -6.091, -6.2049, -6.2168, -6.259])
k_arr = np.linspace(0,1, len(energ_ar))
plt.scatter(k_arr, energ_ar)
plt.plot(k_arr, energ_ar)
plt.xlabel('k-value')
plt.ylabel('Energy')
plt.title('First iteration of LiH')
plt.show()

"""








































# from scipy.interpolate import interp1d

# non_temp_k_ar = [ 1, 0.2, 0.05, 0.01, 0.5, 0.9, 0.8, 0.005, 0.4, 0.07, 0.6, 0.1]

# energ_array = [-0.64, -0.08, 0.45, -0.40, -0.36, -0.95, -0.53, -0.25, -0.41, 0.45, -0.38, -0.34]

# f = interp1d(non_temp_k_ar, energ_array)
# f2 = interp1d(non_temp_k_ar, energ_array, kind='cubic')

# inters_energs = []

# xnew = np.linspace(0.01, 1, num=41, endpoint=True)

# for x in xnew:
#     inters_energs = np.append(inters_energs, f2(x))

# print(xnew[np.argmin(inters_energs)])

# plt.plot(non_temp_k_ar, energ_array, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
# #plt.scatter(non_temp_k_ar, energ_array)
# plt.show()































# name_LM_file = "S001try2.csv"
# name_opt_energ_file = "S001energ.csv"
# name_opt_time_file = "S001time.csv"
# name_opt_exec_file = "S001ex.csv"
# name_opt_iter_file = "S001it.csv"

# df_LM = pd.read_csv(name_LM_file)
# df_en = pd.read_csv(name_opt_energ_file)
# df_time = pd.read_csv(name_opt_time_file)
# df_iter = pd.read_csv(name_opt_iter_file)
# df_exec = pd.read_csv(name_opt_exec_file)

# print(df_LM.to_string())
# print()
# print()
# print(df_en.to_string())
# print()
# print()
# print(df_time.to_string())
# print()
# print()
# print(df_iter.to_string())
# print()
# print()
# print(df_exec.to_string())




























# energy_lm = 8
# energy_adam = 71
# energy_grad = 234
# energy_scipy=22

# name_opt_energ_file = "Energy_file_opty.csv"

# energy_dic = {"LM":[energy_lm], "Adam": [energy_adam], "Grad": [energy_grad], "SciPy": [energy_scipy]}
# energy_dic2 = {"LM":energy_lm, "Adam": energy_adam, "Grad": energy_grad, "SciPy": energy_scipy}
# df_temp_en_opt = pd.DataFrame(energy_dic)
# #df_temp_en_opt.to_csv(name_opt_energ_file)

# print(df_temp_en_opt.to_string())
# df_en_opt = pd.read_csv(name_opt_energ_file)
# df_en_opt = df_en_opt.append(energy_dic2, ignore_index=True, sort=False)
# #df_en_opt = pd.concat([df_en_opt, df_temp_en_opt], ignore_index=True)
# df_en_opt.drop(columns=df_en_opt.columns[0], axis=1, inplace=True)
# df_en_opt.to_csv(name_opt_energ_file)

# print(df_en_opt.to_string())

























# a = [1, 7, 2]

# myvar = pd.Series(a, index=["x", "y", "z"])

# print(myvar["x"])

# calories = {"day1": 420, "day2": 380, "day3": 390}

# myvar = pd.Series(calories)

# print(myvar)

# data = {
#   "calories": [420, 380, 390],
#   "duration": [50, 40, 45]
# }

# myvar = pd.DataFrame(data)

# print()
# print(myvar)
# print()

# name_csv_file = 'PandaTry.csv'

# df = pd.read_csv(name_csv_file)
# print(df)

# Col = [0.9, 0.3, 0.77, 0.1]

# df_temp = pd.DataFrame(Col, columns=["Temporary2"])

# print(df_temp)
# print()

# df = pd.concat([df, df_temp], axis=1)
# print(df)

# df.to_csv(name_csv_file)






























# aa = 4
# bb = 6
# TL.print_everything_func(aa,bb)


# th1 = -1.87990278
# th2 = -4.16093261
# ans = (th1*th2/2)*np.sin(th1/2)*np.cos(th1/2)
# print(ans)

# a = np.array([4, 3, 2])
# b = cp.copy(a) 
# b[0] = 2
# #print(a)
# #print(b)

# def different_regularization(Array, tolerance):

#     #Input array is the array that contains the energies of the different regularizations
#     #Output is the index which I want to choose
#     #If there is a big energy difference, choose the one with the lowest energy
#     #If there is a small energy difference <0.001 between the lowest and some others
#     #identify which others also have this energy and then choose the one with 
#     #the lowest regularization 
#     lowest_energies = []

#     sorted_array = np.argsort(Array)
#     print(sorted_array)
#     for i in range(len(sorted_array)):
#         print("Comparing these numbers for ", i)
#         print(np.abs(Array[sorted_array[i]]-Array[sorted_array[0]]))
#         if np.abs(Array[sorted_array[i]]-Array[sorted_array[0]])<tolerance:
#             lowest_energies = np.append(lowest_energies, sorted_array[i])
#     print(lowest_energies)
#     return int(np.amax(lowest_energies))

# print("I got here!")
# ar = np.array([-1.016, -1.019, -1.01003, -1.0053])
# print(different_regularization(ar, 0.01))

# print("And this is the energy which I chose: ")
# print(ar[different_regularization(ar, 0.01)])
















# Thets = np.random.normal(0, np.pi, (4,3))

# Hamilt_written_outt = -0.24274280513140462*qml.PauliZ(wires=2) + -0.24274280513140462*qml.PauliZ(wires=3) +  0.1777128746513994*qml.PauliZ(wires=1)+0.17771287465139946*qml.PauliZ(wires=0)+0.12293305056183798*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))+0.12293305056183798*(qml.PauliZ(wires=1)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=0)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=1)@qml.PauliZ(wires=2)) +0.17059738328801052*(qml.PauliZ(wires=0)@qml.PauliZ(wires=1))+0.17627640804319591*(qml.PauliZ(wires=2)@qml.PauliZ(wires=3))+-0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliY(wires=1)@qml.PauliX(wires=2)@qml.PauliX(wires=3))+-0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliX(wires=1)@qml.PauliY(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliX(wires=1)@qml.PauliX(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliY(wires=1)@qml.PauliY(wires=2)@qml.PauliX(wires=3))


# def circuit(params, wires):
#     qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
#     for i in wires:
#         qml.Rot(params[i][2], params[i][1], params[i][0], wires=i)
#     qml.CNOT(wires=[2, 3])
#     qml.CNOT(wires=[2, 0])
#     qml.CNOT(wires=[3, 1])

# def energy_calc(circuit, Hamilt_written_out, device, Thets):
#     costy = qml.ExpvalCost(circuit, Hamilt_written_out, device)  ###for now we're simply going to do it with the specific case circuit
#     energ = costy(Thets)
#     return energ

# dev2 = qml.device('default.qubit', wires=4)

# energy_array_adam = []
# n_array_adam = []
# cost_function_adam = qml.ExpvalCost(circuit, Hamilt_written_outt, dev2)
# opt = qml.AdamOptimizer(stepsize=0.1)
# for n in range(300):
#     Thets, Prev_energ = opt.step_and_cost(cost_function_adam, Thets)
#     if n>2:
#         if (np.abs(Prev_energ-energy_array_adam[-1])<0.000001) & (n>1):
#             print()
#             print("Reached convergence!")
#             break
#     energy_array_adam = np.append(energy_array_adam, Prev_energ)
#     n_array_adam = np.append(n_array_adam, n)

# print(dev2.num_executions)

# plt.plot(n_array_adam, energy_array_adam)
# plt.show()
