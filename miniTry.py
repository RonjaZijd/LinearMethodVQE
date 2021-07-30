
import pennylane as qml
from pennylane import numpy as np
import copy as cp
import TryOutLibrary as TL
import matplotlib.pyplot as plt
import pandas as pd

name_LM_file = "S001try2.csv"
name_opt_energ_file = "S001energ.csv"
name_opt_time_file = "S001time.csv"
name_opt_exec_file = "S001ex.csv"
name_opt_iter_file = "S001it.csv"

df_LM = pd.read_csv(name_LM_file)
df_en = pd.read_csv(name_opt_energ_file)
df_time = pd.read_csv(name_opt_time_file)
df_iter = pd.read_csv(name_opt_iter_file)
df_exec = pd.read_csv(name_opt_exec_file)

print(df_LM.to_string())
print()
print()
print(df_en.to_string())
print()
print()
print(df_time.to_string())
print()
print()
print(df_iter.to_string())
print()
print()
print(df_exec.to_string())




























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
