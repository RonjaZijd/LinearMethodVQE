"""
last edited: 19/07
purpose: 
        program uses the hamiltonian and the circuit of the standard VQE problem, to then change the one or two of 
        the parameters at a time and plot the energy landscape with the variation. 

        It relies on: 
            -self input hamiltonian
            -pennylane circuit
            -a linspace which changes the parameters
            -plotting

"""
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import copy as cp


Hamilt_written_outt = -0.24274280513140462*qml.PauliZ(wires=2) + -0.24274280513140462*qml.PauliZ(wires=3) +  0.1777128746513994*qml.PauliZ(wires=1)+0.17771287465139946*qml.PauliZ(wires=0)+0.12293305056183798*(qml.PauliZ(wires=0) @ qml.PauliZ(wires=2))+0.12293305056183798*(qml.PauliZ(wires=1)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=0)@qml.PauliZ(wires=3))+0.1676831945771896*(qml.PauliZ(wires=1)@qml.PauliZ(wires=2)) +0.17059738328801052*(qml.PauliZ(wires=0)@qml.PauliZ(wires=1))+0.17627640804319591*(qml.PauliZ(wires=2)@qml.PauliZ(wires=3))+-0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliY(wires=1)@qml.PauliX(wires=2)@qml.PauliX(wires=3))+-0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliX(wires=1)@qml.PauliY(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliY(wires=0)@qml.PauliX(wires=1)@qml.PauliX(wires=2)@qml.PauliY(wires=3))+0.04475014401535161*(qml.PauliX(wires=0)@qml.PauliY(wires=1)@qml.PauliY(wires=2)@qml.PauliX(wires=3))
no_of_wires =5
dev = qml.device('default.qubit', wires=4)

def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=wires)   ##let's try taking this one away and see what it does
    for i in wires:
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

def energy_calc(circuit, Hamilt_written_out, device, Thets):
    costy = qml.ExpvalCost(circuit, Hamilt_written_out, device)  ###for now we're simply going to do it with the specific case circuit
    energ = costy(Thets)
    return energ

Thets_basic = np.array([[3.34, 5.14, 2.17], [3.05, 1.55, 0.14], [2.09, 6.09, 2.46], [6.21, 0.25, 4.68]])
change_array = np.linspace(-3, 3, 15)
#to record everything in:
energy_array = []

for i in range(len(change_array)):
    param_chosen1 = 2 #first parameter to change
    param_chosen_further1 = 1
    Thets_new = cp.copy(Thets_basic)
    Thets_new[param_chosen1][param_chosen_further1] = Thets_basic[param_chosen1][param_chosen_further1] + change_array[i]
    for j in range(len(change_array)):
        param_chosen2 = 3 #second parameter change (counting in the python way)
        param_chosen_further2 = 1
        Thets_new[param_chosen2][param_chosen_further2] = Thets_basic[param_chosen2][param_chosen_further2] + change_array[j]
        energy = energy_calc(circuit, Hamilt_written_outt, dev, Thets_new)
        energy_array = np.append(energy_array, energy)

energy_array = np.reshape(energy_array, (len(change_array), len(change_array)))
print(energy_array)

i_array2 = np.linspace(Thets_basic[param_chosen1][param_chosen_further1]-3, Thets_basic[param_chosen1][param_chosen_further1]+3, 15)
j_array2 = np.linspace(Thets_basic[param_chosen2][param_chosen_further2]-3, Thets_basic[param_chosen2][param_chosen_further2]+3, 15)

I, J = np.meshgrid(i_array2, j_array2)

#plotting it: 
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(I, J, energy_array)
ax.set_xlabel(f'Element {param_chosen1}, {param_chosen_further1}')
ax.set_ylabel(f'Element {param_chosen2}, {param_chosen_further2}')
ax.set_zlabel('Energy')


#plt.plot_surface(I, J, energy_array)
#plt.xlabel(f'Element {param_chosen1}, {param_chosen_further1}')
#plt.ylabel(f'Element {param_chosen2}, {param_chosen_further2}')
#plt.title('Energy')
plt.show()















"""

for i in range(len(change_array)):
    parameter_chosen = 3
    parameter_chosen_further = 2
    Thets_new1 = np.zeros((Thets_basic.shape))
    Thets_new1 = cp.copy(Thets_basic)
    Thets_new1[parameter_chosen][parameter_chosen_further] = Thets_basic[parameter_chosen][parameter_chosen_further] + change_array[i]


    energy1 = energy_calc(circuit, Hamilt_written_outt, dev, Thets_new1)
    energy_array1 = np.append(energy_array, energy)




    parameter_chosen2 = 0
    parameter_chosen_further2 = 0
   
    energy_array = np.append(energy_array, energy)
    i_array1 = np.append(i_array,  Thets_new1[parameter_chosen][parameter_chosen_further])










"""
