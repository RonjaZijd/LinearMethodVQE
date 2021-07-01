import pennylane as qml
from pennylane import numpy as np

U_gates = np.array(['RX', 'RY', 'RZ', 'RX', 'RX', 'RY', 'RX']) ##the U gates in order
Thets = np.array([0.34, 0.21, 0.78, 0.45, 0.99, 0.21, 0.11])
U_gener = np.array(['CNOT', 'CY', 'CZ', 'CNOT', 'CNOT', 'CY', 'CNOT'])

def gate_creator(string, thet, wir): ##a function to create the gates
    if string=='RX':
        return qml.RX(thet, wires=wir)
    if string=='RY':
        return qml.RY(thet, wires=wir)
    if string=='RZ':
        return qml.RZ(thet, wires=wir)
    if string=='CNOT' or string=='X':
        return qml.CNOT(wires=[0,wir])
    if string=='CY' or string=='Y':
        return qml.CY(wires=[0,wir])
    if string=='CZ' or string=='Z':
        return qml.CZ(wires=[0,wir])

def make_into_list(string):
    print('test')
    l = list(string)
    return l

def c_notting_hamil(input_array):
    i=0
    #input_array = ['Z1', 'X2', 'Y3']
    print("Test1")
    while i<len(input_array):
        print("Test2")
        as_characters = list(input_array[i])
        #as_characters = make_into_list(input_array[i])
        print("Test3")
        gate = as_characters[0]
        wire = int(as_characters[1])
        print("Test4")
        gate_creator(gate, 0, wire)
        i=i+1

dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def try_circ(inp_array):
    c_notting_hamil(inp_array.numpy())
    return qml.expval(qml.PauliZ(wires=0))

Inp = ['Z1', 'X2', 'Y2']

c_notting_hamil(Inp)

try_circ(Inp)
print(try_circ.draw())