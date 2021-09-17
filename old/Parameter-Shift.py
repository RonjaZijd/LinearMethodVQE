import pennylane as qml
from pennylane import numpy as np

#set random seed
np.random.seed(42)

#create a decive to execute the circuit on
dev = qml.device("default.qubit", wires = 3)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)

    qml.broadcast(qml.CNOT, wires=[0,1,2], pattern="ring")

    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)

    qml.broadcast(qml.CNOT, wires=[0,1,2], pattern="ring")
    return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))

params = np.random.random([6])

print("Parameters: ", params)
print("Expectation value: ", circuit(params))

print(circuit.draw())

def parameter_shift_term(qnode, params, i):
    shifted = params.copy()
    shifted[i] += np.pi/2
    forward = qnode(shifted) #forward evaluation

    shifted[i] -= np.pi 
    backward = qnode(shifted) #backward evaluation

    return 0.5 * (forward - backward)

print(parameter_shift_term(circuit, params, 0))

#computing the gradient wrt to all paramters

def parameter_shift(qnode, params):
    gradients = np.zeros([len(params)])

    for i in range(len(params)):
        gradients[i] = parameter_shift_term(qnode, params, i)

    return gradients

print(parameter_shift(circuit, params))

#can also do this with PennyLane's built-in parameter shift

grad_function = qml.grad(circuit)
print(grad_function(params) [0])

dev2 = qml.device("default.qubit", wires = 4)

@qml.qnode(dev2, diff_method="parameter-shift", mutable=False)
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2, 3])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)) ####What does this @ sign mean

params = qml.init.strong_ent_layers_normal(n_wires=4, n_layers=15)
print(params.size)
print(circuit(params))

import timeit

reps = 3
num = 10
times = timeit.repeat("circuit(params)", globals=globals(), number=num, repeat=reps)
forward_time = min(times) / num

print(f"Forward pass (best of {reps}): {forward_time} sec per loop")