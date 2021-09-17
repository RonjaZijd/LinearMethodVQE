import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def circuit(weights):
    qml.PauliX(wires=0) ##to put it into the basisstate 1100 (as the basisstate function is giving me trouble)
    qml.PauliX(wires=1)
    for i in range(4):
        #qml.RX(*weights[i], wires = i)
        qml.Rot(*weights[i], wires = i)
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[2,0])
    qml.CNOT(wires=[3,1])
    return qml.probs(wires=0)

def func(weights):
    quant = circuit(weights)
    return quant[0]
 
#weights = np.random.random(size=[4,3], requires_grad=True)
weights2 = np.random.normal(0, np.pi, size=[4,3])


grad_fn = qml.grad(func)  ##standard grad calculation performed by pennylane 

grad = grad_fn(weights2)
print(np.round(grad, 5))

