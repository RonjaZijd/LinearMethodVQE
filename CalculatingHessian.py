import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(weights):
    for i in range(2):
        qml.Rot(*weights[i,0], wires =0)
        qml.Rot(*weights[i, 1], wires = 1)
        qml.CNOT(wires=[0,1])
    return qml.probs(wires=0)

@qml.qnode(dev)
def circuity(weights):
    for i in range(2):
        print(i)
        qml.RX(weights[i], wires = 1)
        qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliX(wires=0))

def f(weights): #this is some random function which we define this way because we want to be able to calculate the Hessian of 'sth'
    #quantum = circuit(np.sin(weights))
    quantum = circuit(weights)
   
    print("And this is quantum:")
    print(quantum)
    #np.sum(np.abs(quantum-x) / np.cos(x))
    return quantum[0]

weights = np.random.random(size=[2, 2, 3], requires_grad=True)




circuit(weights)
print(circuit.draw())

print("These are the weights.")
print(weights)
x = np.array([0.54, 0.1], requires_grad=False)
grad_fn = qml.grad(f)  ##standard grad calculation performed by pennylane 

grad = grad_fn(weights)
print("And the grad is: ")
print(np.round(grad, 5))

#hess_fn = qml.jacobian(grad_fn)
#H = hess_fn(weights)
#H.shape
#print(H.shape)
#print(H)
