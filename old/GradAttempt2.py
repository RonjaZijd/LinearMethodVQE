import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuity(weights):
    for i in range(2):
        print(*weights[i])
        qml.RX(*weights[i], wires = i)
        qml.CNOT(wires=[0,1])
    return qml.probs(wires=0)  #when I had this qml.expval(qml.PauliX(wires=0)) as output it didn't work.. why?

def f(weights): #this is some random function which we define this way because we want to be able to calculate the Hessian of 'sth'
    print("These are my weights inside f: ")
    print(weights)
    quantum = circuity(weights)
    print("And this is quantum:")
    print(quantum)
    return quantum[0]

weights = np.random.random(size=[2, 2, 3], requires_grad=True)
weightsy = np.random.random(size=[2,1], requires_grad=True)

print(weightsy.shape)

print("These are weightsy's")
print(weightsy)

grad_fn = qml.grad(f)  ##standard grad calculation performed by pennylane 


grad = grad_fn(weightsy)
print("And the grad is: ")
print(np.round(grad, 5))