#fitting the parity function:


##THESE ARE ALL MACHINE LEARNING THINGS WHICH IM NOT THAT INTERESTED IN



import pennylane as qml
from pennylane import numpy as np 
from pennylane.optimize import NesterovMomentumOptimizer

dev = qml.device("default.qubit", wires=4)

#our circuit layer consists of arbitrary rotation on every qubit, as well as CNOTS that entangle each qubit with its neighbour

def layer(W):

    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])

def statepreparation(x):
    qml.BasisState(x, wires=[0,1,2,3])

@qml.qnode(dev)
def circuit(weights, x):
    statepreparation(x)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))

#keyword arguments of a quantum node are considered fixed when calculating a gradient, they are never trained

#adding 'classical' bias parameter:
def variational_classifier(var,x):
    weights=var[0]
    bias=var[1]
    return circuit(weights,x) + bias

def square_loss(labels, predictions):
    loss=0
    for l, p in zip(labels, predictions):
        loss = loss+(l-p)**2

    loss = loss / len(labels)
    return loss

def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss

def cost(var, X, Y):
    predictions = [variational_classifier(var, x) for x in X]
    return square_loss(Y, predictions)

data = np.loadtxt("variational_classifer/data/parity.txt")
X = np.array(data[:, :-1], requires_grad=False)
Y = np.array(data[:, -1], requires_grad=False)
Y = Y * 2 - np.ones(len(Y))  # shift label from {0, 1} to {-1, 1}

for i in range(5):
    print("X = {}, Y = {: d}".format(X[i], int(Y[i])))

print("...")

np.random.seed(0)
num_qubits = 4
num_layers = 2
var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)

print(var_init)

opt = NesterovMomentumOptimizer(0.5)
batch_size = 5

var = var_init
for it in range(25):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, len(X), (batch_size,))
    X_batch = X[batch_index]
    Y_batch = Y[batch_index]
    var = opt.step(lambda v: cost(v, X_batch, Y_batch), var)

    # Compute accuracy
    predictions = [np.sign(variational_classifier(var, x)) for x in X]
    acc = accuracy(Y, predictions)

    print(
        "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
            it + 1, cost(var, X, Y), acc
        )
    )
