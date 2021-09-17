#From hamiltonians to circuits:

import pennylane as qml

H = qml.Hamiltonian(
    [1,1,0.5],
    [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
)

print(H)

dev= qml.device('default.qubit', wires=2)
t=1
n=2

@qml.qnode(dev)
def circuit():
    qml.templates.ApproxTimeEvolution(H,t,n)
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

circuit()
#print(circuit.draw())

#layering circuits can be useful so that you can repeat something multiple times.

def circ(theta):
    qml.RX(theta, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0,1])

@qml.qnode(dev)
def circuit(param):
    circ(param)
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

circuit(0.5)
#print(circuit.draw())

#now to ahve it repeat 3 times:

@qml.qnode(dev)
def circuit(params, **kwargs):
    qml.layer(circ, 3, params)
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

circuit([0.3, 0.4, 0.5])
print(circuit.draw())

########## NOW UNTO THE ACTUAL QAOA





