import pennylane as qml
from pennylane import numpy as np

Thets = np.random.normal(0, np.pi, (4,3))
dev = qml.device('default.qubit', wires=4)
#desired circuit: 
@qml.qnode(dev)
def circuit(params):
    qml.BasisState(np.array([1, 1, 0, 0], requires_grad=False), wires=[0,1,2,3])   ##let's try taking this one away and see what it does
    for i in range(3):
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])
    return qml.expval(qml.PauliZ(wires=0))

#grady = qml.grad(circuit)
#staty = grady(Thets)
#print(staty)
#print()
#print(staty.size)
#staty = np.reshape(staty, (staty.size)) 
#print(staty)

def creating_S(circuit, Thets):
    grady = qml.grad(circuit)
    staty = grady(Thets)
    staty = np.reshape(staty, (staty.size)) 
    S_matrix = np.zeros((len(staty), len(staty)))
    for n in range(len(staty)):
        for m in range(len(staty)):
            if n==m: 
                S_matrix[n][m] = 1
            if n>m: 
                S_matrix[n][m] = staty[n]*staty[m]
                S_matrix[m][n] = staty[m]*staty[n]

    return S_matrix

S = creating_S(circuit, Thets)
print(S)

print("Is S positive definite: ")
print(np.all(np.linalg.eigvals(S)>0))
print()