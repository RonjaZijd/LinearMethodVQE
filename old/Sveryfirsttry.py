import pennylane as qml
from pennylane import qchem
import numpy as np

#take the variationally prepared wavefunction from the pennylane demo

qubits = 4
dev = qml.device('default.qubit', qubits)

@qml.qnode(dev)
def circ(params):
    qml.PauliX(wires=0) ##to put it into the basisstate 1100 (as the basisstate function is giving me trouble)
    qml.PauliX(wires=1)
    for i in range(4):
        print(*params[i])
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[0,2])
    qml.CNOT(wires=[1,3])
    qml.CNOT(wires=[2,3])
    return qml.probs(wires=0)

def f(parameters):
    quant = circ(params) ##multiple values are coming out because of the qml.probs and we're selecting the first one. No fricking clue why. 
    print(quant)
    return quant[0]

params = np.random.normal(0, np.pi, (qubits, 3))
print("This is how the parameters look: ")
print(params)

grad_f = qml.grad(f)
graa = grad_f(params) 
print(graa) ##don't understand how we are


#circ(params)  ###  just used to check the circuit
#print(circ.draw())






#In this circuit we technically have 16 parameters, how do I keep track of 