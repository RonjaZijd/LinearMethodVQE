import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
from pennylane.templates.subroutines import UCCSD
from functools import partial

name = "h2"
geometry = "h2o.xyz" #I don't have this file yet

symbols, coordinates = qchem.read_structure(geometry)
#H, qubits = qchem.molecular_hamiltonian(symbols, coordinates, mapping="jordan_wigner") once again doesn't work

qubits = 4
Z2 = qml.PauliZ(wires=2)
Z3 = qml.PauliZ(wires=3)
#H = (-0.24274280513140462) [qml.PauliZ2] + (-0.24274280513140462) [qml.PauliZ3] + (-0.04207897647782276) [qml.Identity] + (0.1777128746513994) [qml.PauliZ1] + (0.17771287465139946) [qml.PauliZ0] + (0.12293305056183798) [qml.PauliZ0, qml.PauliZ2] + (0.12293305056183798) [qml.PauliZ1, qml.PauliZ3] + (0.1676831945771896) [qml.PauliZ0, qml.PauliZ3] + (0.1676831945771896) [qml.PauliZ1, qml.PauliZ2] + (0.17059738328801052) [qml.PauilZ0, qml.PauliZ1] + (0.17627640804319591) [qml.PauliZ2, qml.PauliZ3] + (-0.04475014401535161) [qml.PauliY0, qml.PauliY1, qml.PauliX2, qml.PauliX3] + (-0.04475014401535161) [qml.PauliX0, qml.PauliX1, qml.PauliY2, qml.PauliY3] + (0.04475014401535161) [qml.PauilY0, qml.PauliX1, qml.PauliX2, qml.PauliY3] + (0.04475014401535161) [qml.PauliX0, qml.PauliY1, qml.PauliY2, qml.PauliX3] 
H = (-0.24274280513140462) [Z2] + (-0.24274280513140462) [Z3]
print("Number of qubits = ", qubits)
print("Hamiltonian is ", H)

electrons = 2
S2 = qchem.spin2(electrons, qubits, mapping="jordan_wigner")
print(S2)

dev = qml.device("default.qubit", wires=qubits) #defining the device

singles, doubles = qchem.excitations(electrons, qubits, delta_sz=0)
print(singles)
print(doubles)

s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
print(s_wires)
print(d_wires)

hf_state = qchem.hf_state(electrons, qubits)
print(hf_state)

ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)
cost_fn = qml.ExpvalCost(ansatz, H, dev) #H is at present not properly defined.

#and then optimizing this, however can't do this, because of the missing plugins