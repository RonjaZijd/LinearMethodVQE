import pennylane as qml
from pennylane import qaoa
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
graph = nx.Graph(edges)

#nx.draw(graph, with_labels=True)
#plt.show()

cost_h, mixer_h = qaoa.min_vertex_cover(graph, constrained=False)

print("Cost Hamiltonian", cost_h)
print("Mixer Hamiltonian", mixer_h)

def qaoa_layer(gamma, alpha):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(alpha, mixer_h)

wires = range(4)
depth = 2

def circuit(params, **kwargs):
    for w in wires:
        qml.Hadamard(wires=w)
    qml.layer(qaoa_layer, depth, params[0], params[1]) #recall this is how we use the layers command



dev = qml.device("default.qubit", wires=wires)
cost_function = qml.ExpvalCost(circuit, cost_h, dev)

@qml.qnode(dev)
def circuit2(params, **kwargs):
    for w in wires:
        qml.Hadamard(wires=w)
    qml.layer(qaoa_layer, depth, params[0], params[1]) #recall this is how we use the layers command
    return qml.expval(qml.PauliX(0))

optimizer = qml.GradientDescentOptimizer()
steps = 70
params = [[0.5, 0.5], [0.5, 0.5]]

#I can't draw the circuit, even though I feel like I should be able to
circuit2([[0.5, 0.5], [0.5, 0.5]])
#print(circuit2.draw())

for i in range(steps):
    params = optimizer.step(cost_function, params)

print("Optimal Parameters")
print(params)

#now displaying the output
@qml.qnode(dev)
def probability_circuit(gamma, alpha):
    circuit([gamma, alpha])
    return qml.probs(wires=wires)


probs = probability_circuit(params[0], params[1])

plt.style.use("seaborn")
plt.bar(range(2 ** len(wires)), probs)
plt.show()

#####last bit is about how to customize this even further.