import pennylane as qml
from pennylane import numpy as np

dev = qml.device('default.mixed', wires=2)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

circuit()
print(circuit.draw())
print(f"QNode output = {circuit():.4f}")
print(f"Output state is = \n{np.real(dev.state)}")

@qml.qnode(dev)
def bitflip_circuit(p):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1])
    qml.BitFlip(p, wires=0)
    qml.BitFlip(p, wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


ps = [0.001, 0.01, 0.1, 0.2]
for p in ps:
    print(f"QNode output for bit flip probability {p} is {bitflip_circuit(p):.4f}")

print(f"Output state for bit flip probability {p} is \n{np.real(dev.state)}")    

###NOW examplifying a different built-in noisy channel
@qml.qnode(dev)
def depolarizing_circuit(p):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.DepolarizingChannel(p, wires=0)
    qml.DepolarizingChannel(p, wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


ps = [0.001, 0.01, 0.1, 0.2]
for p in ps:
    print(f"QNode output for depolarizing probability {p} is {depolarizing_circuit(p):.4f}")

##now attempting to mimic an experimentally obtained value:
ev = np.tensor([0.7781], requires_grad=False)  # observed expectation value

def sigmoid(x):
    return 1/(1+np.exp(-x))

@qml.qnode(dev)
def damping_circuit(x): ##the circuit which is believed to be the cause of the loss
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.AmplitudeDamping(sigmoid(x), wires=0)  # p = sigmoid(x)
    qml.AmplitudeDamping(sigmoid(x), wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

def cost(x, target):
    return (damping_circuit(x) - target[0])**2

opt = qml.GradientDescentOptimizer(stepsize=10)
steps = 35
x = np.tensor([0.0], requires_grad=True)

for i in range(steps):
    (x, ev), cost_val = opt.step_and_cost(cost, x, ev)
    if i % 5 == 0 or i == steps - 1:
        print(f"Step: {i}    Cost: {cost_val}")

print(f"QNode output after optimization = {damping_circuit(x):.4f}")
print(f"Experimental expectation value = {ev[0]}")
print(f"Optimized noise parameter p = {sigmoid(x.take(0)):.4f}") ##I don't really understand what the sigmoid does