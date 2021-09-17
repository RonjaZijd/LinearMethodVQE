import tensorflow as tf
import pennylane as qml
from pennylane.qnodes import PassthruQNode ##don't know why it doesn't recognize this, cause believe not a new thing
from pennylane import numpy as np

dev = qml.device('default.tensor.tf', wires=2)

@qml.qnode(dev)
def circuit(weights):
    qml.PauliX(wires=0) ##to put it into the basisstate 1100 (as the basisstate function is giving me trouble)
    qml.PauliX(wires=1)
    for i in range(4):
        qml.Rot(*weights[i], wires = i)
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[2,0])
    qml.CNOT(wires=[3,1])
    return qml.expval(qml.PauliZ(0))

qnode = PassthruQNode(circuit, dev)

#weights2 = tf.convert_to_tensor(np.random.normal(0, np.pi, size=[4,3]))
weights = tf.Variable(np.random.normal(0, np.pi, size=[4,3]))

with tf.GradientTape() as tape:
    tape.watch(weights)
    qnode(weights)
    state = dev._state

grad = tape.gradient(state, weights)

print("State: ", state)