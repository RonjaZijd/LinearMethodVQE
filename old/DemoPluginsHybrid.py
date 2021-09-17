import pennylane as qml
from pennylane import numpy as np

dev_fock = qml.device("strawberryfields.fock", wires=2, cutoff_dim=2)

@qml.qnode(dev_fock, diff_mode="parameter-shift")
def photon_redirection(params):
    qml.FockState(1, wires=0)
    qml.Beamsplitter(params[0], params[1], wires=[0,1])
    return qml.expval(qml.NumberOperator(1))

def cost(params): #since we wish to maximize mean photon number, we minimize the negative of the circuit output
    return -photon_redirection(params)

init_params = np.array([0.01, 0.01])
print(cost(init_params))

#why don't we choose theta =0, phi =0, becuase the gradient is 0 at those values, and the optimization algorith will never descend from maximum
dphoton_redirection = qml.grad(photon_redirection, argnum=0)
print(dphoton_redirection([0.0, 0.0]))


# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# set the number of steps
steps = 100
# set the initial parameter values
params = init_params

for i in range(steps):
    # update the circuit parameters
    params = opt.step(cost, params)

    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

print("Optimized rotation angles: {}".format(params))
