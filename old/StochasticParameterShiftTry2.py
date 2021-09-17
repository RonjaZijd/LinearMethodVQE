"""
sample s between 0 and 1
set gp = 0
for t = 1, 2, 3 do    #choose t=1
    for all qubits do
        for r=+ and r=- do
            -basisstate computer
            -apply gates 0, to t-1 //not applicable as t=1 //not applied, because we only apply 1 gate
            -apply U^(1-s) //applied
            -apply exp^(m i pi/4 sigma_t,v) //not sure how it's determined that this should be x?
            -apply U^s
            -apply t+1, until T //not applicable ass t=1 
            -measure C //the C which we've decided is PauliZ and that's our own choise.
        set g_t,v = r+ - r-
        update gp = gp + g_t,v * #I don't understand what needs to be done here  //not needed as we only have 1 gate. 
sample gp is such that gradient is the average of gp


//let's make it easy for ourselves and only differentiate a single gate
how to test it:
-vary theta
-calculate gradient
-plot both

-they give an example which can be used for the stochastic parameter rule

"""
import pennylane as qml
from pennylane import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

dev = qml.device('default.qubit', wires=2)

I = np.eye(2)
X = [[0,1], [1,0]]
Z = [[1,0], [0,-1]]

##the thing inside the exponents of the cross resonance gate:
def cross_resonance_exp(theta1, theta2, theta3):
    return theta1* np.kron(X,I) - theta2 * np.kron(Z,X) + theta3 * np.kron(I,X)

@qml.qnode(dev)
def norm_circ(thets):
    qml.QubitUnitary(expm(1j*cross_resonance_exp(thets[0], thets[1], thets[2])), wires=[0,1]) ##not completely sure whether this step needs to be done, these steps SHOULDN't be done as we have t=1
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def test_circ():
    qml.RX(0.5, wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def big_circ(s, m, thets):
    #applying U*(1-s)
    qml.QubitUnitary(expm(1j*(1-s)*cross_resonance_exp(thets[0], thets[1], thets[2])), wires=[0,1]) #feel like maybe something else should go here
    #applying the special middle gate
    #qml.QubitUnitary(expm(1j*m*np.pi*X/4), wires=0)
    qml.QubitUnitary(expm(-1j*m*np.pi*np.kron(Z,X)/4), wires=[0,1])
    #applying the U^s
    qml.QubitUnitary(expm(1j*(s)*cross_resonance_exp(thets[0], thets[1], thets[2])), wires=[0,1])
    #and now finishing off the circuit
    return qml.expval(qml.PauliZ(0))

def g_calc(s, thets):
    rplus = big_circ(s, 1, thets)
    rminus = big_circ(s, -1, thets)
    return rplus-rminus

def grad_calc(thets):
    g_array = []
    for i in range(20):
        s = np.random.random()
        gtemp = g_calc(s, thets)
        g_array = np.append(g_array, gtemp)
    return np.average(g_array)

start_thets = [0.34, -0.15, 1.6]
thets = np.array(start_thets)
#now want to vary thet1 

thet1 = np.linspace(0, 2*np.pi, 50)
grad_array = []
norm_array = []
for th in thet1:
    thets[1] = th
    print(th)
    grad_array = np.append(grad_array, grad_calc(thets))
    norm_array = np.append(norm_array, norm_circ(thets))

plt.plot(thet1, grad_array)
plt.plot(thet1, norm_array)
plt.show()

