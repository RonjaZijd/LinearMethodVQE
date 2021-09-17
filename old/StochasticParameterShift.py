import pennylane as qml
from pennylane import numpy as np

"""
#need to generate a random distribution to sample from
#need to prepare the state correctly, (same basis state as regular parameter-shift program)
#define two gates, one which calculates the r+ and one which calculates the r-
#it takes in the same s which is sampled from a uniform distribution between 0 and 1

#for the r+ measurement we need the gates:
    #exp(i(1-s)(H+xV))
    #exp(i pi V /4)
    #exp(i s (H+xV))

#for the r- measurement we need the gates:
    #exp(i(1-s)(H+xV))
    #exp(-i pi V /4)
    #exp(i s (H+xV))

#ISSUES:
    #before I attempt to start writing the circuit, I am unsure which gates those described belong to and whether they can even be done in a general way
    #not sure whether I need to repeat this calculation a lot of times before I converge on something?
    #in algorithm 2 they're talking about a rescaling which I don't completely understand --> I believe I understand the scaling
    #how do I decide what I measure

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev, )
define circuit_rplus(params): one of the parameters should definitely be s
    -the different things

    -return 

define circuit_rminus(params):
    -the correct gates


#translate to phi before inputting it to the general circuit, are the only differences are minuses
#could also try doing it using 
"""

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(x, s, ph):
    #qml. not entirely sure how to code in pauli0
    phi = -2*x*(1-s)
    qml.RY(phi, wires=0)
    qml.RZ(phi, wires=0) ##I'm relatively sure that this is incorrect 
    qml.RX(phi, wires=0)

    #the special rotation
    qml.RX(ph, wires=0) #this will have to be specified when it's done. 

    #the third gate
    phi = -2*x*s
    qml.RY(phi, wires=0)
    qml.RZ(phi, wires=0)
    qml.RX(phi, wires=0)

    return qml.expval(qml.PauliZ(0)) #let's pretend for now that PauliZ is the thing which we need to measure

circuit(0.2, 0.8, -np.pi/2)
print(circuit.draw())

#x is our parameter
x = np.random.random() #random number between 0 and 1
s = np.random.random() #need to figure out how many times I need to recalculate s?

def parameter_shift_term(x,s):
    rplus = circuit(x, s, -np.pi/2)
    rminus = circuit(x, s, np.pi/2)

    return rplus - rminus ##we've gotten a number although it is most definitely not the correct number. 
    #continue reading the paper and figure out how to combine it for multiple paremeters, let's aim for 2 parameters.//

print(parameter_shift_term(x,s))