import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import psi4

symbols, coordinates = qchem.read_structure('h2o.xyz')

print("The total number of atoms is: {}".format(len(symbols)))
print(symbols)
print(coordinates)

name = 'water'
charge = 0
multiplicity = 1
basis_set = 'sto-3g'

#building the hartrock free file using the mean field calculation and information supplied:

#
hf_file = qchem.meanfield(
    symbols,
    coordinates,
    name=name,
    charge=charge,
    mult=multiplicity,
    basis=basis_set,
    package='psi4' #could also do the Psi4 file #can't get this package to work somehow
)

#print(hf_file)

electrons = 10
orbitals = 7
core, active = qchem.active_space(electrons, orbitals, active_electrons=4, active_orbitals=4)
print("List of core orbitals: {:}".format(core))
print("List of active orbitals: {:}". format(active))
print("Number of qubits required for quantum simulation: {:}".format(2*len(active)))

no_core_orbitals, all_active = qchem.active_space(electrons, orbitals)
print("List of core orbitals: {:}".format(no_core_orbitals))
print("List of active orbitals: {:}". format(all_active))
print("Number of qubits required for quantum simulation: {:}".format(2*len(all_active))) 

##building the hamiltionian without the seperate hartrock free approximation file:

H, qubits = qchem.molecular_hamiltonian(
    symbols,
    coordinates,
    charge=charge,
    mult=multiplicity,
    basis=basis_set,
    package='psi4',    #what if I simply don't specify this? 
    active_electrons=4,
    active_orbitals=4,
    mapping='jordan_wigner '
) 

print("Number of qubits required to perform simulations: {:}".format(qubits))
print("Electronic Hamiltonian of the water molecule represented in Pauli basis")
print(H)
