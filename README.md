
Welcome to the READMe of this GitHub repository. 

**Practical information**

Hi, I am Ronja Johanna Zijderveld, the creator of this repository. I am a student in Physics at the University of Manchester who produced this method as a summer internship project lasting 3 months at the University of Cologne in the research group of David Gross. This internship was made possible by the DAAD RISE program which connects PhD students and undergradute students. The internship was funded by the MLQ4 cluster and the supervisor of the project is David Wierichs. 

**The linear method in short**

The linear method is used as an optimization algorithm for the variational quantum eigensolver. It relies on linearizing the wavefunction and then calculating the quantities of the 
H matrix and the S matrix, wich are done on the quantum computer. Following this the generalized eigenvalue equation Hv=eSv is solved and the eigenvector corresponding to the lowest
eigenvalue is used as an update step for the parameters in the variational quantum eigensolver. Details and a full derivation can be found in the file 'LinearizingLinearMethod.pdf'

**Important Files**

- LMandVQE --> contains the linear method as well as 3 other optimizers (SciPy, Adam and Gradient Descent) which the Linear method can be compared to..- 
- LMandVQEjit --> same as LMandVQE, however with use of a Jax package to speed up certain parts of the algorithm. 
- LMLibrary --> contains all the functions relevant to the Linear Method
- Hamiltonians --> contains the written out Hamiltonians of the systems tested as well as functions to switch from an array representation to the hamiltonian written out in 1 line
- LMandVQEGUI --> contians the same as LMandVQE, however now with a simple user interface to change certain settings. 
- LinearizingLinearMethod --> Derivation showing how the Linear Method works
- HandSMatrixLinearMethod --> Derivation showing how to obtain the H and S matrix using a quantum computer
- RealEigenvaluesLInearMethod --> Derivation showing why the eigenvalues of the relevant eigenvalue equation are real 
- Presentation --> slides which accompanied a talk given to the research group near the end of the internship

**SetUpInstructions**

The main file to use is 'LinearMethodandVQE.py'. In its current form it compares the linear method to the optimizers SciPy, Adam and GradientDescent. It uses a simple variational circuit of
12 parameters and uses the hamiltonian of a Hydrogen molecule. Settings to test a different system can be changed at the beginning of the file. The variational circuit can be
filled in manually, the form it should have is: [[gate1 on wire1, gate2 on wire1, ...], [gate1 on wire2, gate2 on wire2, ... ] ...]. Furthermore when switching to a different
variational circuit the shape of the parameters should be changed in both 'LinearMethodandVQE.py' AND 'LMLibrary.py'. The hamiltonian can also be changed to the hamiltonian of a 
different system either by using the hamiltonians provided in the 'HamiltoninasLibrary.py' or by using the quantum chemistry package of Pennylane. In the case of using the quantum
chemistry package one should use the function called '....' from the 'HamiltoniansLibrary.py' to get the Hamiltonian in the correct form for use for the program. 

**Eig vs Eigh**

Note! The goal of the linear method is to solve the generalized eigenvalue equation Hv = eSv. To do this two different optimizers were used: scipy.eig and scipy.eigh, which surprisingly enough give different results. This was the point at which the project had to be ended so it has not yet been resolved, however it was seen that while scipy.eig manages to optimize effectively (having the same performance as the other optimizers which it was compared to), scipy.eigh does not. The LMLibrary file contains two seperate functions for solving the generalized eigenvalue equation to give the user choice in which one to use. 
