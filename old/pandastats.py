#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:09:35 2021
Panda stats
@author: rz

Stats which I want to include: 
    -final average of each column
    -percentage of being above -1 (aka getting stuck in 0.43)

"""

import pandas as pd

name_file = "HHSystem60ItsReg0.01.csv"


df = pd.read_csv(name_file)
print(df.mean(axis=0))
print(df)
LM_stuck = 0
Grad_stuck = 0
Adam_stuck = 0
Scipy_stuck = 0
tots=0

for x in df['Linear Method']:
    if x>-1:
        LM_stuck = LM_stuck+1
    tots=tots+1

for x in df['Gradient Descent']:
    if x>-1:
        Grad_stuck+=1

for x in df['ADAM']:
    if x>-1:
        Adam_stuck+=1
        
for x in df['BFGS']:
    if x>-1:
        Scipy_stuck+=1
        
print("Percentages getting stuck: ")
print("LM: ", (LM_stuck/tots)*100)
print("Adam: ", (Adam_stuck/tots)*100)
print("Grad: ", (Grad_stuck/tots)*100)
print("SciPy: ", (Scipy_stuck/tots)*100)
        
        

