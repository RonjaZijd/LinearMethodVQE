import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#to see the whole dataframe 
pd.set_option("display.max_rows", None, "display.max_columns", None)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, font_scale=1.15)

#name_csv_file = "LiHSystem70ItsReg0.01.csv"
name_csv_file = "HHSystem50ItsReg0.01.csv"
name_other_csv_file = "BeHSystem50ItsReg0.01execsfile.csv"

name_csv_eigh_file = "HHSystemVarItsReg0.01wEIGH (3).csv"
name_csv_eigh_other_file = "LiHSystemVarItsReg0.001execsfileweigh.csv"

#name_csv_file = "BeHSystem50ItsReg0.01.csv"
#name_csv_file2 = "LiHSystemVarItsReg0.001wEIGH.csv"
df_cost = pd.read_csv(name_other_csv_file)
df = pd.read_csv(name_csv_file)
df.drop(columns = df.columns[0], axis=1, inplace=True)
df_cost.drop(columns = df_cost.columns[0], axis=1, inplace=True)
#df.columns = ['BFGSenergy', 'Gradenergy', 'ADAMenergy', 'LMenergy']

print("This is df_cost before changing anything about it!")
print(df_cost)
print()

print(df.mean())
dfeigH = pd.read_csv(name_csv_eigh_file)
dfeigH = dfeigH.rename(columns={"Linear Method":"Linear Method Eigh"})
#print(df2)
LMwEIGH = dfeigH["Linear Method Eigh"]
df = df.join(LMwEIGH)

df = df.rename(columns = {"Linear Method":"Linear Method Eig"})
 
#df = df[["BFGS", "Gradient Descent", "ADAM", "Linear Method Eig", "Linear Method Eigh"]]



dfeigHcost = pd.read_csv(name_csv_eigh_other_file)
dfeigHcost = dfeigHcost.rename(columns={"Linear Method":"Linear Method Eigh"})
LMwEIGHcost = dfeigHcost["Linear Method Eigh"]
df_cost = df_cost.join(LMwEIGHcost)
df_cost = df_cost.rename(columns = {"Linear Method":"Linear Method Eig"})

print("This is df_cost after changing things...")
print(df_cost)
##to do for the graph which I want: 

print("For HH system, this is the mean of the executions::")
print(df_cost.mean())

df = df.rename(columns = {"Linear Method Eig": "LMenergy", "Gradient Descent":"Gradenergy", "ADAM":"ADAMenergy", "BFGS":"BFGSenergy", "Linear Method Eigh":"LMenergy2"})

    #SINGULAR PLOTTING
#sns.scatterplot(data=df)
#sns.stripplot(jitter=0.4, data=df)
#sns.violinplot(data=df)
#sns.boxplot(data=df)
#plt.xlabel('Optimizer')
#plt.ylabel('Energy')
#plt.title('HHusingEIGH')
#plt.show()

print(df_cost)
print()
concatenated = pd.concat([df, df_cost], axis=1)
print(concatenated)

sns.scatterplot(x='Linear Method Eigh', y='LMenergy2', data=concatenated, label='Linear Method')
sns.scatterplot(x='Linear Method Eig', y='LMenergy', data=concatenated, label='Linear Method')
sns.scatterplot(x='Gradient Descent', y='Gradenergy', data=concatenated, label='Gradient Descent')
sns.scatterplot(x='ADAM', y='ADAMenergy', data=concatenated, label='ADAM')
sns.scatterplot(x='BFGS', y='BFGSenergy', data=concatenated, label='BFGS (SciPy)')
plt.xlabel('Device Executions')
plt.ylabel('Energy')
plt.legend()
plt.show()

###what do I want in the cost graph: 
    #clearly showing the final points
    #if I can, using different markers multiple molecules
    #figure out which files I have here: 
    