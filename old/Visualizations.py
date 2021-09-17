import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#to see the whole dataframe 
pd.set_option("display.max_rows", None, "display.max_columns", None)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, font_scale=1.15)

name_csv_file = "LiHSystem70ItsReg0.01.csv"
#name_csv_file = "HHSystem50ItsReg0.01.csv"
name_csv_file = "BeHSystem50ItsReg0.01.csv"
#name_csv_file2 = "LiHSystemVarItsReg0.001wEIGH.csv"
name_other_csv_file = "HHSystem50ItsReg0.01execsfile.csv"
df_cost = pd.read_csv(name_other_csv_file)
df = pd.read_csv(name_csv_file)
df.drop(columns = df.columns[0], axis=1, inplace=True)
df_cost.drop(columns = df_cost.columns[0], axis=1, inplace=True)
#df.columns = ['BFGSenergy', 'Gradenergy', 'ADAMenergy', 'LMenergy']
print(df)
print()
print(df_cost)
print()

print(df.mean())
# df2 = pd.read_csv(name_csv_file2)
# df2 = df2.rename(columns={"Linear Method":"Linear Method Eigh"})
# print(df2)
# LMwEIGH = df2["Linear Method Eigh"]
# df = df.join(LMwEIGH)
print(df)
df = df.rename(columns = {"Linear Method":"Linear Method Eig"})
 
df = df[["BFGS", "Gradient Descent", "ADAM", "Linear Method Eig"]]
print(df)

#### Getting the percentages ###############

LM_stuck = 0
Grad_stuck = 0
Adam_stuck = 0
Scipy_stuck = 0
tots=0

for x in df['Linear Method Eig']:
    if x>-1.6:
        LM_stuck = LM_stuck+1
    tots=tots+1

for x in df['Gradient Descent']:
    if x>-1.6:
        Grad_stuck+=1

for x in df['ADAM']:
    if x>-1.6:
        Adam_stuck+=1
        
for x in df['BFGS']:
    if x>-1.6:
        Scipy_stuck+=1
        
print("Percentages getting stuck: ")
print("LM: ", 100-(LM_stuck/tots)*100)
print("Adam: ", 100-(Adam_stuck/tots)*100)
print("Grad: ", 100-(Grad_stuck/tots)*100)
print("SciPy: ", 100-(Scipy_stuck/tots)*100)





##to do for the graph which I want: 

#df = df.rename(columns = {"Linear Method Eig": "LMenergy", "Gradient Descent":"Gradenergy", "ADAM":"ADAMenergy", "BFGS":"BFGSenergy"})


#sns.scatterplot(data=df)
sns.stripplot(jitter=0.4, data=df)
#sns.violinplot(data=df)
#sns.boxplot(data=df)
plt.ylim(-1.75, 0)
plt.xlabel('Optimizer')
plt.ylabel('Energy')
#plt.title('HHusingEIGH')
plt.show()

#concatenated = pd.concat([df, df_cost], axis=1)
#print(concatenated)

# sns.scatterplot(x='Linear Method', y='LMenergy', data=concatenated, label='Linear Method')
# sns.scatterplot(x='Gradient Descent', y='Gradenergy', data=concatenated, label='Gradient Descent')
# sns.scatterplot(x='ADAM', y='ADAMenergy', data=concatenated, label='ADAM')
# sns.scatterplot(x='BFGS', y='BFGSenergy', data=concatenated, label='BFGS (SciPy)')
# plt.xlabel('Device Executions')
# plt.ylabel('Energy')
# plt.legend()
# plt.show()
