import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#to see the whole dataframe 
pd.set_option("display.max_rows", None, "display.max_columns", None)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, font_scale=1.15)

data = {
    "Molecule" : ['H2', 'H2', 'H2', 'H2', 'H2', 'LiH', 'LiH', 'LiH', 'LiH', 'LiH', 'BeH2','BeH2', 'BeH2', 'BeH2'],
    "Optimizer": ['BFGS', 'Gradient Descent', 'ADAM', 'Linear Method Eig', 'Linear Method Eigh', 'BFGS', 'Gradient Descent', 'ADAM', 'Linear Method Eig', 'Linear Method Eigh', 'BFGS', 'Gradient Descent', 'ADAM', 'Linear Method Eig' ],
    "Executions": [258, 1236, 501, 8055, 8055, 2250, 3928, 3291, 9024, 9024,3546, 11616, 5632, 19191]
}

df = pd.DataFrame(data)
print(df)

sns.barplot(x='Molecule', y='Executions', hue='Optimizer', data=df)
plt.show()