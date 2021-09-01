import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
name_csv_file = "LiHSystem70ItsReg0.01.csv"
#name_csv_file = "HHSystem60ItsReg0.01.csv"
df = pd.read_csv(name_csv_file)
df.drop(columns = df.columns[0], axis=1, inplace=True)
print(df)

sns.stripplot(jitter=0.4, data=df)
plt.xlabel('Optimizer')
plt.ylabel('Energy')
plt.show()