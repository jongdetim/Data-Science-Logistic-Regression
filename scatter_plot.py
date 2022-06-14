#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = pd.read_csv('datasets/dataset_train.csv')

#%%
x, y = 'Defense Against the Dark Arts', 'Astronomy'
plt.scatter(data[x], data[y], s=15)
plt.xlabel(x)
plt.ylabel(y)
plt.show()
