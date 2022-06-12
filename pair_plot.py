#%% import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% read
data = pd.read_csv('datasets/dataset_train.csv')

#%% clean
data = data.select_dtypes([np.number])
data = data.drop(['Index','index'], axis=1, errors='ignore')

#%% plot
sns.pairplot(data)
plt.show()
