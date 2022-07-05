#%% import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% read
orig_data = pd.read_csv('datasets/dataset_train.csv')

#%% clean
data = orig_data.select_dtypes([np.number])
data = data.drop(['Index','index'], axis=1, errors='ignore')
data.insert(0, 'Hogwarts House', orig_data['Hogwarts House'])

#%% plot
palette = {'Gryffindor' : 'red', 'Ravenclaw' : 'royalblue', 'Slytherin' : 'darkgreen', 'Hufflepuff' : 'gold'}
sns.pairplot(data, diag_kind='hist', hue='Hogwarts House', palette=palette, plot_kws={'alpha':0.5})
plt.show()

# %%
