#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = pd.read_csv('datasets/dataset_train.csv')
colors = iter(['red', 'gold', 'royalblue', 'darkgreen'])
grps = data.groupby(['Hogwarts House'])

#%%
for key, grp in grps:
    plt.hist(grp['Care of Magical Creatures'], label=key, alpha=0.5, color=next(colors))
plt.legend()
plt.title('Care of Magical Creatures')
plt.ylabel('Number of Students')
plt.xlabel('Grade')
plt.show()
