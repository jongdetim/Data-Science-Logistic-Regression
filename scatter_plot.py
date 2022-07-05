#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = pd.read_csv('datasets/dataset_train.csv')

#%%
grps = data.groupby(['Hogwarts House'])
x, y = 'Defense Against the Dark Arts', 'Astronomy'
colors = iter(['red', 'gold', 'royalblue', 'darkgreen'])
for key, grp in grps:
    plt.scatter(grp[x], grp[y], label=key, s=15, alpha=0.5, color=next(colors))
plt.xlabel(x)
plt.ylabel(y)
plt.legend()
plt.show()

# %%
