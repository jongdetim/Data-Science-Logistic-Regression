# %%
import sys
import numpy as np
import pandas as pd
import importlib

import descriptives as descr

#%%
descr = importlib.reload(descr)

# %%
if not sys.__stdin__.isatty():
    dataset_path = "datasets/dataset_train.csv"
elif len(sys.argv) <= 1:
    print("please provide path to a .csv file")
    exit(1)
else:
    dataset_path = sys.argv[1]
data = pd.read_csv(dataset_path)

# %%
print(data.head())
print(data.dtypes)

# %%
# Delete all rows with missing data from dataset
# data = data.dropna()

# %%
# deletes only elements in column that are missing
print(len(data["Astronomy"]))
col = data['Astronomy'].dropna().astype(float).tolist()
print(len(col))
print(data["Astronomy"].isna().sum())

# %%
print(descr.mean(col))
print(descr.count(col))
print(descr.min(col))
print(descr.max(col))
print(descr.var(col))
print(descr.std(col))

# %%
print(np.percentile(col, 50, method='nearest'))
print(descr.percentile(col, 50))

# %%
