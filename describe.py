# %%
import sys
import importlib
import pandas as pd
import numpy as np

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
# print(data.head())
print(data.dtypes)

#%%
data.describe()
data = data.select_dtypes([np.number])
data = data.drop(['Index','index'], axis=1, errors='ignore')
print(data.describe())

# %%
data_descriptives = pd.DataFrame(columns=data.columns)

for col in data.columns:
    newcol = []
    cleaned_col = data[col].dropna().astype(float)
    print(len(col))
    newcol.append(descr.count(cleaned_col))
    newcol.append(descr.mean(cleaned_col))
    newcol.append(descr.std(cleaned_col))
    newcol.append(descr.min(cleaned_col))
    newcol.append(descr.percentile(cleaned_col, 25))
    newcol.append(descr.percentile(cleaned_col, 50))
    newcol.append(descr.percentile(cleaned_col, 75))
    newcol.append(descr.max(cleaned_col))
    data_descriptives[col] = newcol

data_descriptives.index = ['Count', 'Mean', 
'Std', 'Min', '25%', '50%', '75%', 'Max']

# %%
print(data_descriptives)
data.describe()

# %%
