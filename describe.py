# %%
import sys
import pandas as pd

from descriptive_funcs import Descriptives as descr

# %%
if not sys.__stdin__.isatty():
    dataset_path = "datasets/dataset_test.csv"
elif not len(sys.argv) > 1:
    print("please provide path to a .csv file")
    exit(1)
else:
    dataset_path = sys.argv[1]
data = pd.read_csv(dataset_path)

# %%
data.head()

# %%
print(data.dtypes)

# %%

# %%
# data['Astronomy'].notnull()

# %%
col = data['Astronomy'].dropna().astype(float).tolist()

# %%
print(descr.sum(col))

# %%
