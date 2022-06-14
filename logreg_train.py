# %%
import sys
import pandas as pd
import numpy as np

# %% reading file
if not sys.__stdin__.isatty():
    dataset_path = "datasets/dataset_train.csv"
elif len(sys.argv) <= 1:
    print("please provide path to a .csv file")
    exit(1)
else:
    dataset_path = sys.argv[1]
data = pd.read_csv(dataset_path)

# %% removing rows with missing features
data = data.drop(['Index','index'], axis=1, errors='ignore')
x = data.select_dtypes([np.number])
y = data['Hogwarts House']
