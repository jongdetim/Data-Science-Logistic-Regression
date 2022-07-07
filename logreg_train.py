# %%
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from logistic_regression import LogisticRegression

# %% reading file
if not sys.__stdin__.isatty():
    dataset_path = "datasets/dataset_train.csv"
elif len(sys.argv) <= 1:
    print("please provide path to a .csv file")
    sys.exit(1)
else:
    dataset_path = sys.argv[1]
data = pd.read_csv(dataset_path)

# %% removing rows with missing features, splitting data into dependent & independent
data = data.drop(['Index', 'index'], axis=1, errors='ignore')
cleaned_data = data.select_dtypes([np.number])
cleaned_data.insert(0, 'Hogwarts House', data['Hogwarts House'])
cleaned_data = cleaned_data.dropna()
cleaned_data.reset_index(drop=True, inplace=True)
x = cleaned_data.select_dtypes([np.number]).values
y = cleaned_data['Hogwarts House'].values

# %% scale data to std
scaler = StandardScaler()
scaler.fit(x)
x = scaler.fit_transform(x)

# %% train the model
model = LogisticRegression(alpha=0.01, n_iteration=10).fit(x, y)

# %% write model to file
model.save_model()

#%%
model.accuracy(x, y)
# %%
model.plot_cost()
# %%
