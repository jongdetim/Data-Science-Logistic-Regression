# %%
import csv
import sys
import numpy as np
# needed to eval file from str to list of tuples of np.array & string
from numpy import array
import pandas as pd
from sklearn.preprocessing import StandardScaler

from logistic_regression import LogisticRegression

output_path = 'houses.csv'

# %% reading file
if not sys.__stdin__.isatty():
    dataset_path = 'datasets/dataset_test.csv'
    weight_path = 'datasets/weights.txt'
elif len(sys.argv) <= 2:
    print("please provide: 1. path to a .csv file, and 2. a file containing weights")
    sys.exit(1)
else:
    dataset_path = sys.argv[1]
    weight_path = sys.argv[2]
data = pd.read_csv(dataset_path)

# %%
with open(weight_path, 'r', encoding="utf8") as weights_file:
    contents = weights_file.read()
    weights = eval(contents)

# %% set model weights to loaded weights from file
model = LogisticRegression()
model.theta = weights

# %% clean up and scale data
df = pd.read_csv(dataset_path)
data = df.drop(['Index', 'index', 'Hogwarts House'], axis=1, errors='ignore')
cleaned_data = data.select_dtypes([np.number])
cleaned_data = cleaned_data.fillna(method='ffill')
cleaned_data.reset_index(drop=True, inplace=True)
x = cleaned_data.select_dtypes([np.number]).values
scaler = StandardScaler()
scaler.fit(x)
x = scaler.fit_transform(x)

# %% predict houses
predictions = model.predict(x)

# %% write predictions to file
with open(output_path, 'w', encoding='utf8', newline='') as output_file:
    output_writer = csv.writer(output_file, delimiter=',', lineterminator='\n')
    output_writer.writerow(('Index', 'Hogwarts House'))
    for i, prediction in enumerate(predictions):
        output_writer.writerow((i, prediction))

# %%
