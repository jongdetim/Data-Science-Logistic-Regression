#%%
import csv
import numpy as np
# needed to cast file from str to list of tuples of np.array & string
from numpy import array
import pandas as pd
from sklearn.preprocessing import StandardScaler

from logistic_regression import LogisticRegression

weight_path = './datasets/weights.csv'
dataset_path = './datasets/dataset_test.csv'
output_path = 'houses.csv'

#%%
with open(weight_path, 'r') as weights_file:
    contents = weights_file.read()
    weights = eval(contents)

# %%
model = LogisticRegression()
model.theta = weights

#%%
df = pd.read_csv(dataset_path)
data = df.drop(['Index', 'index', 'Hogwarts House'], axis=1, errors='ignore')
cleaned_data = data.select_dtypes([np.number])
cleaned_data = cleaned_data.fillna(method='ffill')
cleaned_data.reset_index(drop=True, inplace=True)
x = cleaned_data.select_dtypes([np.number]).values
scaler = StandardScaler()
scaler.fit(x)
x = scaler.fit_transform(x)
#%%
np.set_printoptions(threshold=np.inf)
print(x)
predictions = model.predict(x)

# %% write predictions to file

with open(output_path, 'w', newline='') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', lineterminator='\n')
        output_writer.writerow(('Index', 'Hogwarts House'))
        for i, prediction in enumerate(predictions): 
            output_writer.writerow((i, prediction))
# %%
