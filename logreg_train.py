# %%
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from logistic_regression import LogisticRegression

# %% reading file
if not sys.__stdin__.isatty():
    dataset_path = "datasets/dataset_train.csv"
elif len(sys.argv) <= 1:
    print("please provide path to a .csv file")
    exit(1)
else:
    dataset_path = sys.argv[1]
data = pd.read_csv(dataset_path)

# %% removing rows with missing features, splitting data into dependent & independent. resetting index
data = data.drop(['Index','index'], axis=1, errors='ignore')
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

# # %%
# model = LogisticRegression(n_iteration=2)

# #%%
# model = model.fit(x, y)
# score = model.accuracy(x, y)
# print(score)
# model._plot_cost(model.cost)

#%%
# prediction = model.predict(x)
# print(sum(prediction[i] == y[i] for i in range(len(prediction))) / len(prediction))

#%%
# model.accuracy(x, y)

# %%
# model._plot_cost(model.cost)

# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33)
logi = LogisticRegression(alpha=0.01, n_iteration=30).fit(x_train, y_train)
# prediction1 = logi.predict2(x_test)
score1 = logi.accuracy(x_train,y_train)
print("the accuracy of the modelon TRAIN DATASET is:", score1)
score2 = logi.accuracy(x_test,y_test)
print("the accuracy of the modelon TEST DATASET is:", score2)

#%%
# logi._plot_cost(logi.cost)

# %%
# logi._plot_weights()

# %%
truth = pd.read_csv('datasets/dataset_truth.csv')
truth.head()

# %%
logi.save_model()

# %%
print(type(logi.theta[0][0]))
print(logi.theta[0][0])
# %%
