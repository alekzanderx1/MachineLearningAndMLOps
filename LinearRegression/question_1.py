import numpy as np
import pandas as pd

# read the housing data as from csv
housing_data = pd.read_csv("../data/train.csv")

# select the features to use in a 2d array
X_df = housing_data[["1stFlrSF","2ndFlrSF","TotalBsmtSF"]]

# insert the feature for bias term as the first column in X
X_df.insert(loc=0,column='first',value=1)
X = X_df.to_numpy()
y_df = housing_data["SalePrice"]
y = y_df.to_numpy()

# calculate b value using formula
b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# generate predictions 
y_pred = X.dot(b)

# function to calculate R2 score using numpy
def calc_R2_score(actual,pred):
  error = actual - pred
  error2 = np.power(error,2)
  error_sum = np.sum(error2)
  mean = np.mean(actual)
  e2 = actual - mean
  e2_squared = np.power(e2,2)
  e2_sum = np.sum(e2_squared)
  res = 1 - error_sum/e2_sum
  return res

print(str(calc_R2_score(y,y_pred)))