import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

# These commands adjust various font sizes in the matplotlib plots.
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18

# function to plot graph using given metric array
def plot_metric(x_values, metric,label):
  plt.xticks(np.arange(min(x_values), max(x_values)+1, 1.0))
  plt.xlabel("Number of Features")
  plt.ylabel(label)
  plt.plot(x_values,metric)
  plt.show()

# features to use for traning the model iteratively
features = [
    "1stFlrSF",
    "2ndFlrSF",
    "TotalBsmtSF",
    "LotArea",
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
]

housing_data = pd.read_csv("data/train.csv")

# initialize the model and y value which remains the same
model = LinearRegression(fit_intercept=True)
y = housing_data["SalePrice"]
r2 = []
m2 = []
mabs = []
mabsP = []

# iterate through features and train model using 1, 2, 3.. and so on of the features
for i in range(1,len(features) + 1):
  current_features = features[0:i]
  X = housing_data[current_features]
  model = model.fit(X, y)
  predictions = model.predict(X)
  r2_current = r2_score(y, predictions)
  m2_error = mean_squared_error(y,predictions)
  mabs_error = mean_absolute_error(y,predictions)
  mabs_perror = mean_absolute_percentage_error(y,predictions)
  r2.append(r2_current)
  m2.append(m2_error)
  mabs.append(mabs_error)
  mabsP.append(mabs_perror)
  
  
# plot graphs using matplotlib for each metric
x_coordinate = [1+ i for i in range(len(r2))]
plot_metric(x_coordinate,r2,"R2")
plot_metric(x_coordinate,m2,"Mean Squared Error")
plot_metric(x_coordinate,mabs,"Mean Absolute  Error")
plot_metric(x_coordinate,mabsP,"Mean Absolute Error %")
