from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

RANDOM_SEED = 666
TEST_SIZE = 0.30

def get_X_y(data):
    ignore_features = ["id"]
    target = "satisfaction"
    features = []
    for col in data.columns:
        if col == target:
            continue
        if col not in ignore_features:
            features.append(col)

    X = data[features].copy()
    y = LabelEncoder().fit_transform(data[target]).copy()
    return X, y

# split input into train and test sets
data = pd.read_csv("../data/airline_satisfaction/train.csv", index_col=0)
train, test = train_test_split(
    data, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
X_train, y_train = get_X_y(train)
X_test, y_test = get_X_y(test)


# Create a pipeline to scale, impute and transform the various numeric and categorical columns
numeric_features = ["Age","Flight Distance", "Arrival Delay in Minutes", "Departure Delay in Minutes", 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness']
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler()), ("imputer", SimpleImputer())]
)

categorical_features = ['Gender','Customer Type', 'Type of Travel', 'Class']
categorical_transformer = OneHotEncoder(categories="auto", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(random_state=RANDOM_SEED))]
)

# hypterparameter combinations for Grid search for LogisticRegression
grid={"classifier__C":np.logspace(-1, 6, 3), "classifier__penalty":["l2"], 'classifier__fit_intercept': [True], 'classifier__solver':['lbfgs','newton-cg']}

# create and run GridSearch for 5 fold cross validation using above hyperparameters
grid_search = GridSearchCV(
    clf, grid, cv=5, scoring="roc_auc", return_train_score=True
)
grid_search = grid_search.fit(X_train, y_train)

# use the best parameters to train classifier using the same pipeline
# while Testing this turned out to be the result - 
# {'classifier__C': 0.1,
#  'classifier__fit_intercept': True,
#  'classifier__penalty': 'l2',
#  'classifier__solver': 'newton-cg'}
best_params = grid_search.best_params_
print(f"Best Parameter based on Grid Search: ", best_params)
clf = clf.set_params(**best_params)
clf = clf.fit(X_train, y_train)

# calculate and print AUC score
y_pred_proba = clf.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("Area under the ROC curve ie AUC Score: " + str(auc))