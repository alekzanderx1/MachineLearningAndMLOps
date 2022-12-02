import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

RANDOM_SEED = 666
np.random.seed(RANDOM_SEED)


payments = pd.read_csv("./data/payments.csv")
merchants = pd.read_csv("./data/merchants.csv")
buyers = pd.read_csv("./data/buyers.csv")

# Data Prepration  - Clean, Sort, and Join the datasets 
merchants.rename(columns={'country': 'merchant_country'}, inplace=True)
merchants.rename(columns={'category': 'merchant_category'}, inplace=True)
merchants.rename(columns={'id': 'merchantId'}, inplace=True)
buyers.rename(columns={'country': 'buyer_country'}, inplace=True)
buyers.rename(columns={'id': 'buyerId'}, inplace=True)

# join payments to buyer and merchant datasets
payments = payments.merge(merchants, left_on='merchant_id', right_on='merchantId')
payments = payments.merge(buyers, left_on='buyer_id', right_on='buyerId')

payments[['transaction_timestamp','chargeback_timestamp']] = payments[['transaction_timestamp','chargeback_timestamp']].apply(pd.to_datetime) 
payments = payments.sort_values(by=['transaction_timestamp','chargeback_timestamp'])




# Part 1
chargebackPayments = payments[payments['chargeback_timestamp'].notnull()].copy()
chargebackPayments['chargebackInterval'] = (chargebackPayments['chargeback_timestamp'] - chargebackPayments['transaction_timestamp']).dt.days
chargebackInterval = chargebackPayments['chargebackInterval'].quantile(.95)
print("Chargeback interval for 95% of payments is " + str(chargebackInterval) + " days")





# Part 3 
# since data is sorted above by transaction date, we get cumulative count for buyer and merchant transaction on each row to help with calculation
payments['merchantCount'] = payments.groupby('merchant_id').cumcount()
payments['buyerCount'] = payments.groupby('buyer_id').cumcount()

for row in payments.itertuples():
    buyerFraudRate = 0
    if row.buyerCount != 0:
        buyerFraudRate = payments[(payments['chargeback_timestamp'].notnull()) & (payments['buyer_id'] == row.buyer_id) & (payments['transaction_timestamp'] < row.transaction_timestamp) & (payments['chargeback_timestamp'] < row.transaction_timestamp)].shape[0]/row.buyerCount
    payments.at[row.Index,'buyerFraudRate'] = buyerFraudRate
    
for row in payments.itertuples():
    merchantFraudRate = 0
    if row.merchantCount != 0:
        merchantFraudRate = payments[(payments['chargeback_timestamp'].notnull()) & (payments['merchant_id'] == row.merchant_id) & (payments['transaction_timestamp'] < row.transaction_timestamp) & (payments['chargeback_timestamp'] < row.transaction_timestamp)].shape[0]/row.merchantCount
    payments.at[row.Index,'merchantFraudRate'] = merchantFraudRate

payments = payments.drop(columns=['merchantCount', 'buyerCount'])

print("Sum of average merchant fraud rates: " + str(payments['merchantFraudRate'].sum()))
print("Sum of average buyer fraud rates: " + str(payments['buyerFraudRate'].sum()))
print("Sum of both fraud rates: " + str(payments['buyerFraudRate'].sum() + payments['merchantFraudRate'].sum()))





# Part 2
test_start_date = '2022-8-01'
test_end_date = '2022-8-31'
payments_test = payments[(payments['transaction_timestamp'] >= test_start_date) & (payments['transaction_timestamp'] <= test_end_date)].copy()
payments_train = payments[(payments['transaction_timestamp'] < pd.to_datetime(test_start_date) - pd.to_timedelta(chargebackInterval, unit='d'))].copy()
print(f"Test dataset timestamp range: {payments_test['transaction_timestamp'].min()} to {payments_test['transaction_timestamp'].max()}")
print(f"Train dataset timestamp range: {payments_train['transaction_timestamp'].min()} to {payments_train['transaction_timestamp'].max()}")




# Part 4
# Create a target variable as isFraud based on existance of charback_timestamp
def create_target(row):
    val = 1
    if row['chargeback_timestamp'] is pd.NaT:
        val = 0
    return val
payments_test['isFraud'] = payments_test.apply(create_target, axis=1)
payments_train['isFraud'] = payments_train.apply(create_target, axis=1)

# Separate train and test data using the requires features
X_test = payments_test[['payment_amount','merchant_category','merchant_country','buyer_country','merchantFraudRate','buyerFraudRate']]
y_test = payments_test['isFraud']
X_train = payments_train[['payment_amount','merchant_category','merchant_country','buyer_country','merchantFraudRate','buyerFraudRate']]
y_train = payments_train['isFraud']

# create a Pipeline to pre-process numerical and categorical columns and train it using above data
numeric_features = ['payment_amount','merchantFraudRate','buyerFraudRate']
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler()), ("imputer", SimpleImputer())]
)
categorical_features = ['merchant_category','merchant_country', 'buyer_country']
categorical_transformer = OneHotEncoder(categories="auto", sparse=False)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(class_weight='balanced',random_state=RANDOM_SEED))]
)
clf = clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)[::,1]


auc = metrics.roc_auc_score(y_test, y_pred_proba)
print("Area under the ROC curve ie AUC Score: " + str(auc))
rocDisplay = RocCurveDisplay.from_predictions(y_test, y_pred_proba)
plt.show()

