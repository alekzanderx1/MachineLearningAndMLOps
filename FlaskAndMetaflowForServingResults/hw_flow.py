"""

Script showing end-to-end training of a Loan Approval classification model using Metaflow.

"""


from metaflow import FlowSpec, step, Parameter, IncludeFile, current, card
from metaflow.cards import Image
from datetime import datetime
import os
import requests, pandas, string
#from comet_ml import Experiment


# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'


class LoanApprovalClassificationFlow(FlowSpec):
    """
    DecisionTreeFlow is a DAG showcasing reading loan approval dataset from a file 
    and training a model and performing generalized and fairness tests on the same.
    """
    
    # if a static file is part of the flow, 
    # it can be called in any downstream process,
    # gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files
    DATA_FILE = IncludeFile(
        'dataset',
        help='Text file with the dataset',
        is_text=True,
        default='loans_dataset.csv')

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing and validation',
        default=0.20
    )
    
    MAX_DEPTHS = Parameter(
        name='max_depths',
        help='Hyperparameter for DecisionTreeClassifier, requires comma separated values',
        default='2,4,6,8'
    )

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the static file
        """
        from io import StringIO
        import pandas as pd

        self.dataframe = pd.read_csv(StringIO(self.DATA_FILE))
        self.Xs = self.dataframe
        self.Ys = self.Xs.pop('loan_approved')
        print("Total of {} rows in the dataset!".format(len(self.dataframe)))
        # go to the next step
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        """
        Check data is ok before training starts
        """
        assert(all(y == 1.0 or y == 0.0 for y in self.Ys.values))
        assert(all(age <= 65 or age >= 21 for age in self.Xs['age'].values))
        assert(all(creditScore <= 850 for creditScore in self.Xs['credit_score'].values))
        self.next(self.prepare_train_test_and_validation_dataset)

    @step
    def prepare_train_test_and_validation_dataset(self):
        """
        Prepare train, test, and validation splits from dataset
        """
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Xs, 
            self.Ys, 
            test_size=self.TEST_SPLIT, 
            random_state=42
            )
        
        self.X_train_val, self.X_val, self.y_train_val, self.y_val = train_test_split(
            self.X_train, 
            self.y_train, 
            test_size=self.TEST_SPLIT, 
            random_state=42
            )

        self.max_depth_hyperparameter = [int(val) for val in self.MAX_DEPTHS.split(',')]
        self.next(self.train_model, foreach="max_depth_hyperparameter")

    @step
    def train_model(self):
        """
        Train a Decision Tree classifier on the training set
        """
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        self.max_depth_value = self.input
        numeric_features = ["age","credit_score","loan_amount"]
        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler()), ("imputer", SimpleImputer())]
        )

        categorical_features = ['gender','race', 'customer_type']
        categorical_transformer = OneHotEncoder(categories="auto", sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder = 'passthrough'
        )

        clf = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier(max_depth=self.max_depth_value))]
        )

        clf = clf.fit(self.X_train_val, self.y_train_val)
        # now, make sure the model is available downstream
        self.model = clf
        # go to the testing phase
        self.next(self.test_model_on_validation)
            

    @step 
    def test_model_on_validation(self):
        """
        Test the model on the validation sample
        """
        from sklearn import metrics
        
        self.y_predicted = self.model.predict(self.X_val)
        self.y_predict_proba = self.model.predict_proba(self.X_val)[:,1]
        self.roc_auc = metrics.roc_auc_score(self.y_val, self.y_predict_proba)
        print('ROC AUC score is {}'.format(self.roc_auc))
        # all is done, next compare the scores and select the best one
        self.next(self.join)
        
    @step
    def join(self,inputs):
        """
        Finds the best hyperpameter based on score from all training iterations
        """
        self.merge_artifacts(inputs, exclude=['y_predicted','y_predict_proba','roc_auc','max_depth_value','model'])
        print("Choosing the best  model based on validations scores")
        best_model = max(inputs, key=lambda x: x.roc_auc)
        print(f"Best validation score for max_depth: {best_model.max_depth_value} , score: {best_model.roc_auc}")
        self.best_depth = best_model.max_depth_value
        self.best_model = best_model.model
        self.next(self.test_best_model)
    
    @card(type='blank',id='visual')
    @step
    def test_best_model(self):
        """
        Train and Evaluate best model on Test split and log the results
        Evalution done using Generalized tests and per-group Fairness tests using Miss Rate
        """
        from sklearn import metrics
        import matplotlib.pyplot as plt
    
        # Fit model on entire Train dataset
        self.best_model = self.best_model.fit(self.X_train,self.y_train)
        
        # Generalization results
        y_pred = self.best_model.predict(self.X_test)
        y_predict_proba = self.best_model.predict_proba(self.X_test)[:,1]
        self.roc_auc = metrics.roc_auc_score(self.y_test, y_predict_proba)
        print(f"Test ROC_AUC score for best model is: {self.roc_auc}")
        self.test_miss_rate = sum(x != y for x, y in zip(self.y_test, y_pred))/len(self.y_test)
        print(f"Miss Rate for best model is: {self.test_miss_rate}")
        
        # Fairness testing using Miss Rate
        # First slice test set into gender groups
        group_indices = self.X_test.groupby(by='gender').indices
        slices = []
        for group in group_indices:
          slices.append([group, self.X_test.iloc[group_indices[group]], self.y_test.iloc[group_indices[group]]])
        
        perGroupMissRate = dict()
        # Make prediction on each group and store result
        for slice in slices:
            y_pred_slice = self.best_model.predict(slice[1])
            prediction_errors = sum(x != y for x, y in zip(slice[2], y_pred_slice))
            perGroupMissRate[slice[0]] = prediction_errors/len(y_pred_slice)
            
        self.missRates = perGroupMissRate
        print(f"Miss rates by gender: {self.missRates}")
        
        # Document a Bar chart in a DAG card
        fig = plt.figure()
        plt.title('Miss Rate by Gender')
        plt.bar(*zip(*self.missRates.items()))
        current.card['visual'].append(Image.from_matplotlib(fig))
        
        self.next(self.end)
        

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    LoanApprovalClassificationFlow()
