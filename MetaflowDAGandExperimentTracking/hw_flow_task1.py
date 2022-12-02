"""

Script showing end-to-end training of a classification model using Metaflow with tracking done in Comet ML. 

"""


from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os
from comet_ml import Experiment


# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'


class ClassificationFlow(FlowSpec):
    """
    DecisionTreeFlow is a DAG showcasing reading data from a file 
    and training a model with cross validation successfully.
    """
    
    # if a static file is part of the flow, 
    # it can be called in any downstream process,
    # gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files
    DATA_FILE = IncludeFile(
        'dataset',
        help='Text file with the dataset',
        is_text=True,
        default='classification_dataset.txt')

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing and validation',
        default=0.20
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

        raw_data = StringIO(self.DATA_FILE).readlines()
        print("Total of {} rows in the dataset!".format(len(raw_data)))
        self.dataset = [[float(_) for _ in d.strip().split('\t')] for d in raw_data]
        print("Raw data: {}, cleaned data: {}".format(raw_data[0].strip(), self.dataset[0]))
        self.Xs = [[_[0],_[1],_[2],_[3],_[4],_[5]] for _ in self.dataset]
        self.Ys =  [_[6] for _ in self.dataset]
        # go to the next step
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        """
        Check data is ok before training starts
        """
        assert(all(y == 1.0 or y == 0.0 for y in self.Ys))
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
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, 
            self.y_train, 
            test_size=self.TEST_SPLIT, 
            random_state=42
            )

        self.max_depth_hyperparameter = [2, 4, 8, 16]
        self.next(self.train_model, foreach="max_depth_hyperparameter")

    @step
    def train_model(self):
        """
        Train a Decision Tree classifier on the training set
        """
        from sklearn.tree import DecisionTreeClassifier
        self.max_depth_value = self.input
        clf = DecisionTreeClassifier(max_depth=self.max_depth_value)
        clf = clf.fit(self.X_train, self.y_train)
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
        
        experiment = Experiment(
            api_key="",
            project_name="hw4-metaflow",
            workspace="",
        )
        
        experiment.add_tag("Split:" + str(self.TEST_SPLIT))
        self.y_predicted = self.model.predict(self.X_val)
        self.y_predict_proba = self.model.predict_proba(self.X_val)[:,1]
        self.roc_auc = metrics.roc_auc_score(self.y_val, self.y_predict_proba)
        print('ROC AUC score is {}'.format(self.roc_auc))
        experiment.log_parameters({"max_depth":self.max_depth_value})
        experiment.log_metrics({"roc_auc": self.roc_auc})
        experiment.log_confusion_matrix([ int(val) for val in self.y_val], [ int(val) for val in self.y_predicted])
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
        self.next(self.evaludate_best_model)
    
    @step
    def evaludate_best_model(self):
        """
        Evaluate best model on Test split and log the results
        """
        from sklearn import metrics
    
        y_pred = self.best_model.predict(self.X_test)
        y_predict_proba = self.best_model.predict_proba(self.X_test)[:,1]
        self.roc_auc = metrics.roc_auc_score(self.y_test, y_predict_proba)
        print(f"Test score for best model is: {self.roc_auc}")
        self.next(self.end)
        

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    ClassificationFlow()
