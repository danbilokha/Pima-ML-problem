import pandas as pd
from sklearn.model_selection import train_test_split

from algorithms.LogisticRegression import logistic_regression_predict, \
    logistic_regression_predict_with_shifted_threshold, \
    logistic_regression_trained
from metrics.classification import classification_metrics
from metrics.confusion_matrix import confusion_matrix


class PreparedDataset:
    def __init__(self, dataset_path='data/diabetes.csv'):
        self.dataset_path = dataset_path
        self._load_dataset()
        self._split_dataset_train_test()

    def _load_dataset(self):
        self.diabetes_dataset = pd.read_csv(self.dataset_path)

    def _split_dataset_train_test(self):
        train, test = train_test_split(self.diabetes_dataset, test_size=0.20, random_state=0,
                                       stratify=self.diabetes_dataset['Outcome'])

        self.train_X = train[train.columns[:8]]
        self.train_Y = train['Outcome']

        self.test_X = test[test.columns[:8]]
        self.test_Y = test['Outcome']


prepared_dataset = PreparedDataset('data/diabetes.csv')
logistic_regression_trained = logistic_regression_trained(prepared_dataset)

logistic_regression_accuracy, logistic_regression_prediction = logistic_regression_predict(logistic_regression_trained,
                                                                                           prepared_dataset)

logistic_regression_confusion_matrix = confusion_matrix(logistic_regression_prediction, prepared_dataset.test_Y)
logistic_regression_sensivity, logistic_regression_specificy = classification_metrics(
    logistic_regression_confusion_matrix)

print('LogisticRegression, accuracy:{accuracy}; sensiKvity:{sensivity}, specificy:{specificy}'.format(
    accuracy=logistic_regression_accuracy, sensivity=logistic_regression_sensivity,
    specificy=logistic_regression_specificy))

logistic_regression_accuracy2, logistic_regression_prediction2 = logistic_regression_predict_with_shifted_threshold(
    logistic_regression_trained,
    prepared_dataset)

logistic_regression_confusion_matrix2 = confusion_matrix(logistic_regression_prediction2, prepared_dataset.test_Y)
logistic_regression_sensivity2, logistic_regression_specificy2 = classification_metrics(
    logistic_regression_confusion_matrix2)

print('LogisticRegression, accuracy:{accuracy}; sensivity:{sensivity}, specificy:{specificy}'.format(
    accuracy=logistic_regression_accuracy2, sensivity=logistic_regression_sensivity2,
    specificy=logistic_regression_specificy2))