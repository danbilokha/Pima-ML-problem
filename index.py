import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as Scaler

from algorithms.LogisticRegression import logistic_regression_predict, \
    logistic_regression_predict_with_shifted_threshold, \
    logistic_regression_fit
from metrics.classification import classification_metrics
from metrics.confusion_matrix import confusion_matrix


class PreparedDataset:
    def data_cleaning(self):
        # Calculate the median value for BMI
        median_bmi = self.diabetes_dataset['BMI'].median()
        # Substitute it in the BMI column of the
        # dataset where values are 0
        self.diabetes_dataset['BMI'] = self.diabetes_dataset['BMI'].replace(
            to_replace=0, value=median_bmi)

        # Calculate the median value for BloodPressure
        median_bloodp = self.diabetes_dataset['BloodPressure'].median()
        # Substitute it in the BloodP column of the
        # self.diabetes_dataset where values are 0
        self.diabetes_dataset['BloodPressure'] = self.diabetes_dataset['BloodPressure'].replace(
            to_replace=0, value=median_bloodp)

        # Calculate the median value for Glucose
        median_plglcconc = self.diabetes_dataset['Glucose'].median()
        # Substitute it in the PlGlcConc column of the
        # self.diabetes_dataset where values are 0
        self.diabetes_dataset['Glucose'] = self.diabetes_dataset['Glucose'].replace(
            to_replace=0, value=median_plglcconc)

        # Calculate the median value for SkinThickness
        median_skinthick = self.diabetes_dataset['SkinThickness'].median()
        # Substitute it in the SkinThickness column of the
        # self.diabetes_dataset where values are 0
        self.diabetes_dataset['SkinThickness'] = self.diabetes_dataset['SkinThickness'].replace(
            to_replace=0, value=median_skinthick)

        # Calculate the median value for TwoHourSerIns
        median_twohourserins = self.diabetes_dataset['Insulin'].median()
        # Substitute it in the TwoHourSerIns column of the
        # self.diabetes_dataset where values are 0
        self.diabetes_dataset['Insulin'] = self.diabetes_dataset['Insulin'].replace(
            to_replace=0, value=median_twohourserins)

    def feature_scaling(self):
        scaler = Scaler()
        scaler.fit(self.train_dataset)

        # print(self.train_dataset, self.test_dataset)

        self.train_set_scaled = pd.DataFrame(scaler.transform(self.train_dataset), index=self.train_dataset.index,
                                             columns=self.train_dataset.columns)
        self.test_set_scaled = pd.DataFrame(scaler.transform(self.test_dataset), index=self.test_dataset.index,
                     columns=self.test_dataset.columns)

        self.train_X = self.train_set_scaled[self.train_set_scaled.columns[:8]]
        self.train_Y = self.train_set_scaled['Outcome']

        self.test_X = self.test_set_scaled[self.test_set_scaled.columns[:8]]
        self.test_Y = self.test_set_scaled['Outcome']

    def __init__(self, diabetes_dataset_path='data/diabetes.csv'):
        self.dataset_path = diabetes_dataset_path
        self._load_dataset()
        self._split_dataset_train_test()

    def _load_dataset(self):
        self.diabetes_dataset = pd.read_csv(self.dataset_path)

    def _split_dataset_train_test(self):
        train, test = train_test_split(self.diabetes_dataset, test_size=0.20, random_state=0,
                                       stratify=self.diabetes_dataset['Outcome'])

        self.train_dataset = train
        self.test_dataset = test

        self.train_X = train[train.columns[:8]]
        self.train_Y = train['Outcome']

        self.test_X = test[test.columns[:8]]
        self.test_Y = test['Outcome']


prepared_dataset = PreparedDataset('data/diabetes.csv')
logistic_regression_trained = logistic_regression_fit(prepared_dataset)

logistic_regression_accuracy, logistic_regression_prediction = logistic_regression_predict(logistic_regression_trained,
                                                                                           prepared_dataset)

logistic_regression_confusion_matrix = confusion_matrix(logistic_regression_prediction, prepared_dataset.test_Y)
logistic_regression_sensivity, logistic_regression_specificy = classification_metrics(
    logistic_regression_confusion_matrix)

print('LogisticRegression, accuracy:{accuracy}; sensivity:{sensivity}, specificy:{specificy}'.format(
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

print('\n========================================>>> Apply data cleaning')
prepared_dataset.data_cleaning()
logistic_regression_trained3 = logistic_regression_fit(prepared_dataset)

logistic_regression_accuracy3, logistic_regression_prediction3 = logistic_regression_predict(
    logistic_regression_trained3,
    prepared_dataset)

logistic_regression_confusion_matrix3 = confusion_matrix(logistic_regression_prediction3, prepared_dataset.test_Y)
logistic_regression_sensivity3, logistic_regression_specificy3 = classification_metrics(
    logistic_regression_confusion_matrix3)

print('LogisticRegression, accuracy:{accuracy}; sensivity:{sensivity}, specificy:{specificy}'.format(
    accuracy=logistic_regression_accuracy3, sensivity=logistic_regression_sensivity3,
    specificy=logistic_regression_specificy3))

logistic_regression_accuracy4, logistic_regression_prediction4 = logistic_regression_predict_with_shifted_threshold(
    logistic_regression_trained3,
    prepared_dataset)

logistic_regression_confusion_matrix4 = confusion_matrix(logistic_regression_prediction4, prepared_dataset.test_Y)
logistic_regression_sensivity4, logistic_regression_specificy4 = classification_metrics(
    logistic_regression_confusion_matrix4)

print('LogisticRegression, accuracy:{accuracy}; sensivity:{sensivity}, specificy:{specificy}'.format(
    accuracy=logistic_regression_accuracy4, sensivity=logistic_regression_sensivity4,
    specificy=logistic_regression_specificy4))

print('\n========================================>>> Apply data cleaning & feature scaling')
prepared_dataset.feature_scaling()
logistic_regression_trained5 = logistic_regression_fit(prepared_dataset)

logistic_regression_accuracy5, logistic_regression_prediction5 = logistic_regression_predict(
    logistic_regression_trained5,
    prepared_dataset)

logistic_regression_confusion_matrix5 = confusion_matrix(logistic_regression_prediction5, prepared_dataset.test_Y)
logistic_regression_sensivity5, logistic_regression_specificy5 = classification_metrics(
    logistic_regression_confusion_matrix5)

print('LogisticRegression, accuracy:{accuracy}; sensivity:{sensivity}, specificy:{specificy}'.format(
    accuracy=logistic_regression_accuracy5, sensivity=logistic_regression_sensivity5,
    specificy=logistic_regression_specificy5))

logistic_regression_accuracy6, logistic_regression_prediction6 = logistic_regression_predict_with_shifted_threshold(
    logistic_regression_trained5,
    prepared_dataset)

logistic_regression_confusion_matrix6 = confusion_matrix(logistic_regression_prediction6, prepared_dataset.test_Y)
logistic_regression_sensivity6, logistic_regression_specificy6 = classification_metrics(
    logistic_regression_confusion_matrix6)

print('LogisticRegression, accuracy:{accuracy}; sensivity:{sensivity}, specificy:{specificy}'.format(
    accuracy=logistic_regression_accuracy6, sensivity=logistic_regression_sensivity6,
    specificy=logistic_regression_specificy6))
