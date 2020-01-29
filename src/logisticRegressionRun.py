from src.algorithms.LogisticRegression import logistic_regression_predict, \
    logistic_regression_predict_with_shifted_threshold, \
    logistic_regression_fit
from src.metrics.classification import classification_metrics
from src.metrics.confusion_matrix import confusion_matrix
from src.prepareData import PreparedDataset


def run():
    prepared_dataset = PreparedDataset('data/diabetes.csv')
    logistic_regression_trained = logistic_regression_fit(prepared_dataset)

    logistic_regression_accuracy, logistic_regression_prediction = logistic_regression_predict(
        logistic_regression_trained,
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
