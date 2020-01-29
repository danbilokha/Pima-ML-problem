from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize


def logistic_regression_fit(prepared_dataset):
    model = LogisticRegression(max_iter=10000)
    model.fit(prepared_dataset.train_X, prepared_dataset.train_Y)

    return model


def logistic_regression_predict(model, prepared_dataset):
    prediction = model.predict(prepared_dataset.test_X)
    accuracy = metrics.accuracy_score(prepared_dataset.test_Y, prediction)

    return accuracy, prediction


def logistic_regression_predict_with_shifted_threshold(model, prepared_dataset):
    predictions_proba = model.predict_proba(prepared_dataset.test_X)[:, 1]
    prediction = binarize(predictions_proba.reshape(-1, 1), 0.33)
    accuracy = metrics.accuracy_score(prepared_dataset.test_Y, prediction)

    return accuracy, prediction
