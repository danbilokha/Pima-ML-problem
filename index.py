import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

diabetes_dataset = pd.read_csv('data/diabetes.csv')

# print(diabets_dataset.head())
# print(diabets_dataset.describe())

diabetes_dataset_outcome = diabetes_dataset.groupby('Outcome')
# print(diabets_dataset_outcome)

train, test = train_test_split(diabetes_dataset, test_size=0.10, random_state=0, stratify=diabetes_dataset['Outcome'])

train_X = train[train.columns[:8]]
train_Y = train['Outcome']

test_X = test[test.columns[:8]]
test_Y = test['Outcome']

# print(train_X)
# print(test_X)

model = LogisticRegression(max_iter=10000)
model.fit(train_X, train_Y)

prediction = model.predict(test_X)
print('Prediction accuracy: %s' % metrics.accuracy_score(test_Y, prediction))

confusion_matrix = metrics.confusion_matrix(test_Y, prediction)
# print(confusion_matrix)

plt.matshow(confusion_matrix, cmap='Pastel1')

for x in range(0, 2):
    for y in range(0, 2):
        plt.text(x, y, confusion_matrix[x, y])

plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()

TP = confusion_matrix[1, 1]
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]

print("Sensitivity: %.4f" % (TP / float(TP + FN)))
print("Specificy  : %.4f" % (TN / float(TN + FP)))

########

from sklearn.metrics import roc_curve, auc

save_predictions_proba = model.predict_proba(test_X)[:, 1]  # column 1

plt.hist(save_predictions_proba, bins=10)
plt.xlim(0, 1)  # x-axis limit from 0 to 1
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')
plt.show()

# function roc_curve
# input: IMPORTANT: first argument is true values, second argument is predicted probabilities
#                   we do not use y_pred_class, because it will give incorrect results without
#                   generating an error
# output: FPR, TPR, thresholds
# FPR: false positive rate
# TPR: true positive rate
FPR, TPR, thresholds = roc_curve(test_Y, save_predictions_proba)

plt.figure(figsize=(10, 5))  # figsize in inches
plt.plot(FPR, TPR)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 50%
plt.plot(FPR, TPR, lw=2, label='Logaristic Regression (AUC = %0.2f)' % auc(FPR, TPR))
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()

# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize

# it will return 1 for all values above 0.3 and 0 otherwise
# results are 2D so we slice out the first column
prediction2 = binarize(save_predictions_proba.reshape(-1, 1), 0.3)  # [0]

confusion_matrix2 = metrics.confusion_matrix(test_Y, prediction2)
TP = confusion_matrix2[1, 1]
TN = confusion_matrix2[0, 0]
FP = confusion_matrix2[0, 1]
FN = confusion_matrix2[1, 0]

print("new Sensitivity: %.4f" % (TP / float(TP + FN)))
print("new Specificy  : %.4f" % (TN / float(TN + FP)))

from sklearn.metrics import roc_curve, auc

# function roc_curve
# input: IMPORTANT: first argument is true values, second argument is predicted probabilities
#                   we do not use y_pred_class, because it will give incorrect results without
#                   generating an error
# output: FPR, TPR, thresholds
# FPR: false positive rate
# TPR: true positive rate
FPR, TPR, thresholds = roc_curve(test_Y, save_predictions_proba)

plt.figure(figsize=(10, 5))  # figsize in inches
plt.plot(FPR, TPR)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 50%
plt.plot(FPR, TPR, lw=2, label='Logaristic Regression (AUC = %0.2f)' % auc(FPR, TPR))
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.legend(loc="lower right")


def evaluate_threshold(threshold):
    print("Sensitivity: %.4f" % (TPR[thresholds > threshold][-1]))
    print("Specificy  : %.4f" % (1 - FPR[thresholds > threshold][-1]))


print('Threshold = 0.5')
evaluate_threshold(0.5)
print('Threshold = 0.35')
evaluate_threshold(0.35)
print('Threshold = 0.2')
evaluate_threshold(0.2)

spec = []
sens = []
thres = []

threshold = 0.00
for x in range(0, 90):
    thres.append(threshold)
    sens.append(TPR[thresholds > threshold][-1])
    spec.append(1 - FPR[thresholds > threshold][-1])
    threshold += 0.01

plt.plot(thres, sens, lw=2, label='Sensitivity')
plt.plot(thres, spec, lw=2, label='Specificity')
ax = plt.gca()
ax.set_xticks(np.arange(0, 1, 0.1))
ax.set_yticks(np.arange(0, 1, 0.1))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('Sensitivity vs. Specificity')
plt.xlabel('Threshold')
plt.grid(True)
plt.legend(loc="center right")
plt.show()

# class MachineLearningAlgs:
#     def get_logistic_regression_prediction(self):
#         model = LogisticRegression(max_iter=10000)
#         model.fit(self.train_X, self.train_Y)
#
#         prediction = model.predict(self.test_X)
#         print('LogisticRegression prediction accuracy: %s' % metrics.accuracy_score(self.test_Y, prediction))
#
#         return prediction
#
#     @staticmethod
#     def confusion_matrix(test_result, prediction):
#         confusion_matrix = metrics.confusion_matrix(test_result, prediction)
#         # print(confusion_matrix)
#
#         plt.figure()
#         plt.matshow(confusion_matrix, cmap='Pastel1')
#
#         for x in range(0, 2):
#             for y in range(0, 2):
#                 plt.text(x, y, confusion_matrix[x, y])
#
#         plt.ylabel('expected label')
#         plt.xlabel('predicted label')
#         plt.show()
#
#         return confusion_matrix
#
#     @staticmethod
#     def classification_metrics(confusion_matrix, used_ML = 'Unknown'):
#         TP = confusion_matrix[1, 1]
#         TN = confusion_matrix[0, 0]
#         FP = confusion_matrix[0, 1]
#         FN = confusion_matrix[1, 0]
#
#         print(used_ML + " sensitivity: %.4f" % (TP / float(TP + FN)))
#         print(used_ML + " specificy  : %.4f" % (TN / float(TN + FP)))
#
#     def __init__(self, dataset_path='data/diabetes.csv'):
#         self.dataset_path = dataset_path
#         self._load_dataset()
#         self._split_dataset_train_test()
#
#     def _load_dataset(self):
#         self.diabetes_dataset = pd.read_csv(self.dataset_path)
#
#     def _split_dataset_train_test(self):
#         train, test = train_test_split(self.diabetes_dataset, test_size=0.15, random_state=0,
#                                        stratify=self.diabetes_dataset['Outcome'])
#
#         self.train_X = train[train.columns[:8]]
#         self.train_Y = train['Outcome']
#
#         self.test_X = test[test.columns[:8]]
#         self.test_Y = test['Outcome']
#
#
# machineLearning = MachineLearningAlgs('data/diabetes.csv')
#
# machineLearning_LR_prediction = machineLearning.get_logistic_regression_prediction()
# machineLearning_LR_confusion_matrix = machineLearning.confusion_matrix(machineLearning.test_Y,
#                                                                        machineLearning_LR_prediction)
# machineLearning.classification_metrics(machineLearning_LR_confusion_matrix, 'LogisticRegression')
