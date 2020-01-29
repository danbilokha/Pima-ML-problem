import matplotlib.pyplot as plt
from sklearn import metrics


def confusion_matrix(prediction, prediction_for_results):
    confusion_matrix = metrics.confusion_matrix(prediction_for_results, prediction)

    plt.matshow(confusion_matrix, cmap='Pastel1')

    for x in range(0, 2):
        for y in range(0, 2):
            plt.text(x, y, confusion_matrix[x, y])

    plt.ylabel('expected label')
    plt.xlabel('predicted label')
    plt.show()

    return confusion_matrix
