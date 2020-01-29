def classification_metrics(confusion_matrix):
    tp = confusion_matrix[1, 1]
    tn = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]

    sensivity = (tp / float(tp + fn))
    specificy = (tn / float(tn + fp))

    # print(used_ML + " sensitivity: %.4f" % (tp / float(tp + fn)))
    # print(used_ML + " specificy  : %.4f" % (tn / float(tn + fp)))

    return sensivity, specificy
