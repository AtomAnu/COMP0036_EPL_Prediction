import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# different model with different parameters
classifiers = {
    'L2 logistic (Multinomial)': LogisticRegression(solver='lbfgs', multi_class='auto'),
    'k-nearest neighbor': KNeighborsClassifier()
}


def get_accuracy(X, y):
    # n_classifiers = len(classifiers)
    x_data = []
    y_data = []
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X, y)
        y_pred = classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        x_data.append(name)
        y_data.append(accuracy)
        # print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

        # # View probabilities:
        # probas = classifier.predict_proba(X)
        # print("Probability for the prediction of %s:" % (name))
        # print(probas)

    plt.bar(x=x_data, height=y_data, label='Accuracy',
            color='steelblue', alpha=0.8)
    for x, y in enumerate(y_data):
        plt.text(x, y + 0.01, '%s' % y, ha='center', va='bottom')

    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.show()
