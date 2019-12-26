import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# different model with different parameters
classifiers = {
    'L2 logistic (Multinomial)': LogisticRegression(solver='lbfgs', multi_class='auto'),
    'k-nearest neighbor': KNeighborsClassifier(),
    'ridge regression': Ridge(alpha = 1.0)
}

def get_best_model(X, y):
    # n_classifiers = len(classifiers)
    x_data = []
    y_data = []
    classifier_f = []
    max_a = 0
    count = 0
    result = 0
    for index, (name, classifier) in enumerate(classifiers.items()):
        if name == 'ridge regression':
            max_r = 0
            count_2 = 0
            n_alphas = 200
            result_2 = 0
            alphas = np.logspace(-10, -2, n_alphas)
            classifier_r = []
            for a in alphas:
                classifier = classifier.set_params(alpha = a)
                classifier.fit(X, y)
                y_pred = classifier.predict(X)
                accuracy = classifier.score(X, y)
                classifier_r.append(classifier)
                if accuracy >= max_r:
                    max_r = accuracy
                    result_2 = count_2
                count_2 += 1
            accuracy = max_r
            x_data.append(name + ' ' + str(a))
            y_data.append(accuracy)
            classifier_f.append(classifier_r[result_2])
            print("Accuracy (train) for %s: %0.1f%% " % (name + ' ' + str(a), accuracy * 100))
        else:
            classifier.fit(X, y)
            y_pred = classifier.predict(X)
            accuracy = accuracy_score(y, y_pred)
            x_data.append(name)
            y_data.append(accuracy)
            classifier_f.append(classifier)
            print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        if accuracy > max_a:
            max_a = accuracy
            result = count
        # # View probabilities:
        # probas = classifier.predict_proba(X)
        # print("Probability for the prediction of %s:" % (name))
        # print(probas)
        count += 1

    plt.bar(x=x_data, height=y_data, label='Accuracy',
            color='steelblue', alpha=0.8)
    for x, y in enumerate(y_data):
        plt.text(x, y + 0.001, '%s' % y, ha='center', va='bottom')

    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.show()

    return classifier_f[result]
