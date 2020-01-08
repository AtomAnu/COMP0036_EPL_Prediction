from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from time import time

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split

from neural_network import Metrics, Neural_Network

class Data:
    def __init__(self, X, y):
        self.X_all = X
        self.y_all = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=100)

class Compare:
    def __init__(self, X, y):
        self.accuracies = pd.DataFrame(columns=['Models', 'Accuracies'])
        self.data = Data(X, y)

    def update_accuracies(self, model_name, accuracy):
        add_data = pd.Series({model_name: accuracy})
        self.accuracies = self.accuracies.append(add_data, ignore_index=True)

    def classifier_train(self, classifier, X, y):
        start = time()
        classifier.fit(X, y)
        end = time()
        print("Training time: {}".format(end - start))

    def classifier_score(self, classifier, X, y):
        y_pred = classifier.predict(X)
        return f1_score(y, y_pred, pos_label=1, average='macro'), sum(y == y_pred) / float(len(y_pred))

    def print_result(self, f1, acc, label):
        print("==================================")
        print("Result on the " + label + " set")
        print("F1 score value: " + str(f1))
        print("Accuracy: " + str(acc))
        print("==================================\n")

    def get_cross_val_score(self, classifier, name, X, y):
        start = time()
        scores = cross_val_score(classifier, X, y, cv=5)
        self.update_accuracies(name, sum(scores)/5)
        print("==================================")
        print("Mean of cross-validated scores for testing set:", sum(scores)/5)
        print("The standard deviation of the scores", np.std(scores))
        end = time()
        print("Evaluating time {:}".format(end - start))

    def plot_confusion(self, classifier, class_names, X_test, y_test, title):
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize='true')
        disp.ax_.set_title(title)
        plt.show()

    def tryLR(self):
        # defining parameter range
        param_grid = {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}

        grid = GridSearchCV(LogisticRegression(multi_class="ovr", solver="sag",
                                               max_iter=4000),
                                               param_grid,
                                               refit=True, verbose=2, cv=5)

        start = time()
        # fitting the model for grid search
        grid.fit(self.data.X_all, self.data.y_all)
        # print best parameter after tuning
        print(grid.best_params_)
        # print how our model looks after hyper-parameter tuning
        lr = grid.best_estimator_
        end = time()
        print(end - start)

        self.classifier_train(lr, self.data.X_train, self.data.y_train)
        f1, acc = self.classifier_score(lr, self.data.X_test, self.data.y_test)
        self.print_result(f1, acc, 'testing')
        f1, acc = self.classifier_score(lr, self.data.X_train, self.data.y_train)
        self.print_result(f1, acc, 'training')

        self.get_cross_val_score(lr, 'Logistic Regression', self.data.X_all, self.data.y_all)

        class_name = ['A', 'D', 'H']
        title = "Confusion matrix with normalization"
        self.plot_confusion(lr, class_name, self.data.X_test, self.data.y_test, title)

    def trySVM(self):
        # defining parameter range
        param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['linear', 'rbf', 'sigmoid']}

        grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3, cv=5)

        # fitting the model for grid search
        grid.fit(self.data.X_all, self.data.y_all)
        # print best parameter after tuning
        print(grid.best_params_)

        # print how our model looks after hyper-parameter tuning
        gsvm = grid.best_estimator_

        self.classifier_train(gsvm, self.data.X_train, self.data.y_train)
        f1, acc = self.classifier_score(gsvm, self.data.X_test, self.data.y_test)
        self.print_result(f1, acc, 'testing')
        f1, acc = self.classifier_score(
            gsvm, self.data.X_train, self.data.y_train)
        self.print_result(f1, acc, 'training')

        self.get_cross_val_score(
            gsvm, 'Support Vector Machine', self.data.X_all, self.data.y_all)

        class_name = ['A', 'D', 'H']
        title = "Confusion matrix with normalization"
        self.plot_confusion(gsvm, class_name, self.data.X_test,
                            self.data.y_test, title)
        
    def tryGNB(self):
        gnb = GaussianNB()
        self.classifier_train(gnb, self.data.X_train, self.data.y_train)
        f1, acc = self.classifier_score(
            gnb, self.data.X_test, self.data.y_test)
        self.print_result(f1, acc, 'testing')
        f1, acc = self.classifier_score(
            gnb, self.data.X_train, self.data.y_train)
        self.print_result(f1, acc, 'training')

        self.get_cross_val_score(
            gnb, 'Guassian Naive Bayesian', self.data.X_all, self.data.y_all)

    def trykNN(self):
        knn = KNeighborsClassifier()
        self.classifier_train(knn, self.data.X_train, self.data.y_train)
        f1, acc = self.classifier_score(
            knn, self.data.X_test, self.data.y_test)
        self.print_result(f1, acc, 'testing')
        f1, acc = self.classifier_score(
            knn, self.data.X_train, self.data.y_train)
        self.print_result(f1, acc, 'training')

        self.get_cross_val_score(
            knn, 'k-Nearest Neighbor', self.data.X_all, self.data.y_all)
    
    def tryNN(self, f1):
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(self.data.X_all, self.data.y_all,
                                                                test_size=0.2,
                                                                random_state=100)
        nn = Neural_Network(self.data.X_all.shape[1], self.data.y_all.shape[1])
        es = EarlyStopping(monitor='val_accuracy',
                           mode='max', verbose=1, patience=50)
        train = nn.model.fit(X_train_1, y_train_1, batch_size=64,
                                  epochs=500, verbose=0,
                                  validation_data=(X_test_1, y_test_1), callbacks=[es])
        print(train.history.keys())
        result = nn.model.evaluate(self.data.X_test, self.data.y_test)
        print("==================================")
        print("Result on the testing set")
        print("F1 score value: " + str(result[1] * 100))
        print("Accuracy: " + str(result[2]))
        print("==================================\n")

        y_pred = nn.model.predict(self.data.X_test)
        nn_confusion_matrix = confusion_matrix(
            self.data.y_test, y_pred, labels=['H', 'D', 'A'])
        # plot confusion matrix
        
