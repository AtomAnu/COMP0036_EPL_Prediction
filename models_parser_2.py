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

from neural_network import Neural_Network
from sklearn.model_selection import StratifiedKFold

from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

from mlxtend.plotting import plot_confusion_matrix as plot_confusion_matrix_2

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
        self.best_models = []
        self.best_idx = self.get_best_idx()
        self.best_model = self.get_best_model()

    def update_accuracies(self, model_name, accuracy):
        add_data = pd.Series({'Models': model_name, 'Accuracies': accuracy})
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
        self.best_models.append(lr)
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

        grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, cv=5)

        # fitting the model for grid search
        grid.fit(self.data.X_all, self.data.y_all)
        # print best parameter after tuning
        print(grid.best_params_)
        # print how our model looks after hyper-parameter tuning
        gsvm = grid.best_estimator_
        self.best_models.append(gsvm)

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
        self.best_models.append(gnb)
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
        param_grid = {'n_neighbors': range(1,101)}

        grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=2, cv=5)

        # fitting the model for grid search
        grid.fit(self.data.X_all, self.data.y_all)
        # print best parameter after tuning
        print(grid.best_params_)
        # print how our model looks after hyper-parameter tuning
        knn = grid.best_estimator_
        self.best_models.append(knn)
        # knn = KNeighborsClassifier()
        # self.best_models.append(knn)

        self.classifier_train(knn, self.data.X_train, self.data.y_train)
        f1, acc = self.classifier_score(
            knn, self.data.X_test, self.data.y_test)
        self.print_result(f1, acc, 'testing')
        f1, acc = self.classifier_score(
            knn, self.data.X_train, self.data.y_train)
        self.print_result(f1, acc, 'training')

        self.get_cross_val_score(
            knn, 'k-Nearest Neighbor', self.data.X_all, self.data.y_all)
    
    def tryNN(self):
        def to_result(data):
            result = []
            for i in range(len(data)):
                d_max = np.amax(data[i])
                if data[i][0] == d_max:
                    result.append(1)
                elif data[i][1] == d_max:
                    result.append(2)
                else:
                    result.append(3)
            return result
        
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(self.data.X_all, self.data.y_all,
                                                                test_size=0.2,
                                                                random_state=100)
        nn = Neural_Network(self.data.X_all.shape[1], len(self.data.y_all.unique()))
        es = EarlyStopping(monitor='val_accuracy',
                           mode='max', verbose=1, patience=50)
        train = nn.model.fit(X_train_1, to_categorical(y_train_1)[:, [1, 2, 3]], batch_size=64,
                                  epochs=500, verbose=0,
                             validation_data=(X_test_1, to_categorical(y_test_1)[:, [1, 2, 3]]), callbacks=[es])
        print(train.history.keys())
        result = nn.model.evaluate(
            self.data.X_test, to_categorical(self.data.y_test)[:, [1, 2, 3]])
        print("==================================")
        print("Result on the testing set")
        print("F1 score value: " + str(result[1] * 100))
        print("Accuracy: " + str(result[2]))
        print("==================================\n")

        plt.plot(train.history['accuracy'])
        plt.plot(train.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.show()
        # summarize history for loss
        plt.plot(train.history['loss'])
        plt.plot(train.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

        y_pred = nn.model.predict(self.data.X_test)
        nn_cm = confusion_matrix(
            self.data.y_test, to_result(y_pred), labels=[1, 2, 3])
        print(nn_cm)
        fig, ax = plot_confusion_matrix_2(conf_mat=nn_cm,
                                        colorbar=True,
                                        show_absolute=False,
                                        show_normed=True)
        # plot confusion matrix
        plt.show()
        # print(s)
        es_2 = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
        #because we are using cross validation, no val set is used, so we early stop using the training
        #loss metrics,but we set the epochs to 150, because the loss function turns to be stable
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
        cross_val = []
        for train, test in kfold.split(self.data.X_all, self.data.y_all):
            nn_temp = Neural_Network(
                self.data.X_all.shape[1], len(self.data.y_all.unique()))
            #standardised input
            train_history = nn_temp.model.fit(self.data.X_all.loc[train], to_categorical(
                self.data.y_all)[:, [1, 2, 3]][train],
                                            batch_size=64, verbose=0, epochs=60,
                                            callbacks=[es_2])
            val_history = nn_temp.model.evaluate(
                self.data.X_all.loc[test], to_categorical(self.data.y_all)[:, [1, 2, 3]][test], verbose = 0)
            print("%s: %.2f%%" % (nn_temp.model.metrics[0], val_history[1]*100))
            cross_val.append(val_history[1])
        print("%.2f%% (+/- %.2f%%)" %
            (np.average(cross_val)*100, np.std(cross_val)*100))
        self.update_accuracies('Neural Network', np.average(cross_val))
        
    def get_best_idx(self):
        self.tryLR()
        self.trySVM()
        self.tryGNB()
        self.trykNN()
        self.tryNN()
        print(self.accuracies)
        models = ['LR', 'SVM', 'GNB', 'KNN', 'NN']
        # models = ['GNB', 'KNN']
        plt.bar(models, np.array(self.accuracies['Accuracies']))
        plt.title('Accuracy Comparison')
        plt.xlabel('model')
        plt.ylabel('accuracy')
        plt.show()
        best_idx = np.argmax(np.array(self.accuracies['Accuracies']))
        return best_idx

    def get_best_model(self):
        if self.best_idx == 4:
            return Neural_Network(self.data.X_all.shape[1], len(self.data.y_all.unique()))
        else:
            return self.best_models[self.best_idx]

