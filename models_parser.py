from team_parser import Team

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import LeaveOneOut

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compare_regression_models(team_name, X_data, y_data):

    degrees = range(1,15)
    alphas = [1e-20, 1e-15, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 5, 20, 1000]

    loo = LeaveOneOut()

    regressions = {
        'Ridge Regression': Ridge(alpha=alphas[0], max_iter=1e5),
        'Lasso': Lasso(alpha=alphas[0], max_iter=1e5),
        'Linear Regression': LinearRegression()
    }

    final_score = []
    final_prediction = []
    y_pred = []
    r_name = []

    for index, (name, regression) in enumerate(regressions.items()):
        val_errors_degree = []
        val_errors_alpha = []

        for degree in degrees:
            team_obj = Team(team_name, X_data, y_data, degree=degree)
            errors = np.sum(-cross_val_score(regression, team_obj.X_train, y=team_obj.y_train,
                                             scoring='neg_mean_squared_error',
                                             cv=10,
                                             n_jobs=-1))
            val_errors_degree.append(np.sqrt(errors))

        best_degree = degrees[np.argmin(val_errors_degree)]

        team_obj = Team(team_name, X_data, y_data, degree=best_degree)

        if index == 2:
            team_obj_reg = Team(team_name, X_data, y_data,
                                  degree=best_degree)
            # regression.fit(team_obj_reg.X_train,
            #                team_obj_reg.y_train)
            
            for train_idx, test_idx in loo.split(team_obj_reg.X_all):
                X_all_temp = team_obj_reg.X_all
                y_temp = team_obj_reg.y
                y_temp.index = range(len(y_temp))
                regression.fit(X_all_temp[train_idx], y_temp.loc[train_idx])
                y_pred.append(regression.predict(X_all_temp[test_idx]))
            
            final_prediction.append(y_pred)
            # y_pred = regression.predict(team_obj_reg.X_test)
            # final_prediction.append(y_pred)
            score_temp = regression.score(
                team_obj_reg.X_test, team_obj_reg.y_test)
            final_score.append(score_temp)
            r_name.append(name + ' (degree: ' + str(best_degree) + ')')
            print(print('{} Score: {}'.format(name, score_temp)))
            continue

        for alpha in alphas:
            regression.set_params(alpha=alpha)

            errors = np.sum(-cross_val_score(regression, team_obj.X_train, y=team_obj.y_train,
                                             scoring='neg_mean_squared_error',
                                             cv=10,
                                             n_jobs=-1))
            val_errors_alpha.append(np.sqrt(errors))

        best_alpha = alphas[np.argmin(val_errors_alpha)]

        if index == 0:
            team_obj_reg = Team(team_name, X_data, y_data,
                                degree=best_degree)
            regression.set_params(alpha=best_alpha)
            # regression.fit(team_obj_reg.X_train,
            #                team_obj_reg.y_train)
            
            for train_idx, test_idx in loo.split(team_obj_reg.X_all):
                X_all_temp = team_obj_reg.X_all
                y_temp = team_obj_reg.y
                y_temp.index = range(len(y_temp))
                regression.fit(X_all_temp[train_idx], y_temp.loc[train_idx])
                y_pred.append(regression.predict(X_all_temp[test_idx]))

            final_prediction.append(y_pred)

            # y_pred = regression.predict(team_obj.X_test)
            # final_prediction.append(y_pred)
            score_temp = regression.score(
                team_obj_reg.X_test, team_obj_reg.y_test)
            final_score.append(score_temp)
            r_name.append(name + ' (degree: ' + str(best_degree) +
                          ')' + ' (alpha: ' + str(best_alpha) + ')')
            print(print('{} Score: {}'.format(name, score_temp)))

        if index == 1:
            team_obj_reg = Team(team_name, X_data, y_data,
                                degree=best_degree)
            regression.set_params(alpha=best_alpha)
            # regression.fit(team_obj_reg.X_train,
            #                team_obj_reg.y_train)
            
            for train_idx, test_idx in loo.split(team_obj_reg.X_all):
                X_all_temp = team_obj_reg.X_all
                y_temp = team_obj_reg.y
                y_temp.index = range(len(y_temp))
                regression.fit(X_all_temp[train_idx], y_temp.loc[train_idx])
                y_pred.append(regression.predict(X_all_temp[test_idx]))

            final_prediction.append(y_pred)

            # y_pred = regression.predict(team_obj.X_test)
            # final_prediction.append(y_pred)
            score_temp = regression.score(
                team_obj_reg.X_test, team_obj_reg.y_test)
            final_score.append(score_temp)
            r_name.append(name + ' (degree: ' + str(best_degree) +
                          ')' + ' (alpha: ' + str(best_alpha) + ')')
            print(print('{} Score: {}'.format(name, score_temp)))
    
    plt.bar(x=r_name, height=final_score, label='Score',
            color='steelblue', alpha=0.8)
    
    for x, y in enumerate(final_score):
        plt.text(x, y + 0.0001, '%s' % y, ha='center', va='bottom')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title(team_name)
    # plt.show()

    return final_prediction[np.argmax(final_score)]

def compare_classification_models(team_name, X_data, y_data):

    if team_name.startswith('HomeTeam'):
        col_temp = 'ExpectedHGoals'
    elif team_name.startswith('AwayTeam'):
        col_temp = 'ExpectedAGoals'

    index_temp = X_data.index.values
    print(index_temp)

    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Gaussian Naive Bayes': GaussianNB(),
        'k-Nearest Neighbor': KNeighborsClassifier(),
        'Support Vector Machine': SVC(gamma='auto')
    }

    loo = LeaveOneOut()

    team_obj = Team(team_name, X_data, y_data)
    X_all_temp = team_obj.X_all
    y_temp = team_obj.y
    y_temp.index = range(len(y_temp))
    final_prediction = []
    final_score =[]
 
    r_name = []

    for index, (name, classifier) in enumerate(classifiers.items()):
        y_pred = pd.DataFrame(columns=[col_temp], index=index_temp)
        print(y_pred)
        for train_idx, test_idx in loo.split(team_obj.X_all):
            classifier.fit(X_all_temp[train_idx], y_temp.loc[train_idx])
            data_temp = classifier.predict(X_all_temp[test_idx])
            # add_data = pd.Series(
            #     {col_temp: data_temp[0]})
            # y_pred = y_pred.append(add_data, ignore_index=True)
            y_pred.loc[index_temp[test_idx], col_temp] = data_temp[0]

        # final_prediction.append(np.array(y_pred).ravel())
        print(y_pred)

        final_prediction.append(y_pred)
        # classifier.fit(team_obj.X_train, team_obj.y_train)
        # y_pred = classifier.predict(team_obj.X_test)
        # final_prediction.append(y_pred)
        score_temp = classifier.score(team_obj.X_test, team_obj.y_test)
        final_score.append(score_temp)
        r_name.append(name)
        print('{} Score: {}'.format(name, score_temp))
    
    plt.bar(x=r_name, height=final_score, label='Score',
            color='steelblue', alpha=0.8)
    
    for x, y in enumerate(final_score):
        plt.text(x, y + 0.0001, '%s' % y, ha='center', va='bottom')

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title(team_name)
    # plt.show()

    return final_prediction[np.argmax(final_score)]
