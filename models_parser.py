from team_parser import Team

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import numpy as np

import matplotlib.pyplot as plt

def compare_regression_models(team_name, X_data, y_data):

    degrees = range(1,15)
    alphas = [1e-20, 1e-15, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 5, 20, 1000]

    # reg_ridge = Ridge(alpha=alphas[0], max_iter=1e5)
    # reg_lasso = Lasso(alpha=alphas[0], max_iter=1e5)
    # reg_lr = LinearRegression()

    # models = [reg_ridge,reg_lasso,reg_lr]

    # for model in models:
    #     val_errors_degree = []
    #     val_errors_alpha = []

    #     for degree in degrees:
    #         team_obj = Team(team_name, X_data, y_data, degree=degree)
    #         errors = np.sum(-cross_val_score(model, team_obj.X_train, y=team_obj.y_train,
    #                                  scoring='neg_mean_squared_error',
    #                                  cv=10,
    #                                  n_jobs=-1))
    #         val_errors_degree.append(np.sqrt(errors))

    #     best_degree = degrees[np.argmin(val_errors_degree)]

    #     team_obj = Team(team_name, X_data, y_data, degree=best_degree)

    #     if models.index(model) == 2:
    #         linear_reg_best_degree = best_degree
    #         continue

    #     for alpha in alphas:
    #         model.set_params(alpha=alpha)

    #         errors = np.sum(-cross_val_score(model, team_obj.X_train, y=team_obj.y_train,
    #                                  scoring='neg_mean_squared_error',
    #                                  cv=10,
    #                                  n_jobs=-1))
    #         val_errors_alpha.append(np.sqrt(errors))

    #     best_alpha = alphas[np.argmin(val_errors_alpha)]

    #     if models.index(model) == 0:
    #         ridge_best_degree = best_degree
    #         ridge_best_alpha = best_alpha

    #     if models.index(model) == 1:
    #         lasso_best_degree = best_degree
    #         lasso_best_alpha = best_alpha

    # team_obj_ridge = Team(team_name, X_data, y_data, degree=ridge_best_degree)
    # ridge_best_model = Ridge(alpha=ridge_best_alpha)
    # ridge_best_model.fit(team_obj_ridge.X_train, team_obj_ridge.y_train)
    # ridge_score = ridge_best_model.score(team_obj_ridge.X_test, team_obj_ridge.y_test)
    # print(ridge_best_degree)
    # print(ridge_best_alpha)

    # team_obj_lasso = Team(team_name, X_data, y_data, degree=lasso_best_degree)
    # lasso_best_model = Lasso(alpha=lasso_best_alpha)
    # lasso_best_model.fit(team_obj_lasso.X_train, team_obj_lasso.y_train)
    # lasso_score =lasso_best_model.score(team_obj_lasso.X_test, team_obj_lasso.y_test)
    # print(lasso_best_degree)
    # print(lasso_best_alpha)

    # team_obj_linear_reg = Team(team_name, X_data, y_data, degree=linear_reg_best_degree)
    # linear_reg_best_model = LinearRegression()
    # linear_reg_best_model.fit(team_obj_linear_reg.X_train, team_obj_linear_reg.y_train)
    # linear_reg_score = linear_reg_best_model.score(team_obj_linear_reg.X_test, team_obj_linear_reg.y_test)
    # print(linear_reg_best_degree)

    # print('Ridge Regression Score: {}'.format(ridge_score))
    # print('Lasso Score: {}'.format(lasso_score))
    # print('Linear Regression Score: {}'.format(linear_reg_score))

    regressions = {
        'Ridge Regression': Ridge(alpha=alphas[0], max_iter=1e5),
        'Lasso': Lasso(alpha=alphas[0], max_iter=1e5),
        'Linear Regression': LinearRegression()
    }

    final_score = []
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
            regression = LinearRegression()
            regression.fit(team_obj_reg.X_train,
                           team_obj_reg.y_train)
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
            regression = Ridge(alpha=best_alpha)
            regression.fit(team_obj_reg.X_train,
                           team_obj_reg.y_train)
            score_temp = regression.score(
                team_obj_reg.X_test, team_obj_reg.y_test)
            final_score.append(score_temp)
            r_name.append(name + ' (degree: ' + str(best_degree) +
                          ')' + ' (alpha: ' + str(best_alpha) + ')')
            print(print('{} Score: {}'.format(name, score_temp)))

        if index == 1:
            team_obj_reg = Team(team_name, X_data, y_data,
                                degree=best_degree)
            regression = Lasso(alpha=best_alpha)
            regression.fit(team_obj_reg.X_train,
                           team_obj_reg.y_train)
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
    plt.show()

def compare_classification_models(team_name, X_data, y_data):

    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Gaussian Naive Bayes': GaussianNB(),
        'k-Nearest Neighbor': KNeighborsClassifier(),
        'Support Vector Machine': SVC(gamma='auto')
    }

    team_obj = Team(team_name, X_data, y_data)
    final_score = []
    r_name = []
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(team_obj.X_train, team_obj.y_train)
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
    plt.show()

    # team_obj = Team(team_name, X_data, y_data)

    # logistic_reg = LogisticRegression()
    # logistic_reg.fit(team_obj.X_train, team_obj.y_train)
    # logistic_reg_score = logistic_reg.score(team_obj.X_test, team_obj.y_test)

    # gaussian_NB = GaussianNB()
    # gaussian_NB.fit(team_obj.X_train, team_obj.y_train)
    # gaussian_NB_score = gaussian_NB.score(team_obj.X_test, team_obj.y_test)

    # k_nearest_neighbor = KNeighborsClassifier()
    # k_nearest_neighbor.fit(team_obj.X_train, team_obj.y_train)
    # k_nearest_neighbor_score = k_nearest_neighbor.score(team_obj.X_test, team_obj.y_test)

    # svm = SVC(gamma='auto')
    # svm.fit(team_obj.X_train, team_obj.y_train)
    # svm_score = svm.score(team_obj.X_test, team_obj.y_test)

    # print('Logistic Regression Score: {}'.format(logistic_reg_score))
    # print('Gaussian Naive Bayes Score: {}'.format(gaussian_NB_score))
    # print('k-Nearest Neighbor Score: {}'.format(k_nearest_neighbor_score))
    # print('Support Vector Machine: {}'.format(svm_score))
