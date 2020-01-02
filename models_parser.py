from team_parser import Team

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge, Lasso, LinearRegression

import numpy as np

def compare_regression_models(team_name, X_data, y_data):

    degrees = range(1,15)
    alphas = [1e-20, 1e-15, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 5, 20, 1000]

    reg_1 = Ridge(alpha=alphas[0], max_iter=1e5)
    reg_2 = Lasso(alpha=alphas[0], max_iter=1e5)
    reg_3 = LinearRegression()

    models = [reg_1,reg_2,reg_3]

    val_errors_degree = []
    val_errors_alpha = []

    for model in models:
        for degree in degrees:
            team_obj = Team(team_name, X_data, y_data, degree=degree)
            errors = np.sum(-cross_val_score(model, team_obj.X_train, y=team_obj.y_train,
                                     scoring='neg_mean_squared_error',
                                     cv=10,
                                     n_jobs=-1))
            val_errors_degree.append(np.sqrt(errors))

        best_degree = degrees[np.argmin(val_errors_degree)]

        team_obj = Team(team_name, X_data, y_data, degree=best_degree)

        if models.index(model) == 2:
            linear_reg_best_degree = best_degree
            continue

        for alpha in alphas:
            model.set_params(alpha=alpha)

            errors = np.sum(-cross_val_score(model, team_obj.X_train, y=team_obj.y_train,
                                     scoring='neg_mean_squared_error',
                                     cv=10,
                                     n_jobs=-1))
            val_errors_alpha.append(np.sqrt(errors))

        best_alpha = alphas[np.argmin(val_errors_alpha)]

        if models.index(model) == 0:
            ridge_best_degree = best_degree
            ridge_best_alpha = best_alpha

        if models.index(model) == 1:
            lasso_best_degree = best_degree
            lasso_best_alpha = best_alpha

    team_obj_linear_reg = Team(team_name, X_data, y_data, degree=linear_reg_best_degree)
    linear_reg_best_model = LinearRegression()
    linear_reg_best_model.fit(team_obj_linear_reg.X_train, team_obj_linear_reg.y_train)
    linear_reg_score = linear_reg_best_model.score(team_obj_linear_reg.X_test, team_obj_linear_reg.y_test)

    team_obj_ridge = Team(team_name, X_data, y_data, degree=ridge_best_degree)
    ridge_best_model = Ridge(alpha=ridge_best_alpha)
    ridge_best_model.fit(team_obj_ridge.X_train, team_obj_ridge.y_train)
    ridge_score = ridge_best_model.score(team_obj_ridge.X_test, team_obj_ridge.y_test)

    team_obj_lasso = Team(team_name, X_data, y_data, degree=lasso_best_degree)
    lasso_best_model = Lasso(alpha=lasso_best_alpha)
    lasso_best_model.fit(team_obj_lasso.X_train, team_obj_lasso.y_train)
    lasso_score =lasso_best_model.score(team_obj_lasso.X_test, team_obj_lasso.y_test)

    print('Linear Regression Score: {}'.format(linear_reg_score))
    print('Ridge Regression Score: {}'.format(ridge_score))
    print('Lasso Score: {}'.format(lasso_score))

# def compare_classification_models(team_name, X_data, y_data):
