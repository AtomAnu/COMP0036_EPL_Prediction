#Import models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#Import error metric
from sklearn.metrics import mean_squared_error

from extract_data import format_data
from team_parser import Team

import matplotlib.pyplot as plt

training_file_name = 'epl-training.csv'
test_file_name = 'epl-test.csv'
training_data, test_data = format_data(training_file_name, test_file_name)

group_8_y = ['FTHG']
group_8_list = ['HF','HY','HR','FTHG']
group_8_binary_list = [col for col in training_data
                        for chosen_col in group_8_list if col.startswith(chosen_col)]
group_8_data = training_data[group_8_binary_list]

home_team_feature_name = ['HomeTeam']
home_team_list = [col for col in training_data
                    for chosen_col in home_team_feature_name if col.startswith(chosen_col)]

team_obj_list = []
for team in home_team_list:
    row_idx_list = []
    for row in range(training_data.shape[0]):
        if training_data.loc[row,team] == 1:
            row_idx_list.append(row)

    degrees = range(1,30)
    alpha_ridge = [1e-20, 1e-15, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 5, 20, 1000]
    mse = []
    for degree in degrees:
        team_obj = Team(team, group_8_data.drop(columns=group_8_y), group_8_data[group_8_y], degree=degree)
        # reg = LinearRegression()
        # reg = Lasso(alpha=alpha_ridge[-1],max_iter=1e5)
        reg = Ridge(alpha=alpha_ridge[-1],max_iter=1e5)
        reg.fit(team_obj.X_train, team_obj.y_train)
        y_pred = reg.predict(team_obj.X_test)

        # print(mean_squared_error(team_obj.y_test, y_pred))
        # print(reg.coef_)
        mse.append(mean_squared_error(team_obj.y_test,y_pred))
        # team_obj_list.append(team_obj)
