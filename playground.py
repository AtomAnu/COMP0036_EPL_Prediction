from extract_data import format_data

from models_parser import compare_regression_models, compare_classification_models
<<<<<<< HEAD
=======

import numpy as np
>>>>>>> 8d4975e202c14f3ab4fea7df02706ce7e10856d6

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
team_ratings = []
for team in home_team_list:
    row_idx_list = []
    for row in range(training_data.shape[0]):
        if training_data.loc[row,team] == 1:
            row_idx_list.append(row)

    if len(row_idx_list)==0:
        print('Data for {} is EMPTY'.format(team))
        continue

<<<<<<< HEAD
    compare_classification_models(team, group_8_data.iloc[row_idx_list].drop(columns=group_8_y), group_8_data[group_8_y].iloc[row_idx_list])
=======
    # compare_regression_models(team, group_8_data.iloc[row_idx_list].drop(columns=group_8_y), group_8_data[group_8_y].iloc[row_idx_list])
    final_prediction = compare_classification_models(team, group_8_data.iloc[row_idx_list].drop(columns=group_8_y), group_8_data[group_8_y].iloc[row_idx_list])

    team_ratings.append(final_prediction)
print(team_ratings)
>>>>>>> 8d4975e202c14f3ab4fea7df02706ce7e10856d6
