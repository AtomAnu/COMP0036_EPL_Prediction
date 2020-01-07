from extract_data import format_data

from models_parser import compare_regression_models, compare_classification_models

from team_ratings import Ratings

import numpy as np
import pandas as pd

training_file_name = 'epl-training.csv'
test_file_name = 'epl-test.csv'
additional_file_name = 'data_updated.csv'
training_data, test_data = format_data(training_file_name, test_file_name, additional_file_name)
# training_data, test_data = format_data(training_file_name, test_file_name)

team_rating = Ratings(training_data)
# print(team_rating.rating)

training_data = pd.merge(training_data, team_rating)

group_8_y = ['FTHG']
group_8_list = ['FTR', 'HF','HY','HR','FTHG']
group_8_binary_list = [col for col in training_data
                        for chosen_col in group_8_list if col.startswith(chosen_col)]
group_8_data = training_data[group_8_binary_list]

all_ratings = []
all_teams = []
# home_team_feature_name = ['HomeTeam']
feature_names = ['HomeTeam', 'AwayTeam']
for feature_name in feature_names:
    # team_list = [col for col in training_data
    #                     for chosen_col in feature_name if col.startswith(chosen_col)]
    team_list = [col for col in training_data
                         if col.startswith(feature_name)]
    team_used = []
    team_obj_list = []
    final_team_rating = []
    team_ratings = []
    for team in team_list:
        row_idx_list = []
        for row in range(training_data.shape[0]):
            if training_data.loc[row,team] == 1:
                row_idx_list.append(row)

        if len(row_idx_list)==0:
            print('Data for {} is EMPTY'.format(team))
            continue
        
        team_used.append(team)
        # compare_regression_models(team, group_8_data.iloc[row_idx_list].drop(columns=group_8_y), group_8_data[group_8_y].iloc[row_idx_list])
        final_prediction = compare_classification_models(team, group_8_data.iloc[row_idx_list].drop(columns=group_8_y), group_8_data[group_8_y].iloc[row_idx_list])
        # print(final_prediction)
        rating = 1500
        result = group_8_data[group_8_y].iloc[row_idx_list]
        result.index = range(len(result))
        for i in range(0, len(final_prediction) - 1):
            rating = rating + 10 * (result.loc[i] - final_prediction[i])
        
        team_ratings.append(rating)
        # for i in range(0, len(final_prediction)):
        #     final_team_rating[i] = sum(final_prediction[:i+1])/len(final_prediction[:i+1])

        # team_ratings.append(final_team_rating)
    all_ratings.append(np.array(team_ratings).ravel())
    all_teams.append(team_used)

all_scores = pd.DataFrame({'Team_1': all_teams[0], 'Team_2': all_teams[1], 'HomeTeam_rating': all_ratings[0], 'AwayTeam_rating': all_ratings[1]})
print(all_scores)
