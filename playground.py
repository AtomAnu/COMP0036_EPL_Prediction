from extract_data import format_data

from models_parser import compare_regression_models, compare_classification_models

from team_ratings import Ratings

from team_tilts import Tilts

import numpy as np
import pandas as pd

training_file_name = 'epl-training.csv'
test_file_name = 'epl-test.csv'
additional_file_name = 'data_updated.csv'
training_data, test_data = format_data(training_file_name, test_file_name, additional_file_name)
# training_data, test_data = format_data(training_file_name, test_file_name)

# group_8_y = ['FTHG']
# group_8_list = ['HF', 'HY', 'HR', 'FTHG']
# group_8_binary_list = [col for col in training_data
#                         for chosen_col in group_8_list if col.startswith(chosen_col)]
# group_8_data = training_data[group_8_binary_list]
team_rating = Ratings(training_data)
# training_data = pd.concat(
#         [training_data, team_rating.result], axis=1, ignore_index=False)

print(s+'s')
# training_data = pd.read_csv('/Users/Manny/Desktop/Data.csv')
# print(training_data)
# team_rating = Ratings(training_data)
# training_data = pd.concat(
#     [training_data, team_rating.result], axis=1, ignore_index=False)
# training_data.to_csv('/Users/Manny/Desktop/Data_2.csv')

# print(s+'s')

all_ratings = []
all_teams = []
result_h = pd.DataFrame(columns=['ExpectedHGoals'])
result_a = pd.DataFrame(columns=['ExpectedAGoals'])
# home_team_feature_name = ['HomeTeam']
feature_names = ['HomeTeam', 'AwayTeam']
for feature_name in feature_names:
    # team_list = [col for col in training_data
    #                     for chosen_col in feature_name if col.startswith(chosen_col)]
    if feature_name == 'HomeTeam':
        group_8_y = ['FTHG']
        group_8_list = ['HF', 'HY', 'HR', 'FTHG']
    elif feature_name == 'AwayTeam':
        group_8_y = ['FTAG']
        group_8_list = ['AF', 'AY', 'AR', 'FTAG']
    group_8_binary_list = [col for col in training_data
                        for chosen_col in group_8_list if col.startswith(chosen_col)]
    group_8_data = training_data[group_8_binary_list]

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
        if team.startswith('HomeTeam'):
            result_h = pd.concat([result_h, final_prediction])
        elif team.startswith('AwayTeam'):
            result_a = pd.concat([result_a, final_prediction])
        # training_data = pd.concat([training_data, final_prediction], ignore_index=False)
        # print(training_data)

        # print(final_prediction)
        # rating = 1500
        # result = group_8_data[group_8_y].iloc[row_idx_list]
        # result.index = range(len(result))
        # for i in range(0, len(final_prediction) - 1):
        #     rating = rating + 10 * (result.loc[i] - final_prediction[i])
        
        # team_ratings.append(rating)
        # for i in range(0, len(final_prediction)):
        #     final_team_rating[i] = sum(final_prediction[:i+1])/len(final_prediction[:i+1])
        # team_ratings.append(final_team_rating)
#     all_ratings.append(np.array(team_ratings).ravel())
#     all_teams.append(team_used)
print(result_h.sort_index())
print(result_a.sort_index())
training_data = pd.concat([training_data, result_h.sort_index(), result_a.sort_index()], axis=1, ignore_index=False)
print(training_data)
team_tilt = Tilts(training_data)
print(team_tilt.result)
training_data = pd.concat([training_data, team_tilt.result], axis=1, ignore_index=False)
training_data.to_csv('/Users/Manny/Desktop/Data.csv')
team_rating = Ratings(training_data)
print(team_rating.result)
training_data = pd.concat([training_data, team_rating.result], axis=1, ignore_index=False)
training_data.to_csv('/Users/Manny/Desktop/Data_2.csv')
print(training_data)
# all_scores = pd.DataFrame({'Team_1': all_teams[0], 'Team_2': all_teams[1], 'HomeTeam_rating': all_ratings[0], 'AwayTeam_rating': all_ratings[1]})
# print(all_scores)
