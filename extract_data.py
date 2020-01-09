import os
import pandas as pd

def load_file(file_name):
    """
    Load .csv file using pd.read_csv
    :param file_name: string containing the file name
    :return: extracted and stripped data in the DataFrame type
    """

    #find the path of the wanted file
    file_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(file_dir):
        for name in files:
            if name == file_name:
                file_path = os.path.join(root, name)

    #load the wanted file using pd.read_csv
    loaded_data = pd.read_csv(file_path)

    #remove blank columns
    data_columns = list(loaded_data)
    for col in data_columns:
        if 'Unnamed' not in col:
            last_col_idx = data_columns.index(col)
    data_stripped = loaded_data.iloc[:, :last_col_idx + 1]

    return data_stripped

def format_data(training_file_name,test_file_name,additional_file_name=None):
    """
    Format the loaded training and test data. Date column is formatted. Columns with categorical features are converted
    to multiple columns with the binary format.
    :param training_file_name: string containing the training file name
    :param test_file_name: string containing the test file name
    :return: formatted training and test data in the DataFrame type
    """

    #load training and test files
    training_data_stripped = load_file(training_file_name)
    test_data_stripped = load_file(test_file_name)
    if additional_file_name is not None:
        additional_file_data_stripped = load_file(additional_file_name)
        all_training_data = pd.concat([training_data_stripped,
                                       additional_file_data_stripped], sort=False, ignore_index=True)
    else:
        all_training_data = training_data_stripped

    all_training_data['FTR'] = all_training_data['FTR'].map({'H': 1, 'D': 2, 'A': 3})
    all_training_data['HTR'] = all_training_data['HTR'].map({'H': 1, 'D': 2, 'A': 3})

    #convert all columns categorical features into multiple columns with the binary format
    #training and test data are combined
    training_and_test_binary = pd.get_dummies(pd.concat([all_training_data.drop(columns=['Date']),
                                                         test_data_stripped.drop(columns=['Date'])], sort=False))

    #extract the training data
    training_binary = training_and_test_binary.iloc[:len(all_training_data),:]

    #extract the test data
    test_binary = training_and_test_binary.iloc[len(all_training_data):,:]
    test_binary_columns = [col for col in training_and_test_binary
                           for original_col in list(test_data_stripped) if col.startswith(original_col)]
    test_binary = test_binary[test_binary_columns]


    #format the Date column
    training_date_formatted = pd.DataFrame(pd.to_datetime(all_training_data.iloc[:,0]))
    test_date_formatted = pd.DataFrame(pd.to_datetime(test_data_stripped.iloc[:,0]))

    #combine the Date column with the rest
    training_data_formatted = pd.concat([training_date_formatted, training_binary], axis=1)
    test_data_formatted = pd.concat([test_date_formatted, test_binary], axis=1)

    return training_data_formatted, test_data_formatted

def get_twenty_latest_team_matches(team_name, data_frame, row_idx):
    """
    Find the latest twenty matches for the specified team.
    -> If the specified row index for the current match is not included
       in the row indices of all the matches, the latest 20 matches of
       all the matches would be returned
    -> If the number of previous matches
       is less than 20, the method would return all previous matches.

    :param team_name: specified team name (e.g. 'HomeTeam_Arsenal', 'AwayTeam_Cardiff', etc.)
    :param data_frame: DataFrame containing all the matches for all the teams
    :param row_index: row index of the current match of interest
    :return: DataFrame with the latest 20 matches
    """

    row_idx_list = []
    for row in range(data_frame.shape[0]):
        if data_frame.loc[row, team_name] == 1:
            row_idx_list.append(row)

    if row_idx not in row_idx_list:
        twenty_matches_row_idx_list = row_idx_list[len(row_idx_list)-20:]
    else:
        number_of_previous_matches = len(row_idx_list[:row_idx_list.index(row_idx)])
        if number_of_previous_matches < 20:
            twenty_matches_row_idx_list = row_idx_list[:row_idx_list.index(row_idx)]
        else:
            twenty_matches_row_idx_list = row_idx_list[row_idx_list.index(row_idx)-20:row_idx_list.index(row_idx)]

    return data_frame.iloc[twenty_matches_row_idx_list]

def non_shot_feature_selection(training_data, team_ratings_data):
    """
    Select features that have correlation values above the defined threshold.

    :param training_data: DataFrame containing the training data
    :param team_ratings_data: Data containing the ratings of the home and away teams at each match
    :return: an array containing names of the selected feature
    """

    data = pd.concat([training_data, team_ratings_data], axis=1, ignore_index=False)
    column_heads_to_drop = ['HomeTeam', 'AwayTeam', 'Date', 'Referee', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS',
                            'HST', 'AST','HTR']
    all_columns_to_drop = [col for col in data
                           for chosen_col in column_heads_to_drop if col.startswith(chosen_col)]

    data = data.drop(columns=all_columns_to_drop)

    corrmat = data.corr()
    FTR_corr = corrmat.loc['FTR'].abs().drop('FTR')
    feature_corr_thershold = 0.05
    selected_features_idx = [row for row in range(FTR_corr.shape[0]) if FTR_corr[row] >= feature_corr_thershold]
    selected_features = FTR_corr[selected_features_idx].index.values

    return selected_features

def extract_test_data(test_data, selected_features, ratings_data, ratings_history):
    test_result = pd.DataFrame(columns=[selected_features])
    print(test_result)
    team_list_h = [col for col in test_data
                               if col.startswith('HomeTeam')]
    team_list_a = [col for col in test_data
                               if col.startswith('AwayTeam')]
    h_list = [col for col in selected_features if col.startswith('H')]
    a_list = [col for col in selected_features if col.startswith('A')]
    for i in range(len(test_data)):
        test = pd.DataFrame(columns=[selected_features])
        for team in team_list_h:
            if test_data.loc[i, team] == 1:
                print(team)
                home_team_twenty_results = get_twenty_latest_team_matches(
                    team, pd.concat([data, ratings_data], axis=1), -1)
                home_team_twenty_results = home_team_twenty_results[h_list]
                for name in h_list:
                    if name == 'HomeRatings':
                        team_used = team.replace('HomeTeam_', '')
                        print(team_used)
                        idx = ratings_history[ratings_history['Team']
                                              == team_used].index.values
                        print(idx)
                        test.loc[i, name] = ratings_history.loc[idx[0], 'Ratings']
                    else:
                        test.loc[i, name] = home_team_twenty_results[name].mean()
        for team in team_list_a:
            if test_data.loc[i, team] == 1:
                away_team_twenty_results = get_twenty_latest_team_matches(
                    team, pd.concat([data, ratings_data], axis=1), -1)
                away_team_twenty_results = away_team_twenty_results[a_list]
                for name in a_list:
                    if name == 'AwayRatings':
                        team_used = team.replace('AwayTeam_', '')
                        idx = ratings_history[ratings_history['Team']
                                              == team_used].index.values
                        test.loc[i, name] = ratings_history.loc[idx[0], 'Ratings']
                    else:
                        test.loc[i, name] = away_team_twenty_results[name].mean()
        print(test)
        test_result = test_result.append(test)
    print(test_result)
    return test_result

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

from team_ratings import Ratings

data, test = format_data('epl-training.csv','epl-test.csv','data_updated.csv')
# ratings = Ratings(data)
# ratings_data = ratings.result
# ratings_history = ratings.rating
# training_data = pd.concat(
#     [data, ratings_data], axis=1, ignore_index=False)
# training_data.to_csv('/Users/Manny/Desktop/Data.csv')
# ratings_history.to_csv('/Users/Manny/Desktop/Data_2.csv')
ratings_data = load_file('Data.csv')
ratings_data = ratings_data[['HomeRatings','AwayRatings']]
ratings_history = load_file('Data_2.csv')
print(ratings_history)
selected_features = non_shot_feature_selection(data, ratings_data)
print(selected_features)
pred_data = extract_test_data(test, selected_features, ratings_data, ratings_history)

from models_parser_2 import Compare

models_comparison_obj = Compare(pd.concat([data,ratings_data],axis=1)[selected_features],data['FTR'])

from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
"""
Final Prediction to be completed
"""
best_model = models_comparison_obj.best_model
print(pred_data)
if models_comparison_obj.best_idx == 4:
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
    best_model.model.fit(pd.concat([data, ratings_data], axis=1)[selected_features], to_categorical(
        data['FTR'])[:, [1, 2, 3]], batch_size=64, verbose=0, epochs=60, callbacks=[es])
    y_r = best_model.model.predict(pred_data)
else:
    best_model.fit(pd.concat([data, ratings_data], axis=1)[selected_features], data['FTR'])
    y_r = best_model.predict(pred_data)

print('The final prediction is:')
print(y_r)
result = pd.DataFrame({'FTR_Predicted': y_r})
print(result)
result = result['FTR_Predicted'].map({1: 'H', 2: 'D', 3: 'A'})
final_team = load_file('epl-test.csv')
print(final_team)
final = pd.concat([final_team, result], axis=1)
print(final)
final.to_csv('/Users/Manny/Desktop/Prediction.csv')
