import os
from dateutil.parser import parse

file_dir = os.path.dirname(os.path.abspath(__file__))

training_file_name = 'epl-training.csv'

for root, dirs, files in os.walk(file_dir):
    for name in files:
        if name==training_file_name:
            data_file_path = os.path.join(root, name)

# print(data_file_path)

#open the .csv file

training_data_file = open(data_file_path, 'r')
training_data = training_data_file.readlines()

training_data_matrix = []
for line in range(0,len(training_data)):
    training_data_matrix.append((training_data[line].split(',,'))[0].split(','))

def is_date(string, fuzzy=False):
    """
    This module check if the input string is convertible to a date format or not.

    :param string: input string
    :param fuzzy: boolean that can be set to True to ignore all tokens in the string

    :return: True if the string is convertible to date and False if vice versa
    """
    try:
        parse(string, fuzzy)
        return True
    except ValueError:
        return False

"""
List of all the features available in the training file.
['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
"""

categorical_features_col_num = []
for cell in training_data_matrix[1]:
    if cell.isdigit() is False:
        if is_date(cell)==False:
            starting_idx = -1
            while True:
                try:
                    idx = training_data_matrix[1].index(cell, starting_idx + 1)
                except ValueError:
                    break
                else:
                    if not idx in categorical_features_col_num:
                        categorical_features_col_num.append(idx)
                    starting_idx = idx

categorical_feature_names = []
for idx in categorical_features_col_num:
    exec(training_data_matrix[0][idx]+'=[]')
    categorical_feature_names.append(training_data_matrix[0][idx])

# list of all the categorical feature names
print('Categorical Feature Names:')
print(categorical_feature_names)
print('\n')

for row in training_data_matrix[1:]:
    for idx in categorical_features_col_num:
        exec('list='+training_data_matrix[0][idx])
        exec('if row[idx] not in list: '+training_data_matrix[0][idx]+'.append(row[idx])')

for feature in categorical_feature_names:
    exec('print("{0}: {1}".format(feature,'+feature+'))')
    print('\n')


