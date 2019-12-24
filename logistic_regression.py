import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

df_seeds = pd.read_csv('/Users/Manny/Desktop/DataFiles/epl-training.csv') # import training data
df_seeds.head()
df_test = pd.read_csv('/Users/Manny/Desktop/DataFiles/epl-test.csv') # import testing data
df_test.head()

df_seeds_2 = df_seeds.dropna(axis=1, how='any') # remove useless data
df_types = df_seeds_2.columns.tolist()

df_teams = df_seeds_2.drop_duplicates(subset='HomeTeam', keep='first') # find each team in the data
df_teams = df_teams.HomeTeam.tolist()

# a class for each team containing the name and the match history
class Team():
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def printT(self):
        print(self.name)
        print(self.data)


teamList = []
for i in range(len(df_teams)):
    count_1 = 0
    count_2 = 0
    data = pd.DataFrame(columns=df_types)
    # add training data to the corresponding team class
    for j, k in zip(df_seeds_2.HomeTeam, df_seeds_2.AwayTeam):
        if j == df_teams[i] or k == df_teams[i]:
            data = data.append(
                df_seeds_2.loc[count_1], ignore_index=True, sort=False)
        count_1 += 1
    data.loc[data['FTR'] == 'H', 'FTR'] = 2
    data.loc[data['FTR'] == 'D', 'FTR'] = 1
    data.loc[data['FTR'] == 'A', 'FTR'] = 0
    # add testing data
    for m, n in zip(df_test.HomeTeam, df_test.AwayTeam):
        if m == df_teams[i] or n == df_teams[i]:
            data = data.append(
                df_test.loc[count_2], ignore_index=True, sort=False)
        count_2 += 1
    data = data.fillna(3) # label testing data result as 3 for convinience
    team = Team(df_teams[i], data[['HomeTeam', 'AwayTeam', 'FTR']])
    teamList.append(team)

x_train = []
x_test = []
y_train = []
y_test = []
test_data = []

for i in range(len(df_teams)):
    # create dummies variable to seperate characteristic variables
    final = pd.get_dummies(teamList[i].data, prefix=[
                           'HomeTeam', 'AwayTeam'], columns=['HomeTeam', 'AwayTeam'])
    test_data_temp = final[final['FTR'] == 3] # get the testing data
    test_data.append(test_data_temp.drop(['FTR'], axis=1))
    final = final[~(final['FTR'] == 3)]
    x = final.drop(['FTR'], axis=1)
    y = final['FTR']
    y = y.astype('int')
    x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(
        x, y, test_size=0.20, random_state=42) # cross validation
    x_train.append(x_train_temp)
    x_test.append(x_test_temp)
    y_train.append(y_train_temp)
    y_test.append(y_test_temp)

logreg = []

for i in range(len(df_teams)):
    logreg_temp = LogisticRegression(solver='lbfgs', multi_class='auto') # create the Logistic Regression model
    logreg_temp.fit(x_train[i], y_train[i]) # train the model
    logreg.append(logreg_temp)

yy = []
yy_prob = []

for i in range(len(test_data)):
    if not test_data[i].empty:
        prediction = logreg[i].predict(test_data[i]) # predict the result
        yy.append(prediction)
        prob = logreg[i].predict_proba(test_data[i]) # get the probabilities for each result
        yy_prob.append(prob)
