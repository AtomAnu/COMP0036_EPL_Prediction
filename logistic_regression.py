import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

df_seeds = pd.read_csv('/Users/Manny/Documents/GitHub/COMP0036_EPL_Prediction/COMP0036_Brief/DataFiles/epl-training.csv')
df_seeds.head()

df_seeds_2 = df_seeds.dropna(axis=1, how='any')
df_types = df_seeds_2.columns.tolist()
print(df_types)

df_teams = df_seeds_2.drop_duplicates(subset = 'HomeTeam', keep = 'first')
df_teams = df_teams.HomeTeam.tolist()
print(df_teams)

class Team():
    def __init__(self, name, data):
        self.name = name
        self.data = data
    
    def printT(self):
        print(self.name)
        print(self.data)
        
teamList = []
for i in range(len(df_teams)):
    count = 0
    data = pd.DataFrame(columns = df_types)
    for j, k in zip(df_seeds_2.HomeTeam, df_seeds_2.AwayTeam):
        if j == df_teams[i] or k == df_teams[i]:
            data = data.append(df_seeds_2.loc[count], ignore_index = True)
        count += 1
    data.loc[data['FTR'] == 'H', 'FTR'] = 2
    data.loc[data['FTR'] == 'D', 'FTR'] = 1
    data.loc[data['FTR'] == 'A', 'FTR'] = 0
    team = Team(df_teams[i], data[['HomeTeam', 'AwayTeam', 'FTR']])
    teamList.append(team)

x_train = []
x_test = []
y_train = []
y_test = []

for i in range(len(df_teams)):
    final = pd.get_dummies(teamList[i].data, prefix = ['HomeTeam', 'AwayTeam'], columns = ['HomeTeam', 'AwayTeam'])
    x = final.drop(['FTR'], axis = 1)
    y = final['FTR']
    y = y.astype('int')
    x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(x, y, test_size = 0.20, random_state = 42)
    x_train.append(x_train_temp)
    x_test.append(x_test_temp)
    y_train.append(y_train_temp)
    y_test.append(y_test_temp)

logreg = []

for i in range(len(df_teams)):
    logreg_temp = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
    logreg_temp.fit(x_train[i], y_train[i])
    logreg.append(logreg_temp)

score = logreg[15].score(x_train[15], y_train[15])
score_2 = logreg[15].score(x_test[15], y_test[15])

print(score)
print(score_2)

df_test_2 = df_test.drop(['Date'], axis = 1)

yy = []

for i in range(len(df_test_2.HomeTeam.tolist())):
    for j in range(len(df_teams)):
        if df_teams[j] == df_test_2.iloc[i, 0]:
            print(j)
#             test_data = pd.get_dummies(df_test_2.iloc[i], prefix = ['HomeTeam', 'AwayTeam'], columns = ['HomeTeam', 'AwayTeam'])
#             yy_temp = logreg[j].predict(df_test_2.iloc[i])
#             yy.append(yy_temp)
