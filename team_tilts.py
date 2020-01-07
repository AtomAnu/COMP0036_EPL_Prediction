import numpy as np
import pandas as pd

class Tilts:
    def __init__(self, X):
        self.home_team_list = [col for col in X
                               if col.startswith('HomeTeam')]
        h_team_used = []
        team_used = []
        for team in self.home_team_list:
            row_idx_list = []
            for row in range(X.shape[0]):
                if X.loc[row, team] == 1:
                    row_idx_list.append(row)

            if len(row_idx_list) == 0:
                print('Data for {} is EMPTY'.format(team))
                continue
            h_team_used.append(team)
            team_used.append(team.replace('HomeTeam_', ''))

        self.home_team_list = h_team_used
        self.team_list = team_used
        self.away_team_list = [col for col in X
                               if col.startswith('AwayTeam')]
        a_team_used = []
        for team in self.away_team_list:
            row_idx_list = []
            for row in range(X.shape[0]):
                if X.loc[row, team] == 1:
                    row_idx_list.append(row)

            if len(row_idx_list) == 0:
                print('Data for {} is EMPTY'.format(team))
                continue

            a_team_used.append(team)
        self.away_team_list = a_team_used
        self.h_goals = X['FTHG']
        self.a_goals = X['FTAG']
        self.expected_h_goals = X['ExpectedHGoals']
        self.expected_a_goals = X['ExpectedAGoals']
        print(type(self.h_goals))
        print(self.expected_h_goals)
        self.tilt = self.initialize()
        print(self.tilt)
        self.result = self.update_tilts(X)
        print(self.result)

    def initialize(self):
        tilts = pd.DataFrame(
            {'Team': self.team_list})
        tilts['Tilts'] = 1
        return tilts

    def update_tilts(self, X):
        match_tilts = pd.DataFrame(columns=['HomeTilts', 'AwayTilts'])
        for i in range(len(X)):
            tilt_h_team = 0
            tilt_a_team = 0
            for team_h in self.home_team_list:
                for team_a in self.away_team_list:
                    if X.loc[i, team_h] == 1 and X.loc[i, team_a] == 1:
                        tilt_h_team = self.tilt[self.tilt['Team'].isin(
                            [team_h.replace('HomeTeam_', '')])].index.values
                        tilt_h_team = tilt_h_team[0]
                        tilt_a_team = self.tilt[self.tilt['Team'].isin(
                            [team_a.replace('AwayTeam_', '')])].index.values
                        tilt_a_team = tilt_a_team[0]
            # if self.expected_h_goals.loc[i]+self.expected_a_goals.loc[i] == 0:
            #     print(s+'s')
            delta_h = self.compute_score(
                self.tilt.loc[tilt_a_team, 'Tilts'], self.h_goals.loc[i]+self.a_goals.loc[i], self.expected_h_goals.loc[i]+self.expected_a_goals.loc[i])
            # delta_a = self.compute_score(
            #     self.tilt.loc[tilt_h_team, 'Tilts'], self.a_goals.loc[i], self.expected_a_goals.loc[i])
            self.tilt.loc[tilt_h_team, 'Tilts'] = 0.98 * \
                self.tilt.loc[tilt_h_team, 'Tilts'] + delta_h
            self.tilt.loc[tilt_a_team, 'Tilts'] = 0.98 * \
                self.tilt.loc[tilt_a_team, 'Tilts'] + delta_h
            add_data = pd.Series(
                {'HomeTilts': self.tilt.loc[tilt_h_team, 'Tilts'], 'AwayTilts': self.tilt.loc[tilt_a_team, 'Tilts']})
            match_tilts = match_tilts.append(add_data, ignore_index=True)
        return match_tilts
            
    def compute_score(self, tilt, goals, expected_goals):
        return (0.02 * goals * expected_goals) / tilt 
