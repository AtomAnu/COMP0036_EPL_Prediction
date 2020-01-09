import pandas as pd
import numpy as np

class Ratings:
    """
    Calculate the Team ELO Ratings for the training data and store the current ratings for all teams

    home_team_list: List containing the team name with 'HomeTeam_'
    team_list: List containing the team name
    away_team_list: List containing the team name with 'AwayTeam'
    rating: DataFrame containing teams' current rating
    result: DataFrame containing ratings for the training data
    """
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
        self.rating = self.initialize()
        print(self.rating)
        self.result = self.update_ratings(X)
        print(self.result)

    def initialize(self):
        ratings = pd.DataFrame(
            {'Team': self.team_list})
        ratings['Ratings'] = 1500
        return ratings

    def update_ratings(self, X):
        match_ratings = pd.DataFrame(columns=['HomeRatings', 'AwayRatings'])
        for i in range(len(X)):
            rating_h_team = 0
            rating_a_team = 0
            for team_h in self.home_team_list:
                for team_a in self.away_team_list:
                    if X.loc[i, team_h] == 1 and X.loc[i, team_a] == 1:
                        rating_h_team = self.rating[self.rating['Team'].isin(
                            [team_h.replace('HomeTeam_', '')])].index.values
                        rating_h_team = rating_h_team[0]
                        rating_a_team = self.rating[self.rating['Team'].isin(
                            [team_a.replace('AwayTeam_', '')])].index.values
                        rating_a_team = rating_a_team[0]
            add_data = pd.Series(
                {'HomeRatings': self.rating.loc[rating_h_team, 'Ratings'], 'AwayRatings': self.rating.loc[rating_a_team, 'Ratings']})
            match_ratings = match_ratings.append(add_data, ignore_index=True)
            expect_h = self.compute_score(
                self.rating.loc[rating_h_team, 'Ratings'], self.rating.loc[rating_a_team, 'Ratings'])
            expect_a = self.compute_score(
                self.rating.loc[rating_a_team, 'Ratings'], self.rating.loc[rating_h_team, 'Ratings'])

            if X.loc[i, 'FTR'] == 1:
                adjust_h = 1
                adjust_a = 0
            elif X.loc[i, 'FTR'] == 3:
                adjust_h = 0
                adjust_a = 1
            elif X.loc[i, 'FTR'] == 2:
                adjust_h = 0.5
                adjust_a = 0.5
            
            self.rating.loc[rating_h_team, 'Ratings'] = self.rating.loc[rating_h_team, 'Ratings'] + \
                self.compute_k(
                    self.rating.loc[rating_h_team, 'Ratings']) * (adjust_h - expect_h)
            self.rating.loc[rating_a_team, 'Ratings'] = self.rating.loc[rating_a_team, 'Ratings'] + \
                self.compute_k(
                    self.rating.loc[rating_a_team, 'Ratings']) * (adjust_a - expect_a)
        return match_ratings

    # calculate the weight for the change of the rating
    def compute_k(self, rating):
        if rating >= 2400:
            return 15
        elif rating >= 2100:
            return 20
        else:
            return 25

    def compute_score(self, rating1, rating2):
        return 1 / (1+pow(10, (rating2 - rating1) / 400))

