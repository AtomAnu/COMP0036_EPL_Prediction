from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

class Team():
    def __init__(self, team_name, X, y, degree=None):
        self.team_name = team_name
        self.X = X
        self.y = y
        self.degree = degree

        self.X_train, self.X_test, \
            self.y_train, self.y_test = self.preprocessing(self.X, self.y, degree)

    def preprocessing(self, X, y, degree):
        """
        Preprocess data by spliting the training data to train and test data sets.
        The X data is also scaled according to the model used.

        :param X: X data (matrix)
        :param y: y data
        :param degree: polynomial degree of the model
        :return: scaled train and test data sets
        """
        X_train, X_test, \
            y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

        if degree is None:
            scaler = StandardScaler()
        else:
            scaler = PolynomialFeatures(degree=degree)

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
