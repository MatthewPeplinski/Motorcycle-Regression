import numpy as np


# Define the LinearRegression class, which will model a simple linear regression
class LinearRegression:
    # The __init__ function is called when an object of the class is created. It initializes the parameters of the model.
    def __init__(self):  #self is analygous to using this. in java
        self.coef_ = None  # This will hold the coefficients (weights) for the features
        self.intercept_ = None  # This will hold the intercept (bias term)

    # The fit function is used to train the model, i.e., to calculate the coefficients and intercept based on the data
    def fit(self, X, y):
        # M is the feature matrix X with an additional column of ones added for the intercept term (bias).
        # This is needed because linear regression models include an intercept (bias) term.
        M = np.column_stack((np.ones(X.shape[0]), X))

        # beta is the vector of coefficients, calculated using the Normal Equation for linear regression.
        # np.linalg.inv() calculates the inverse of a matrix, and @ is the matrix multiplication operator.
        beta = np.linalg.inv(M.T @ M) @ M.T @ y

        # The first element of beta is the intercept, and the rest are the coefficients for each feature.
        self.intercept_ = beta[0]  # The intercept is the first element of beta
        self.coef_ = beta[1:]  # The coefficients for the features are the remaining elements of beta

        # This returns a reference to the current object
        return self

    # The predict function is used to make predictions using the trained model.
    # It calculates the output by multiplying the input features (X) with the coefficients, and then adding the intercept.
    def predict(self, X):
        return X @ self.coef_ + self.intercept_  # Return the predicted values

    # The score function calculates how well the model performs by returning the R-squared score,
    # which tells how much of the variance in the dependent variable is explained by the independent variables.
    def score(self, X, y):
        # np.sum() adds all elements in the array, and np.average() calculates the average of the elements.
        # The formula calculates the R-squared score: 1 minus the ratio of the residual sum of squares to the total sum of squares.
        return 1.0 - np.sum((self.predict(X) - y) ** 2.0) / np.sum((y - np.average(y)) ** 2.0)


    # Root Mean Square Error (RMSE) measures the average error between the predicted and actual values.
    def RMSE(self, X, y):
        return np.sqrt(np.average((self.predict(X) - y) ** 2.0))
