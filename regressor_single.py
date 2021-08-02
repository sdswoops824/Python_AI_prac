import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# define the input file location
input_loc = 'C:/Users/swagata/OneDrive - Hiroshima University/personal/Job/Python_AI_practice/Python_AI_prac/linear.txt'

# read and load the input data file
input_data = np.loadtxt(input_loc, delimiter=',')


X, y = input_data[:, :-1], input_data[:, -1]

# training the model
training_samples = int(0.6*len(X))
testing_samples = len(X) - training_samples

X_train, y_train = X[:training_samples], y[:training_samples]
X_test, y_test = X[training_samples:], y[training_samples:]

# create a linear regressor object
reg_linear = linear_model.LinearRegression()

# Train the object with training samples
reg_linear.fit(X_train, y_train)

# perform prediction
y_test_pred = reg_linear.predict(X_test)

# visualize the data by plotting
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_test_pred, color = 'black', linewidth=2)
plt.xticks(())
plt.yticks(())
plt.show()

# compute the performance of regression
print("Performance of Linear regressor:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred),2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
