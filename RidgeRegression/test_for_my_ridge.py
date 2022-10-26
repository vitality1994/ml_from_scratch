################################
# DO NOT EDIT THE FOLLOWING CODE
################################
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
import numpy as np

from MyRidgeRegression import MyRidgeRegression
from my_cross_val import my_cross_val

# load dataset
X, y = fetch_california_housing(return_X_y=True)

num_data, num_features = X.shape

# shuffle dataset
np.random.seed(2022)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

lambda_vals = [0.01, 0.1, 0, 1, 10, 100]

#####################
# ADD YOUR CODE BELOW
#####################
list_average_err_ridge = []
list_average_err_lasso = []

for lambda_val in lambda_vals:

    # instantiate ridge regression object
	temp_1 = MyRidgeRegression(lambda_val)
    # call to your CV function to compute mse for each fold
	temp_1 = my_cross_val(temp_1, "mse", X_train, y_train, k=10)
    # print mse from CV
	print("Ridge regression CV MSE values when lambda is ", lambda_val)
	print(temp_1, np.mean(temp_1), np.std(temp_1))
    # instantiate lasso object
	temp_2 = Lasso(lambda_val)
    # call to your CV function to compute mse for each fold
	temp_2 = my_cross_val(temp_2, "mse", X_train, y_train, k=10)
    # print mse from CV
	print("Lasso regression CV MSE values when lambda is ", lambda_val)
	print(temp_2, np.mean(temp_2), np.std(temp_2))

	list_average_err_ridge.append(np.mean(temp_1))

	list_average_err_lasso.append(np.mean(temp_2))

# instantiate ridge regression and lasso objects for best values of lambda
average_err_min_ridge = min(list_average_err_ridge)
lambda_min_ridge = lambda_vals[list_average_err_ridge.index(average_err_min_ridge)]

average_err_min_lasso = min(list_average_err_lasso)
lambda_min_lasso = lambda_vals[list_average_err_lasso.index(average_err_min_lasso)]

print("Lambda value makes the smallest average error rate for Ridge is: ", lambda_min_ridge)
print("Lambda value makes the smallest average error rate for Lasso is: ", lambda_min_lasso)

temp_ridge = MyRidgeRegression(lambda_min_ridge)
temp_lasso = Lasso(lambda_min_lasso)

# fit models using all training data
temp_ridge.fit(X_train, y_train)
temp_lasso.fit(X_train, y_train)

# predict on test data
y_hat_ridge = temp_ridge.predict(X_test)
y_hat_lasso = temp_lasso.predict(X_test)

# compute mse on test data
mse_ridge = sum((y_test - y_hat_ridge)**2)/len(y_test)
mse_lasso = sum((y_test - y_hat_lasso)**2)/len(y_test)
# print mse on test data
print("MSE of Ridge is: ", mse_ridge )
print("MSE of Lasso is: ", mse_lasso)