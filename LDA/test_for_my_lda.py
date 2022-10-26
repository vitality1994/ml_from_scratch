################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_cross_val import my_cross_val
from MyLDA import MyLDA

# load dataset
data = pd.read_csv('lda_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

num_data, num_features = X.shape

plt.scatter(X[:1000, 0], X[:1000, 1])
plt.scatter(X[1000:, 0], X[1000:, 1])
plt.show()

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

#####################
# ADD YOUR CODE BELOW
#####################
lambda_vals = [0.00018, 0.00019, 0.0002, 0.00021, 0.00022, 0.00023, 0.00024, 0.00025]
list_average_err = []

for lambda_val in lambda_vals:


	# instantiate LDA object
	temp = MyLDA(lambda_val)

	# call to your CV function to compute error rates for each fold
	# print error rates from CV
	temp = my_cross_val(temp, "err_rate", X_train, y_train, k=10)

	print("LDA CV error rates when lambda is ", lambda_val)
	print(temp, np.mean(temp), np.std(temp))
	
	
	list_average_err.append(np.mean(temp))
	
# instantiate LDA object for best value of lambda
average_err_min = min(list_average_err)
lambda_min = lambda_vals[list_average_err.index(average_err_min)]

print("Lambda value makes the smallest average error rate is: ", lambda_min)
temp_2 = MyLDA(lambda_min)

	
# fit model using all training data
temp_2.fit(X_train, y_train)

# predict on test data
y_hat = temp_2.predict(X_test)

# compute error rate on test data

count = 0
for i in range(len(y_test)):
	if y_test[i] != y_hat[i]:
		count +=1
		err_rate = (1/len(y_test))*count



# print error rate on test data
print("Error rate is: ", err_rate)