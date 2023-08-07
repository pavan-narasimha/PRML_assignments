import numpy as np
import base
import matplotlib.pyplot as plt
from numpy.random import randint

degree = 3

mse_result_subtrain = []
lambda_list = []

# for k in range(1,10):
#     mse_result_train.append(base.find_modified_mse(base.x_test1, base.y_test1, degree, k-10))
#     lambda_list.append(k-10)
values = randint(0,100,50)

for k in range(len(values)):
    mse_result_subtrain.append(base.find_modified_mse(base.x_subtrain, base.y_subtrain, degree,10**k))
    mse_result_subtrain.append(base.find_modified_mse(base.x_subtrain, base.y_subtrain, degree,-(10**k)))
    lambda_list.append(10**k)
    lambda_list.append(-(10**k))

mse_array = np.array(mse_result_subtrain)
lambda_array = np.array(lambda_list)

# print("mse array: ")
# print(mse_array)

fig = plt.figure(figsize=(4, 3))
plt.scatter(lambda_array,mse_array)
plt.xlabel('lambda')
plt.ylabel('MSE mean square error on train data')
plt.title("lambda vs MSE ")
plt.show()


#error report for train and test data
best_lamda = 0
mse_train = base.find_modified_mse(base.x_test1,base.y_test1, degree, 0)
mse_test  = base.find_modified_mse(base.x_test2,base.y_test2, degree, 0)

print("MSE for train data is : ", mse_train)
print("MSE for test data is :  ", mse_test)

#scatter plot for best model output vs expected output for both train and test data
#for train data set

coeff_mat = base.find_coeff_mat(degree)

y_expected_train = base.y_test1
y_best_actual_train   = base.expected_output(base.x_test1,degree, coeff_mat)

fig = plt.figure(figsize=(4, 3))
plt.scatter(y_best_actual_train, y_expected_train)
plt.xlabel('best model output for train data')
plt.ylabel('expected output for train data')
plt.title("best model output vs expected output for train data")
plt.show()

#for test data
y_expected_test = base.y_test2
y_best_actual_test = base.expected_output(base.x_test2, degree, coeff_mat)

fig = plt.figure(figsize=(4, 3))
plt.scatter(y_best_actual_test, y_expected_test)
plt.xlabel('best model output for test data')
plt.ylabel('expected output for test data')
plt.title("best model output vs expected output for test data")
plt.show()

