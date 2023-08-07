import D2base

acc_array = D2base.find_accuracy(1)

print("Accuracy on train data : ", acc_array[0])
print("Accuracy on test data  :",  acc_array[1])

D2base.print_confusion_matrix(1)

D2base.plot_fun(1)