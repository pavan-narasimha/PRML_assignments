import D3base

acc_array = D3base.find_accuracy(2)

print("Accuracy on train data : ", acc_array[0])
print("Accuracy on test data  :",  acc_array[1])

D3base.print_confusion_matrix(2)

D3base.plot_fun(2)