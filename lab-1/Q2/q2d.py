import D3base

acc_array = D3base.find_accuracy(4)

print("Accuracy on train data : ", acc_array[0])
print("Accuracy on test data  :",  acc_array[1])

D3base.print_confusion_matrix(4)

D3base.plot_fun(4)