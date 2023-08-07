import base

base.plot_fun(1,1)
base.plot_fun(2,1)

#for degree 2
base.plot_fun(1,2)
base.plot_fun(2,2)
base.mse_vs_degree(base.x_subtrain, base.y_subtrain,1)
base.mse_vs_degree(base.x_validation,base.y_validation,2)

#error report for train and test data
base.error_report()

base.bestmodel_vs_expected()

#Report observations for the above plots
