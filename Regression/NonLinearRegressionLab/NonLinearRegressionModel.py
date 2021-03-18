import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("china_gdp.csv")
print(df.head(10))

plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show() #plot of raw data


def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2))) # 0 + { 1 / (1 + e^(b1(x-b2)) }
    return y

beta_1 = 0.10
beta_2 = 1990.0

#creating predicted values using our sigmoid function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')
plt.show() # plot of example logistic function curve


#optimising model parameters beta1 and beta2

# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# accuracy testing of model (unguided)

#1. split data into train and test sets
TestSplitArray= np.random.rand(len(df)) < 0.8  #using a 20:80 split, df is our csv data that has been interpreted into an array by pandas
train_y = ydata[TestSplitArray]
train_x = xdata[TestSplitArray]
test_y = ydata[~TestSplitArray]
test_x = xdata[~TestSplitArray]

#2 use train set to create a model
popt, pcov = curve_fit(sigmoid, train_x, train_y)

#3 test this model on the test data
y_hat = sigmoid(test_x, *popt)

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )
