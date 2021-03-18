import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#getting the data
df = pd.read_csv("FuelConsumptionCo2.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#using matplotlib , showing a plot of enginesize against co2emissions, notice it is linear
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#using numpy to randomly assign 80% of the data to training set and 20% to testing set
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
print(msk)


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
#creating polynomial variables for train and test sets i.e. create array of enginesize values for x values (indepenent variable).
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
print(train_x[[0,1,2,3,4,5]])
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

#make an array where each variable now has a row that is representing a 2 degree polynomial
#e.g. enginesize 1.5 gets polynomial 1+1.5^1+1.5^2, predicting a co2emission of 1+1.5+2.25 = 4.75
poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
print('train_x_poly:\n', train_x_poly)

#
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
print('train_y_:\n', train_y_)

# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")