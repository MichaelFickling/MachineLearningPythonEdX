import numpy as np
import matplotlib.pyplot as plt

#
#create linear data and graph it
#

x = np.arange(-5.0, 5.0, 0.1) # create array of numbers regularly spaced in specific range

#You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3 #creates an array of y values
y_noise = 2 * np.random.normal(size=x.size)#creating an array of random numbers to use as errors "noise". arrays need to be the same size 1:1
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()



#
#create polynomial data and graph it
#
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# y = np.power(x,2) : y=x^2

# Y = np.exp(X) : y=e^x  test to show this si true: a=np.log(np.exp(5))
#                                                   print(a)

# Y = 1-4/(1+np.power(3, X-2)) : 1 - 4/(1+3^(X-2))
