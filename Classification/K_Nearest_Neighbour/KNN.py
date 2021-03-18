import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('../teleCust1000t.csv')
print('\nraw data\n', df.head(5))


print('\nnumber of columns?\n' , df['custcat'].value_counts())


print(df.columns)


df.hist(column='income', bins=50) # creating a histogram

axarr = df.hist(column='income', bins=50, sharex=True, sharey=True, layout = (2, 1)) # making a histogram with new axis labels :) "Labels are properties of axes objects, that needs to be set on each of them. Here's an example that worked for me:" https://stackoverflow.com/questions/42832675/setting-axis-labels-for-histogram-pandas

for ax in axarr.flatten():
    ax.set_xlabel("income")
    ax.set_ylabel("number of people")
plt.show()


X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print('\nconverted data from pandas data frame into numpy array\n' , X[0:5])

y = df['custcat'].values
print('\nwhat are our labels?\n', y[0:5])