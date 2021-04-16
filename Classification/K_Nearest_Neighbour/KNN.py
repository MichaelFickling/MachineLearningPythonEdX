import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('../teleCust1000t.csv')
print('\nraw data\n', dataset.head(5))

print('\namount in each custcat catagory?\n', dataset['custcat'].value_counts())

print('list of columns', dataset.columns)

dataset.hist(column='income', bins=50) # creating a histogram

axarr = dataset.hist(column='income', bins=50, sharex=True, sharey=True, layout = (2, 1)) # making a histogram with new axis labels :) "Labels are properties of axes objects, that needs to be set on each of them. Here's an example that worked for me:" https://stackoverflow.com/questions/42832675/setting-axis-labels-for-histogram-pandas

for ax in axarr.flatten():
    ax.set_xlabel("income")
    ax.set_ylabel("number of people")
plt.show()

#converting the pandas 'data frame' into a Numpy Array so we can use Numpy
X = dataset[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']] .values  #.astype(float)
print('\nconverted data from pandas data frame into numpy array\n' , X[0:5])

y = dataset['custcat'].values
print('\nwhat are our labels?\n', y[0:5])

#normalising data / normalising data using sklearn
# The standard score of a sample x is calculated as: z = (x - u) / s
# where u is the mean of the training samples or zero if with_mean=False,
# and s is the standard deviation of the training samples or one if with_std=False.
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print('\nnow our data looks like this\n',X[0:5])

#creating test and training sets using scikit learn
X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y, test_size=0.2, random_state=4)
print('\nTrain set:\n', X_train.shape,  y_train.shape)
print('\nTest set:\n', X_test.shape,  y_test.shape)

#Train Model and Predict
k = 2
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_test)
print('\nfirst 5 prdictions\n',yhat[0:5])

from sklearn import metrics
print("\nTrain set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

err=True
while err == True:
    err=False
    try:
        Ks = int(input('\nWhat value of k would you like (~k=40 is the peak accuracy)\n'))
    except:
        print("\nERROR please put only integers in")
        err=True



mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

print('\ntest accuracy for each k\n', mean_acc)

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()