import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_csv("../drug200.csv", delimiter=",")
print(my_data[0:6])
print('number of rows/patients: ',len(my_data))
print('number of (rows , variables per row): ',my_data.shape)


X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

#converting the non-numeric values into numeric representations
# and then putting those into the dataset instead.
# because the sk learn decision trees can only deal with numeric data
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

print(X[0:5])

