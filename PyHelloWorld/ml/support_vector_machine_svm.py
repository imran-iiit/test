"""
    Binary Classifier
    "Best Seperating Hyperplane" for 2 sets of data. Supposed to be better than neural networks in some cases.
    Best ML algo.
    
    Yi[Xi.Wi + b] = 1 .... this is an optimization problem, where we min(Wi) and max(b)... for given Xi
    or Yi[Xi.Wi + b] = -1
    
    Once we determine the above equation, then classifying new point is just for finding whether it is +/-1 !
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

df = pd.read_csv('breast_cancer_dataset.txt')
#print(df.tail())
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1)) # Features
y = np.array(df['class']) # Labels

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train) # Pickle this for saving time later...

accuracy = clf.score(X_test, y_test)
print(accuracy) # TEST - run this without dropping the 'id' col, and you get only ~50% accuracy, opposed to ~96% now 

# Test
# example_measures = np.array([4,2,1,1,1,2,3,2,1])
# example_measures = example_measures.reshape(1, -1) # To get rid of some warning... but error for me!
example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])

prediction = clf.predict(example_measures)
print(prediction)