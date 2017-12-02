"""
    Which group is the given point closest to i.e. if k=3, what are the 3 nearest points to the given point,̏ so that the new point is in that group.
 
    This is a simple algo, which runs really fast and scales well as we are just calculating the Euclidian 
    distances between two points at a given time.
    
    Data: http://archive.ics.uci.edu/ml/datasets.html
    
    7. Attribute Information: (class attribute has been moved to last column)

   #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
  
  missign data - ?
  
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast_cancer_dataset.txt')
#print(df.tail())
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1)) # Features
y = np.array(df['class']) # Labels

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train) # Pickle this for saving time later...

accuracy = clf.score(X_test, y_test)
print(accuracy) # TEST - run this without dropping the 'id' col, and you get only ~50% accuracy, opposed to ~96% now 

# Test
# example_measures = np.array([4,2,1,1,1,2,3,2,1])
# example_measures = example_measures.reshape(1, -1) # To get rid of some warning... but error for me!
example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])

prediction = clf.predict(example_measures)
print(prediction)





















