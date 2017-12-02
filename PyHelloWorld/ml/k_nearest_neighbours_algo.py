"""
    Write our own K Nearest Neighbours algo and test with the result we got in teh k_nearest_neighbours.py
    
    Euclidean Distance (Father of Geometry)
    
    root(Sigma(i=1, N) (qi - pi)^2) --- q an p and two points
"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings 
style.use('fivethirtyeight') 

# p1 = [1, 3] 
# p2 = [2, 5]
# 
# euclidean_distance = sqrt((p1[0] - p2[0])**2 + (p1[1]-p2[1])**2) #But lets use numpy one as it is faster
# 
# print(euclidean_distance)

dataset = {'k':[[1,2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]} # k = black, r = red in matplotlib
new_features = [5, 7]

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=200, c='m')
# plt.show() # shows that our point is very close to red group

def k_nearest_neighbour(data, predict, k=3): # Simulating scikit learn method
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting groups!')
    
    distances = []
    for group in data:
        for features in data[group]:
#             euclidian_distance1 = np.sqrt(np.sum(np.array(features) - np.array(predict)) ** 2 ) # we will use inbuilt
            euclidian_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidian_distance, group])
    
    votes = [i[1] for i in sorted(distances)[:k]]
#     print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence  = Counter(votes).most_common(1)[0][1] / k
    
    return vote_result, confidence


results = k_nearest_neighbour(dataset, new_features, k=3)
print(results)

"""
    Now, lets actually try to compare our above algo with scikit-learn algo in k_nearest_neighbours.py
"""
import pandas as pd
import random

df = pd.read_csv('breast_cancer_dataset.txt')
#print(df.tail())
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# full_data = df.values.tolist() # Notice numbers as string!!!
full_data = df.astype(float).values.tolist() # Explicit type conversion as some data might be strings
# print(full_data[:10])

random.shuffle(full_data)
# print("#" * 20)
# print(full_data[:10])

test_size = .2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[: -int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1]) # i[-1] has the class info... i.e. malignant/not! thats our grouping

for i in test_data:
    test_set[i[-1]].append(i[:-1]) # i[-1] has the class info... i.e. malignant/not! thats our grouping


correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbour(train_set, data, k=5) # k = 5, since scikit learn doc.. 
                                                            # YOu can fiddle with this for accuracy!
        if group == vote:
            correct += 1
        else:
            print(confidence)
        total += 1

print('Accuracy: ', correct/total)





































