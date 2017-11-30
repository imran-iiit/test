"""
y = mx + c

m = ( mean(x).mean(y) - mean(xy) ) / ( mean(x)**2 - mean(x**2) ) 

"""
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt 
from numpy import float64
from matplotlib import style

style.use('fivethirtyeight')

# xs = [1, 2, 3, 4, 5, 6]
# ys = [5, 4, 6, 5, 6, 7]

# plt.plot(xs, ys)
# plt.scatter(xs, ys)
# plt.show()

xs = np.array([1, 2, 3, 4, 5, 6], dtype=float64) # Default dtype, but still to be explicit
ys = np.array([5, 4, 6, 5, 6, 7], dtype=float64)

def best_fit_slopt(xs, ys):
    m = ( mean(xs) * mean(ys) - mean(xs*ys) ) / ( mean(xs)**2 - mean(xs**2) )
    return m

m = best_fit_slopt(xs, ys)
print(m)

# Now calculate the y InterceptedError
c = mean(ys) - m * mean(xs)
print(c)

regression_line = [m*x+c for x in xs] # calculate the y values

# plt.scatter(xs, ys)
# plt.plot(xs, regression_line)
# plt.show()
# Now, we can even predict the future!
predict_x = 8
predict_y = m * predict_x + c
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()



