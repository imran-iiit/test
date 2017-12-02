"""
y = mx + c

m = ( mean(x).mean(y) - mean(xy) ) / ( mean(x)**2 - mean(x**2) ) 

"""
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt 
from numpy import float64, dtype
from matplotlib import style
import random #Pseudo Random ;)

style.use('fivethirtyeight')

# xs = [1, 2, 3, 4, 5, 6]
# ys = [5, 4, 6, 5, 6, 7]

# plt.plot(xs, ys)
# plt.scatter(xs, ys)
# plt.show()

xs = np.array([1, 2, 3, 4, 5, 6], dtype=float64) # Default dtype, but still to be explicit
ys = np.array([5, 4, 6, 5, 6, 7], dtype=float64)

def create_dataset(hm, variance, step=2, correlation=False): 
    '''
        hm = "how many"
    '''
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        # Why the heck we doing the below? - Ans. this changes the values 
        # returned by random call
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
            
    xs = [i for i in range(len(ys))]    
    
    return np.array(xs, dtype=float64), np.array(ys, dtype=float64)

def best_fit_slopt(xs, ys):
    m = ( mean(xs) * mean(ys) - mean(xs*ys) ) / ( mean(xs)**2 - mean(xs**2) )
    return m

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2 )

def coeff_of_determination(ys_orig, ys_line):
    '''
        Coeff of Determination
        r^2 = 1 - StandardError(hat(y))/StandardError(mean(y))
        
        If the line is very close to the mean line --> r^2 = 0 i.e. very good
    '''
    y_mean_line = mean(ys_orig) #[mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    
    return 1 - squared_error_regr/squared_error_y_mean

### Now, we can TEST the r^2 and plot by changing the variation, i.e., by reducing the variation,
### the plots will be more tight
xs, ys = create_dataset(40, 40, 2, 'pos')
xs, ys = create_dataset(40, 10, 2, 'pos')
xs, ys = create_dataset(40, 80, 2, 'pos')
xs, ys = create_dataset(40, 40, 2, False) # if no correlation, r^2 ~ 0!!

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
predict_x = 50
predict_y = m * predict_x + c
r_squared = coeff_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s = 100, color='g')
plt.plot(xs, regression_line)
plt.show()



