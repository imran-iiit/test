"""
    https://pythonprogramming.net/kernels-with-svm-machine-learning-tutorial/
    
    Kernels are similarity functions, which take two inputs and return a similarity using inner products.
    What kernels are going to allow us to do, possibly, is work in many dimensions, without actually 
    paying the processing costs to do it. Kernels do have a requirement: They rely on inner products. 
    For the purposes of this tutorial, "dot product" and "inner product" are entirely interchangeable.
    
    Types of Kernel:
        1. Polynomial Kernel
        2. Radio Basis Function Kernel (RBF - default)
"""

import numpy as np

a1 = np.array([1, 2, 3, 4, 5])
a2 = np.array([6, 7, 8, 9, 10])

print(np.dot(a1, a2))
print(np.inner(a1, a2))
assert(np.dot(a1, a2) == np.inner(a1, a2)) # Inner Product == Dot Product!


"""
    https://www.youtube.com/watch?v=JHaqodAQqiI&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=31
"""