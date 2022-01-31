from measure_time import catchtime
from functools import cache
import random

### https://www.geeksforgeeks.org/longest-increasing-subsequence-dp-3/

# A naive Python implementation of LIS problem
 
""" To make use of recursive calls, this function must 
return two things:
 1) Length of LIS ending with element arr[n-1]. We use
 max_ending_here for this purpose
 2) Overall maximum as the LIS may end with an element
 before arr[n-1] max_ref is used this purpose.
 The value of LIS of full array of size n is stored in
 *max_ref which is our final result """
 
# global variable to store the maximum
global maximum 

@cache 
def _lis(arr, n):
    global maximum
 
    # Base Case
    if n == 1:
        return 1
 
    # maxEndingHere is the length of LIS ending with arr[n-1]
    maxEndingHere = 1
 
    """Recursively get all LIS ending with arr[0], arr[1]..arr[n-2]
       IF arr[i-1] is maller than arr[n-1], and max ending with
       arr[n-1] needs to be updated, then update it"""
    for i in range(1, n):
        res = _lis(arr, i)
        if arr[i-1] < arr[n-1] and res+1 > maxEndingHere:
            maxEndingHere = res + 1
 
    # Compare maxEndingHere with overall maximum. And
    # update the overall maximum if needed
    maximum = max(maximum, maxEndingHere)
 
    return maxEndingHere
 

def lis(arr):
    global maximum
    n = len(arr)
 
    # maximum variable holds the result
    maximum = 1
 
    # The function _lis() stores its result in maximum
    _lis(tuple(arr), n) # IAS: Making immutable to hash for @cache to work!
 
    return maximum

########################################################
# Dynamic programming Python implementation
# of LIS problem
 
# lis returns length of the longest
# increasing subsequence in arr of size n 
def dp_lis(arr):
    n = len(arr)
 
    # Declare the list (array) for LIS and
    # initialize LIS values for all indexes
    lis = [1]*n
 
    # Compute optimized LIS values in bottom up manner
    for i in range(1, n):
        for j in range(0, i):
            if arr[i] > arr[j] and lis[i] < lis[j] + 1:
                lis[i] = lis[j]+1
 
    # Pick the max from lis 
    maximum = 0
    for i in range(n):
        maximum = max(maximum, lis[i])
 
    return maximum

## https://leetcode.com/problems/longest-increasing-subsequence/discuss/74824/JavaPython-Binary-search-O(nlogn)-time-with-explanation
# O(NlogN) solution - tails! Patience Sort 
def lis_onlogn(nums):
    tails = [0] * len(nums)
    size = 0

    for x in nums:
        i, j = 0, size

        while i != j:
            m = (i + j) // 2
            if tails[m] < x: # See Patience Sort PDF file:///Users/aniron/Downloads/LongestIncreasingSubsequence.pdf
                             # - Basically, if the curr card face is less than
                             # x[], we cannot place this card in this location, but on the right 
                i = m + 1
            else:
                j = m

        tails[i] = x  # overwrite the smallest card val that is on the top on which 
                      # no other smaller card can be kept!
        size = max(i + 1, size)

    return size

if __name__ == '__main__':
    arr1 = [3, 10, 2, 1, 20]
    arr2 = [10, 22, 9, 33, 21, 50, 41, 60, 23, 234, 322, 343, 431, 756, 434, 834, 932, 985, 1000, 342, 1033]
    arr = []
    for i in range(100):
        arr.append(random.randint(-10000, 10000))
    print(arr)

    # with catchtime():
    #     print("Length of lis is ", lis(arr1))
    # with catchtime():
    #     print("Length of lis is ", dp_lis(arr1))
    with catchtime():
        print("Length of lis is O(NlogN) --> ", lis_onlogn(arr1))
    with catchtime():
        print("Length of lis is --> ", dp_lis(arr))
    with catchtime():
        print("Length of lis is @cache --> ", lis(arr))
