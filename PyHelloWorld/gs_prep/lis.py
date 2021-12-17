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
    _lis(arr, n)
 
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
 
if __name__ == '__main__':
    arr1 = [3, 10, 2, 1, 20]
    arr2 = [10, 22, 9, 33, 21, 50, 41, 60]

    print("Length of lis is ", lis(arr1))
    print("Length of lis is ", dp_lis(arr1))
    print("Length of lis is ", lis(arr2))
    print("Length of lis is ", dp_lis(arr2))