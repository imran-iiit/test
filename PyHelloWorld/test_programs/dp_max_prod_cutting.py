"""
Given a rope of length n meters, cut in differnt parts of integers so that product is max.

n = 10 --> 3*3*4
"""
def maxprod(n):
    if n == 0 or n == 1:
        return 0
    
    max_val = 0
    
    for i in range(1, n):
        max_val = max(max_val, i*(n-i), maxprod(n-i)*i)
    
    return max_val

def maxprod_dp(n):
    val = [0]*(n+1)
#     val[0] = val[1] = 0
    
    for i in range(1, n+1):
        max_val = 0
        for j in range(1, i/2+1):
            max_val = max(max_val, j*(i-j), j*val[i-j])
        val[i] = max_val
    
    return val[n]

if __name__ == '__main__':
    print(maxprod_dp(10))