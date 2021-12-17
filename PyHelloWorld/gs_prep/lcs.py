### https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/

# Naive LCS
def rec_lcs(X, Y, m, n):
 
    if m == 0 or n == 0:
        return 0;
    elif X[m-1] == Y[n-1]:
        return 1 + rec_lcs(X, Y, m-1, n-1)
    else:
        return max(rec_lcs(X, Y, m, n-1), rec_lcs(X, Y, m-1, n))
 
# Dynamic Programming implementation of LCS problem 
def lcs(X , Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
 
    # declaring the array for storing the dp values
    L = [[None]*(n+1) for i in range(m+1)]
 
    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0 
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1 # from Note above - filling 
                            # result of comparing
                            # X[i-1] and Y[j-1] in L[i][j]
            else:
                L[i][j] = max(L[i-1][j] , L[i][j-1])
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]

if __name__ == '__main__': 
    X = "AGGTAB"
    Y = "GXTXAYB"
    print(f'LCS of {X} and {Y} --> ', rec_lcs(X , Y, len(X), len(Y)))
    assert rec_lcs(X , Y, len(X), len(Y)) == 4
    assert lcs(X , Y) == 4

    X = "ABCDGH"
    Y = "AEDFHR"
    print(f'LCS of {X} and {Y} --> ', rec_lcs(X , Y, len(X), len(Y)))
    assert rec_lcs(X , Y, len(X), len(Y)) == 3
    assert lcs(X , Y) == 3
