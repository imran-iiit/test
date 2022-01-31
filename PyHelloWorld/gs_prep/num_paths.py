from measure_time import catchtime
from functools import cache
from time import sleep

class Solution:
    ### Method 1 - brute force
    def uniquePaths(self, m, n):
        def dfs(i, j):
            if i >= m or j >= n:      return 0
            if i == m-1 and j == n-1: return 1
            return dfs(i+1, j) + dfs(i, j+1)

        return dfs(0, 0)
    
    ### @cache out of the box!!!
    def uniquePathsCache(self, m, n):
        @cache
        def dfs(i, j):
            if i >= m or j >= n:      return 0
            if i == m-1 and j == n-1: return 1
            return dfs(i+1, j) + dfs(i, j+1)

        return dfs(0, 0)
    
    ### DP!!!!
    def uniquePathsDp(self, m, n):
        dp = [[1]*n for _ in range(m)]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[-1][-1]

if __name__ == '__main__':
    sol = Solution()
    # with catchtime():
    #     print(sol.uniquePaths(10, 20))

    with catchtime():
        print(sol.uniquePathsCache(10, 20))
    
    with catchtime():
        print(sol.uniquePathsDp(10, 20))

    # with catchtime() as t:
    #     sleep(2)