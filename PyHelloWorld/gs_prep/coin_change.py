import sys
from collections import deque

class Solution:
    # https://leetcode.com/problems/coin-change/discuss/77416/Python-11-line-280ms-DFS-with-early-termination-99-up
    # DFS with early termination, faster than 99%!
    def coinChange(self, coins, amount):
        coins.sort(reverse = True)
        min_coins = float('inf')
        
        def count_coins(start_coin, coin_count, remaining_amount):
            nonlocal min_coins
            
            if remaining_amount == 0:
                min_coins = min(min_coins, coin_count)
                return
            
            # Iterate from largest coins to smallest coins
            for i in range(start_coin, len(coins)):
                remaining_coin_allowance = min_coins - coin_count
                max_amount_possible = coins[i] * remaining_coin_allowance
                
                if coins[i] <= remaining_amount and remaining_amount < max_amount_possible:
                    count_coins(i, coin_count + 1, remaining_amount - coins[i])
            
        count_coins(0, 0, amount)
        return min_coins if min_coins < float('inf') else -1

    # https://leetcode.com/problems/coin-change/discuss/720880/Python-DP-Explanation-with-Pictures
    def coinChangeDP(self, coins, amount):
        
        # EDGE CASE
        if amount == 0:
            return 0
        
        # INIT DIMENSIONS
        nrows = len(coins) + 1
        ncols = amount + 1
        
        # BY DEFAULT, 2**64 DENOTES IMPOSSIBLE TO MAKE CHANGE
        dp = [[2**64 for _ in range(ncols)] for _ in range(nrows)]
        
        # BY DEFAULT, IF AMOUNT = 0, WE NEED EXACTLY 0 COINS
        for i in range(nrows):
            dp[i][0] = 0
            
        # OTHER CELLS
        for i in range(1, nrows):
            for j in range(1, ncols):
                
                # CASE 1 - WE MUST LEAVE THE COIN
                if j < coins[i - 1]:
                    dp[i][j] = dp[i - 1][j]
                
                # CASE 2 - WE CAN TAKE OR LEAVE THE COIN
                else:
                    take = 1 + dp[i][j - coins[i - 1]]
                    leave = dp[i - 1][j]
                    dp[i][j] = min(take, leave)
        
        for row in dp:
            print(row)
            
        return -1 if dp[-1][-1] == 2**64 else dp[-1][-1]

    # https://leetcode.com/problems/coin-change/discuss/1521067/Python-or-DP-and-BFS-(Beats-94)-or-Simple-Solutions
    def coinChangeDP1D(self, coins, amount):
        dp = [0] + [sys.maxsize for _ in range(amount)]
        for i in range(1, amount+1):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i-coin] + 1)
        if dp[amount] == sys.maxsize:
            return -1
        return dp[amount]

    # same page - DP (memoization)
    def recur(self, coins, amount, lookup):
        if amount not in lookup:
            if amount == 0:
                return 0
            minCoins = sys.maxsize
            for coin in coins:
                if amount-coin >= 0:
                    minCoins = min(minCoins, 1 + self.recur(coins, amount-coin, lookup))
            lookup[amount] = minCoins
        return lookup[amount]
    
    def coinChangeDPMemo(self, coins, amount):
        minCoins = self.recur(coins, amount, {})
        return minCoins if minCoins != sys.maxsize else -1

    ### BFS
    def coinChangeBFS(self, coins, amount):
        if amount == 0:
            return 0
        if amount in coins:
            return 1

        queue = deque([(amount, 0)])
        lookup = set([amount])
        while queue:
            remainingAmount, coinsUsed = queue.popleft()

            if remainingAmount == 0:
                return coinsUsed
                
            for coin in coins:
                if remainingAmount - coin >= 0 and remainingAmount - coin not in lookup:
                    queue.append((remainingAmount - coin, coinsUsed + 1))
                    lookup.add(remainingAmount - coin)

        return -1

if __name__ == '__main__':
    sol = Solution()
    assert sol.coinChange([1, 2, 5], 11) == 3 # 5 + 5 + 1
    assert sol.coinChangeDP([1, 2, 5], 11) == 3 # 5 + 5 + 1
    assert sol.coinChangeDP1D([1, 2, 5], 11) == 3 # 5 + 5 + 1
    assert sol.coinChangeDPMemo([1, 2, 5], 11) == 3 # 5 + 5 + 1
    assert sol.coinChangeBFS([1, 2, 5], 11) == 3 # 5 + 5 + 1