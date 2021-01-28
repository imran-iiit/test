class Solution:
    def binary_search_util(self, sorted_arr, target, start, end):
        if end >= start:
            mid = int(start + (end-start)/2)
            
            if sorted_arr[mid] == target:
                return mid
            elif sorted_arr[mid] < target:
                return self.binary_search_util(sorted_arr, target, mid+1, end)
            elif sorted_arr[mid] > target:
                return self.binary_search_util(sorted_arr, target, start, mid-1)
        
        else:
            return -1
    
    def binary_search(self, arr, find):
        sorted_arr = sorted(arr)      
        idx = self.binary_search_util(sorted_arr, find, 0, len(sorted_arr)-1)
        if idx != -1:
            return arr.index(sorted_arr[idx])
        else:
            return -1
        
    def twoSum(self, nums, target):
        for i, n in enumerate(nums):
            idx = self.binary_search(nums[:i]+nums[i+1:], target-n)
            if idx == -1:
                continue
            
            if idx >= i:
                idx = idx + 1
                
            return [i, idx]
    
if __name__ == '__main__':
    nums = [3, 2, 4]
    target = 6
    sol = Solution()
#     assert(sol.binary_search([3, 4, 5, 9, 20, 23, 2], 20) == 4)
#     assert(sol.binary_search([9, 8, 7], 7) == 2)
#     assert(sol.binary_search([3, 4], 4) == 1)
#     assert(sol.binary_search([3, 4], 5) == -1)
#     assert(sol.binary_search([3], 3) == 0)
#     assert(sol.binary_search([3, 4, 5, 9, 20, 23, 2], 21) == -1)
    
    print(sol.twoSum(nums, target))
    print('Done')