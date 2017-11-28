from collections import deque

print '########### Stack ##########'
stack = deque()
stack.appendleft('a')
stack.appendleft('b')
stack.appendleft('c')

print stack.popleft()
print stack.popleft()
print stack.popleft()

print '######### Queue ########'
queue = deque()
queue.appendleft('a')
queue.appendleft('b')
queue.appendleft('c')
print queue.pop()
print queue.pop()
print queue.pop()