from collections import defaultdict

def Tree():
    return defaultdict(Tree)

# class Node(object):
#     def __init__(self, val, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#         
#     def inorder(self, root):
#         if not root:
#             return
#         self.inorder(root.left)
#         print root
#         self.inorder(root.right)
#     
#     def __str__(self):
#         return self.val
    
if __name__ == '__main__':
    t = Tree()
    print(t)
    t[1] = 'value'
    print(t)
#     t[2] = 'val2'
    t[2][2] = 'val22'
    print(t)
    
    
#     f = Node('f')
#     e = Node('e')
#     d = Node('d')
#     c = Node('c', f)
#     b = Node('b', d, e)
#     a = Node('a', b, c)
# 
#     a.inorder(a)
        
        