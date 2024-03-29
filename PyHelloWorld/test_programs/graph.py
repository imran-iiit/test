import pprint
# from my_package.example import TestClass
from collections import defaultdict
from collections import deque

class Graph(object):
    """ Graph data structure, undirected by default. """

    def __init__(self, connections, directed=False):
        self._graph = defaultdict(set)
        self._directed = directed
        self.add_connections(connections)

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """

        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """

        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)

    def remove(self, node):
        """ Remove all references to node """

        for _, cxns in self._graph.iteritems():
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """

        return node1 in self._graph and node2 in self._graph[node1]

    def find_path(self, node1, node2, path=[]):
        """ Find any path between node1 and node2 (may not be shortest) """

        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None
    
    def dfs(self, root, visited):
        if root in visited:
            return
        
        visited.append(root)
        for n in self._graph[root]:
            self.dfs(n, visited)
            print(n)

    def bfs(self, root, visited):
        d = deque()
        d.appendleft(root)
        while len(d):
            current = d.pop()
            if current not in visited:
                visited.append(current)
                print(current)
                for n in self._graph[current]:
                    d.appendleft(n)
            
        
        

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
    
if __name__ == '__main__':
        connections = [('A', 'B'), ('B', 'C'), ('B', 'D'),
                   ('C', 'D'), ('E', 'F'), ('F', 'C')]
        g = Graph(connections, directed=True)
        pprint.pprint(g._graph)
        
        g = Graph(connections) # undirected graph!
        pprint.pprint(g._graph)
        
        g.add('E', 'D')
        pprint.pprint(g._graph)
        
        print('Done')
        for k in g._graph.iterkeys():
            print(k, g._graph[k])
        
        print(g.find_path('A', 'D'))
        print('********* DFS *********')
        g.dfs('A', [])
        
        print('********** BFS ************ ')
        g.bfs('A', [])
#         g.remove('A')
#         pprint.pprint(g._graph)

        