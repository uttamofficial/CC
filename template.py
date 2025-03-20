import sys, math
from collections import deque
sys.setrecursionlimit(10**7)
input = sys.stdin.readline

MOD = 10**9 + 7

# ================================
# Utility Functions
# ================================

def modexp(a, b, mod=MOD):
    """
    Fast modular exponentiation.
    Computes (a^b) % mod in O(log b) time.
    """
    result = 1
    a %= mod
    while b > 0:
        if b & 1:
            result = (result * a) % mod
        a = (a * a) % mod
        b //= 2
    return result

def gcd(a, b):
    """
    Computes the Greatest Common Divisor of a and b using Euclid's algorithm.
    """
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """
    Computes the Least Common Multiple of a and b.
    """
    return a * b // gcd(a, b)

# ================================
# Data Structures
# ================================

class DSU:
    """
    Disjoint Set Union (Union-Find) data structure.
    Useful for graph connectivity, MST problems, etc.
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a):
        # Path compression optimization.
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]

    def union(self, a, b):
        # Union by rank optimization.
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True

# ================================
# Segment Tree for Range Sum Queries
# ================================
class SegmentTree:
    def __init__(self, arr):
        """
        Builds the segment tree for an input list 'arr'.
        This version supports range sum queries and point updates.
        """
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self._build(arr, 0, self.n - 1, 0)

    def _build(self, arr, l, r, pos):
        if l == r:
            self.tree[pos] = arr[l]
            return
        mid = (l + r) // 2
        self._build(arr, l, mid, 2 * pos + 1)
        self._build(arr, mid + 1, r, 2 * pos + 2)
        self.tree[pos] = self.tree[2 * pos + 1] + self.tree[2 * pos + 2]

    def query(self, ql, qr, l=0, r=None, pos=0):
        if r is None:
            r = self.n - 1
        # Total overlap
        if ql <= l and r <= qr:
            return self.tree[pos]
        # No overlap
        if r < ql or l > qr:
            return 0
        mid = (l + r) // 2
        return self.query(ql, qr, l, mid, 2 * pos + 1) + self.query(ql, qr, mid + 1, r, 2 * pos + 2)

    def update(self, idx, new_val, l=0, r=None, pos=0):
        if r is None:
            r = self.n - 1
        if l == r:
            self.tree[pos] = new_val
            return
        mid = (l + r) // 2
        if idx <= mid:
            self.update(idx, new_val, l, mid, 2 * pos + 1)
        else:
            self.update(idx, new_val, mid + 1, r, 2 * pos + 2)
        self.tree[pos] = self.tree[2 * pos + 1] + self.tree[2 * pos + 2]

# ================================
# Graph Algorithms
# ================================

def dfs(node, parent, adj, visited):
    """
    Depth First Search (DFS) for traversing a graph.
    :param node: Current node being visited.
    :param parent: Parent node (to avoid revisiting).
    :param adj: Adjacency list of the graph.
    :param visited: List tracking visited nodes.
    """
    visited[node] = True
    # Process the node (e.g., print, store, etc.)
    for neighbor in adj[node]:
        if neighbor != parent and not visited[neighbor]:
            dfs(neighbor, node, adj, visited)

def bfs(start, adj):
    """
    Breadth First Search (BFS) for graph traversal.
    :param start: Starting node.
    :param adj: Adjacency list of the graph.
    """
    n = len(adj)
    visited = [False] * n
    q = deque([start])
    visited[start] = True
    while q:
        cur = q.popleft()
        # Process the current node.
        for neighbor in adj[cur]:
            if not visited[neighbor]:
                visited[neighbor] = True
                q.append(neighbor)

# ================================
# Main Solver Function
# ================================
def solve():
    """
    Main function to solve one test case.
    Modify and extend based on specific problem requirements.
    """
    # Example 1: Modular Exponentiation
    n = int(input().strip())
    print("2^n mod MOD =", modexp(2, n))
    
    # Example 2: DSU usage for processing a graph
    nodes, edges = map(int, input().split())
    dsu = DSU(nodes)
    for _ in range(edges):
        u, v = map(int, input().split())
        # Assuming nodes are 0-indexed; if 1-indexed, subtract 1.
        dsu.union(u, v)
    
    # Example 3: Segment Tree for Range Sum Queries
    arrSize = int(input().strip())
    arr = list(map(int, input().split()))
    segTree = SegmentTree(arr)
    q = int(input().strip())  # Number of queries
    for _ in range(q):
        l, r = map(int, input().split())
        print("Range Sum [", l, ",", r, "] =", segTree.query(l, r))
    
    # Example 4: Graph DFS Example
    numNodes = int(input().strip())
    numEdges = int(input().strip())
    # Build an empty graph (adjacency list) with numNodes nodes.
    graph = [[] for _ in range(numNodes)]
    for _ in range(numEdges):
        u, v = map(int, input().split())
        # Assuming 0-indexed nodes.
        graph[u].append(v)
        graph[v].append(u)
    visited = [False] * numNodes
    dfs(0, -1, graph, visited)  # Start DFS from node 0

def main():
    t = 1  # Number of test cases
    # Uncomment below if multiple test cases are provided:
    # t = int(input().strip())
    for _ in range(t):
        solve()

if __name__ == "__main__":
    main()
