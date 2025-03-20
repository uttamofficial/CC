#include <bits/stdc++.h>
using namespace std;

// ================================
// Debugging Macro (Uncomment for Local Testing)
// ================================
#ifdef LOCAL
    #define debug(x) cerr << #x << " = " << (x) << "\n";
#else
    #define debug(x)
#endif

// ================================
// Type Definitions & Constants
// ================================
typedef long long ll;
typedef vector<int> vi;
typedef vector<ll> vll;
typedef pair<int, int> pii;
const ll MOD = 1e9 + 7;

// ================================
// Utility Functions
// ================================

// Fast modular exponentiation: Computes (a^b) % mod in O(log b) time.
ll modexp(ll a, ll b, ll mod = MOD) {
    ll res = 1;
    a %= mod;
    while (b > 0) {
        if (b & 1)
            res = (res * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return res;
}

// Greatest Common Divisor (GCD) using Euclid's algorithm.
ll gcd(ll a, ll b) {
    return b == 0 ? a : gcd(b, a % b);
}

// Least Common Multiple (LCM) computed via GCD.
ll lcm(ll a, ll b) {
    return (a / gcd(a, b)) * b;
}

// ================================
// Data Structures
// ================================

// Disjoint Set Union (Union-Find) for connectivity or MST problems.
struct DSU {
    vector<int> parent, rank;
    
    // Constructor: Initializes DSU with 'n' elements (0-indexed).
    DSU(int n) {
        parent.resize(n);
        rank.assign(n, 0);
        iota(parent.begin(), parent.end(), 0); // Initialize parent[i] = i
    }
    
    // Find operation with path compression.
    int find(int a) {
        if (parent[a] != a)
            parent[a] = find(parent[a]);
        return parent[a];
    }
    
    // Union operation by rank.
    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (rank[a] < rank[b])
            swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b])
            rank[a]++;
        return true;
    }
};

// ================================
// Segment Tree (Range Sum Query)
// ================================

// This segment tree supports sum queries over a range and point updates.
struct SegmentTree {
    int n;
    vector<ll> tree;
    
    // Constructor: Builds the segment tree from an initial array.
    SegmentTree(const vector<ll>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 0, n - 1, 0);
    }
    
    // Recursively builds the tree.
    void build(const vector<ll>& arr, int l, int r, int pos) {
        if (l == r) {
            tree[pos] = arr[l];
            return;
        }
        int mid = (l + r) / 2;
        build(arr, l, mid, 2 * pos + 1);
        build(arr, mid + 1, r, 2 * pos + 2);
        tree[pos] = tree[2 * pos + 1] + tree[2 * pos + 2];
    }
    
    // Queries the sum in the range [ql, qr].
    ll query(int ql, int qr, int l, int r, int pos) {
        if (ql <= l && r <= qr)
            return tree[pos]; // Total overlap.
        if (r < ql || l > qr)
            return 0; // No overlap. (Identity element for sum)
        int mid = (l + r) / 2;
        return query(ql, qr, l, mid, 2 * pos + 1) + query(ql, qr, mid + 1, r, 2 * pos + 2);
    }
    
    // Helper to call the query function.
    ll query(int ql, int qr) {
        return query(ql, qr, 0, n - 1, 0);
    }
    
    // Updates the value at index 'idx' to 'new_val'.
    void update(int idx, ll new_val, int l, int r, int pos) {
        if (l == r) {
            tree[pos] = new_val;
            return;
        }
        int mid = (l + r) / 2;
        if (idx <= mid)
            update(idx, new_val, l, mid, 2 * pos + 1);
        else
            update(idx, new_val, mid + 1, r, 2 * pos + 2);
        tree[pos] = tree[2 * pos + 1] + tree[2 * pos + 2];
    }
    
    // Helper to call the update function.
    void update(int idx, ll new_val) {
        update(idx, new_val, 0, n - 1, 0);
    }
};

// ================================
// Graph Algorithms
// ================================

// Depth-First Search (DFS) example on an undirected graph.
void dfs(int node, int parent, const vector<vector<int>>& adj, vector<bool>& visited) {
    visited[node] = true;
    // Process current node (e.g., print or store node information).
    for (int neighbor : adj[node]) {
        if (neighbor != parent && !visited[neighbor])
            dfs(neighbor, node, adj, visited);
    }
}

// Breadth-First Search (BFS) example.
void bfs(int start, const vector<vector<int>>& adj) {
    int n = adj.size();
    vector<bool> visited(n, false);
    queue<int> q;
    q.push(start);
    visited[start] = true;
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        // Process current node.
        for (int neighbor : adj[cur]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

// ================================
// Main Function
// ================================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t = 1;
    // Uncomment below if multiple test cases are provided.
    // cin >> t;
    
    while (t--) {
        // ----------------------------
        // Example 1: Modular Exponentiation
        // ----------------------------
        int n;
        cin >> n;
        cout << "2^n mod MOD = " << modexp(2, n) << "\n";
        
        // ----------------------------
        // Example 2: DSU for Graph Connectivity
        // ----------------------------
        int nodes, edges;
        cin >> nodes >> edges;
        DSU dsu(nodes);
        for (int i = 0; i < edges; i++) {
            int u, v;
            cin >> u >> v;
            // Assuming 0-indexed input; adjust if 1-indexed.
            dsu.unite(u, v);
        }
        
        // ----------------------------
        // Example 3: Segment Tree for Range Sum Queries
        // ----------------------------
        int arrSize;
        cin >> arrSize;
        vector<ll> arr(arrSize);
        for (int i = 0; i < arrSize; i++)
            cin >> arr[i];
        SegmentTree segTree(arr);
        int q;
        cin >> q;
        while (q--) {
            int l, r;
            cin >> l >> r;
            cout << "Range Sum [" << l << ", " << r << "] = " << segTree.query(l, r) << "\n";
        }
        
        // ----------------------------
        // Example 4: Graph DFS Example
        // ----------------------------
        int numNodes;
        cin >> numNodes;
        vector<vector<int>> graph(numNodes);
        int numEdges;
        cin >> numEdges;
        for (int i = 0; i < numEdges; i++) {
            int u, v;
            cin >> u >> v;
            // Assuming 0-indexed nodes.
            graph[u].push_back(v);
            graph[v].push_back(u);
        }
        vector<bool> visited(numNodes, false);
        dfs(0, -1, graph, visited);  // Start DFS from node 0
        
        // Add more examples or modify existing ones as needed.
    }
    
    return 0;
}
