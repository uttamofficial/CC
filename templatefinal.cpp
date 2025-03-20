#include <bits/stdc++.h>
using namespace std;
 
// ----------------- Fast I/O, Macros & Typedefs -----------------
#define fast_io ios::sync_with_stdio(false); cin.tie(nullptr)
#define rep(i, a, b) for (int i = a; i < b; i++)
typedef long long ll;
const int MOD = 1e9 + 7, INF = 1e9;
 
// ----------------- Math Utilities -----------------
// GCD, Modular Exponentiation, Sieve, Matrix Exponentiation
ll gcd(ll a, ll b) { return b ? gcd(b, a % b) : a; }
ll modExp(ll b, ll e, ll m = MOD) {
    ll res = 1; while (e) { if (e & 1) res = (res * b) % m; b = (b * b) % m; e /= 2; } return res;
}
vector<int> sieve(int n) {
    vector<bool> is(n + 1, true); vector<int> primes; is[0] = is[1] = false;
    rep(i, 2, n + 1) { if (is[i]) { primes.push_back(i); for (int j = 2 * i; j <= n; j += i) is[j] = false; } }
    return primes;
}
typedef vector<vector<ll>> matrix;
matrix matMul(const matrix &A, const matrix &B, ll mod = MOD) {
    int n = A.size(), m = B[0].size(), p = A[0].size();
    matrix C(n, vector<ll>(m, 0));
    rep(i, 0, n) rep(j, 0, m) rep(k, 0, p)
        C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod;
    return C;
}
matrix matPow(matrix A, ll e, ll mod = MOD) {
    int n = A.size(); matrix res(n, vector<ll>(n, 0));
    rep(i, 0, n) res[i][i] = 1;
    while (e) { if (e & 1) res = matMul(res, A, mod); A = matMul(A, A, mod); e /= 2; }
    return res;
}
 
// ----------------- Data Structures -----------------
// DSU (Union-Find)
struct DSU {
    vector<int> p, r;
    DSU(int n) { p.resize(n); r.assign(n, 0); rep(i, 0, n) p[i] = i; }
    int find(int a) { return p[a] = (p[a] == a ? a : find(p[a])); }
    void merge(int a, int b) { a = find(a), b = find(b); if (a == b) return; if (r[a] < r[b]) swap(a, b); p[b] = a; if (r[a] == r[b]) r[a]++; }
};
 
// Segment Tree (Range Sum & Point Update)
struct SegTree {
    int n; vector<int> tree;
    SegTree(int n): n(n) { tree.assign(4 * n, 0); }
    void build(vector<int>& arr, int idx, int l, int r) {
        if (l == r) { tree[idx] = arr[l]; return; }
        int m = (l + r) / 2;
        build(arr, 2 * idx, l, m); build(arr, 2 * idx + 1, m + 1, r);
        tree[idx] = tree[2 * idx] + tree[2 * idx + 1];
    }
    int query(int idx, int l, int r, int ql, int qr) {
        if (ql > r || qr < l) return 0; if (ql <= l && r <= qr) return tree[idx];
        int m = (l + r) / 2;
        return query(2 * idx, l, m, ql, qr) + query(2 * idx + 1, m + 1, r, ql, qr);
    }
    void update(int idx, int l, int r, int pos, int val) {
        if (l == r) { tree[idx] = val; return; }
        int m = (l + r) / 2;
        if (pos <= m) update(2 * idx, l, m, pos, val);
        else update(2 * idx + 1, m + 1, r, pos, val);
        tree[idx] = tree[2 * idx] + tree[2 * idx + 1];
    }
};
 
// Fenwick Tree (Binary Indexed Tree)
struct Fenw {
    int n; vector<int> bit;
    Fenw(int n): n(n) { bit.assign(n + 1, 0); }
    void update(int i, int d) { for (; i <= n; i += i & -i) bit[i] += d; }
    int query(int i) { int s = 0; for (; i; i -= i & -i) s += bit[i]; return s; }
};
 
// Trie (Prefix Tree)
struct TrieNode {
    bool end; unordered_map<char, TrieNode*> ch;
    TrieNode(): end(false) {}
};
struct Trie {
    TrieNode *root;
    Trie() { root = new TrieNode(); }
    void insert(const string &s) {
        TrieNode* cur = root;
        for (char c : s) { if (!cur->ch.count(c)) cur->ch[c] = new TrieNode(); cur = cur->ch[c]; }
        cur->end = true;
    }
    bool search(const string &s) {
        TrieNode* cur = root;
        for (char c : s) { if (!cur->ch.count(c)) return false; cur = cur->ch[c]; }
        return cur->end;
    }
};
 
// ----------------- Graph Algorithms -----------------
// Graph DFS/BFS & Dijkstra’s Shortest Path
struct Graph {
    int n; vector<vector<int>> adj;
    Graph(int n): n(n) { adj.resize(n); }
    void addEdge(int u, int v) { adj[u].push_back(v); }
    void dfs(int u, vector<bool>& vis) { vis[u] = true; for (int v : adj[u]) if (!vis[v]) dfs(v, vis); }
    void bfs(int s) {
        vector<bool> vis(n, false); queue<int> q; q.push(s); vis[s] = true;
        while (!q.empty()) { int u = q.front(); q.pop(); for (int v : adj[u]) if (!vis[v]) { vis[v] = true; q.push(v); } }
    }
};
vector<int> dijkstra(int n, vector<vector<pair<int, int>>>& g, int s) {
    vector<int> dist(n, INF); dist[s] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto &edge : g[u]) {
            int v = edge.first, w = edge.second;
            if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; pq.push({dist[v], v}); }
        }
    }
    return dist;
}
 
// ----------------- String Algorithms -----------------
// KMP (Knuth-Morris-Pratt) Pattern Matching
vector<int> computeLPS(const string &p) {
    int m = p.size(); vector<int> lps(m, 0);
    for (int i = 1, len = 0; i < m; ) {
        if (p[i] == p[len]) { lps[i++] = ++len; }
        else { if (len) len = lps[len - 1]; else lps[i++] = 0; }
    }
    return lps;
}
vector<int> kmpSearch(const string &s, const string &p) {
    vector<int> lps = computeLPS(p), occ;
    for (int i = 0, j = 0; i < s.size(); ) {
        if (s[i] == p[j]) { i++; j++; if (j == p.size()) { occ.push_back(i - j); j = lps[j - 1]; } }
        else { if (j) j = lps[j - 1]; else i++; }
    }
    return occ;
}
 
// ----------------- Advanced Topics -----------------
// Sparse Table for RMQ (Range Minimum Query)
struct SparseTable {
    int n, logn; vector<vector<int>> st;
    SparseTable(vector<int>& arr) {
        n = arr.size(); logn = floor(log2(n)) + 1;
        st.assign(n, vector<int>(logn));
        rep(i, 0, n) st[i][0] = arr[i];
        rep(j, 1, logn) rep(i, 0, n - (1 << j) + 1)
            st[i][j] = min(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
    }
    int query(int L, int R) {
        int j = floor(log2(R - L + 1));
        return min(st[L][j], st[R - (1 << j) + 1][j]);
    }
};
 
// Heavy-Light Decomposition (Skeleton for Tree Queries)
struct HLD {
    int n; vector<vector<int>> adj;
    vector<int> parent, depth, heavy, head, pos; int curPos;
    HLD(int n, vector<vector<int>> &adj): n(n), adj(adj) {
        parent.assign(n, -1); depth.assign(n, 0); heavy.assign(n, -1);
        head.resize(n); pos.resize(n); curPos = 0; dfs(0); decompose(0, 0);
    }
    int dfs(int u) {
        int size = 1, maxSize = 0;
        for (int v : adj[u]) {
            if (v == parent[u]) continue;
            parent[v] = u; depth[v] = depth[u] + 1;
            int sub = dfs(v);
            if (sub > maxSize) { maxSize = sub; heavy[u] = v; }
            size += sub;
        }
        return size;
    }
    void decompose(int u, int h) {
        head[u] = h; pos[u] = curPos++;
        if (heavy[u] != -1) decompose(heavy[u], h);
        for (int v : adj[u]) if (v != parent[u] && v != heavy[u]) decompose(v, v);
    }
};
 
// Maximum Flow (Edmonds-Karp)
struct MaxFlow {
    struct Edge { int to, rev, cap; };
    int n; vector<vector<Edge>> g;
    MaxFlow(int n): n(n) { g.resize(n); }
    void addEdge(int s, int t, int cap) {
        g[s].push_back({t, (int)g[t].size(), cap});
        g[t].push_back({s, (int)g[s].size() - 1, 0});
    }
    int bfs(int s, int t, vector<int>& level) {
        fill(level.begin(), level.end(), -1); level[s] = 0;
        queue<int> q; q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto &e : g[u])
                if (level[e.to] < 0 && e.cap) { level[e.to] = level[u] + 1; q.push(e.to); }
        }
        return level[t];
    }
    int dfs(int u, int t, int flow, vector<int>& it, vector<int>& level) {
        if (u == t) return flow;
        for (; it[u] < g[u].size(); it[u]++) {
            Edge &e = g[u][it[u]];
            if (e.cap && level[e.to] == level[u] + 1) {
                int curr = dfs(e.to, t, min(flow, e.cap), it, level);
                if (curr) { e.cap -= curr; g[e.to][e.rev].cap += curr; return curr; }
            }
        }
        return 0;
    }
    int maxFlow(int s, int t) {
        int flow = 0; vector<int> level(n), it(n);
        while (bfs(s, t, level) >= 0) {
            fill(it.begin(), it.end(), 0);
            while (int pushed = dfs(s, t, INF, it, level))
                flow += pushed;
        }
        return flow;
    }
};
 
// Bipartite Matching (DFS-based)
struct BipMatch {
    int n, m; vector<vector<int>> adj; vector<int> matchR;
    BipMatch(int n, int m): n(n), m(m) { adj.resize(n); matchR.assign(m, -1); }
    void addEdge(int u, int v) { adj[u].push_back(v); }
    bool bpm(int u, vector<bool>& seen) {
        for (int v : adj[u]) {
            if (!seen[v]) {
                seen[v] = true;
                if (matchR[v] < 0 || bpm(matchR[v], seen)) { matchR[v] = u; return true; }
            }
        }
        return false;
    }
    int maxMatching() {
        int result = 0; rep(u, 0, n) { vector<bool> seen(m, false); if (bpm(u, seen)) result++; }
        return result;
    }
};
 
// Minimax Template (Game Theory, recursive)
int minimax(int depth, bool isMax) {
    if (depth == 0) return 0; // Terminal condition; evaluation function here.
    int best = isMax ? -INF : INF;
    // For each move: best = isMax ? max(best, minimax(depth-1, false)) : min(best, minimax(depth-1, true));
    return best;
}
 
// ----------------- Main Function -----------------
int main() {
    fast_io;
    // Quick example usages:
    cout << "modExp(2, 10): " << modExp(2, 10) << "\n";
    DSU dsu(5); dsu.merge(0, 1); dsu.merge(1, 2);
    cout << "DSU find(2): " << dsu.find(2) << "\n";
    // Add your own test/use cases as needed.
    return 0;
}