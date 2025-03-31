import numpy as np
from collections import defaultdict, deque

class GraphFloat:
    def __init__(self):
        self.graph = defaultdict(dict)
        self.nodes = set()
        self.t_edges = {}

    def add_nodes(self, num):
        idx = len(self.nodes)
        for node in range(idx, idx + num):
            self.nodes.add(node)
        return idx
    
    def add_grid_nodes(self, shape):
        h, w = shape
        nodeids = np.arange(len(self.nodes), len(self.nodes) + h * w).reshape(h, w)
        self.nodes.update(nodeids.flatten())
        return nodeids

    def add_edge(self, u, v, capacity_uv, capacity_vu):
        self.graph[u][v] = self.graph[u].get(v, 0) + capacity_uv
        self.graph[v][u] = self.graph[v].get(u, 0) + capacity_vu

    def add_tedge(self, node, capacity_to_source, capacity_to_sink):
        self.t_edges[node] = (capacity_to_source, capacity_to_sink)

    def bfs(self, residual_graph, source, sink, parent):
        visited = {node: False for node in self.nodes}
        queue = deque([source])
        visited[source] = True

        while queue:
            u = queue.popleft()
            for v, capacity in residual_graph[u].items():
                if not visited[v] and capacity > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        return True
        return False

    def maxflow(self):
        source, sink = -1, -2
        
        self.nodes.add(source)
        self.nodes.add(sink)
        
        residual_graph = defaultdict(dict)

        for u in self.nodes:
            residual_graph[u] = self.graph[u].copy()

        residual_graph[source] = defaultdict(int)
        residual_graph[sink] = defaultdict(int)

        for node, (cap_source, cap_sink) in self.t_edges.items():
            residual_graph[source][node] = cap_source
            residual_graph[node][sink] = cap_sink

        parent = {}
        max_flow = 0

        while self.bfs(residual_graph, source, sink, parent):
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, residual_graph[parent[s]][s])
                s = parent[s]

            max_flow += path_flow
            v = sink
            while v != source:
                u = parent[v]
                residual_graph[u][v] -= path_flow
                residual_graph[v][u] = residual_graph[v].get(u, 0) + path_flow
                v = parent[v]

        self.residual_graph = residual_graph
        self.source = source

        return max_flow

    def get_segment(self, node):
        visited = {n: False for n in self.nodes}
        queue = deque([self.source])
        visited[self.source] = True

        while queue:
            u = queue.popleft()
            for v, capacity in self.residual_graph[u].items():
                if not visited[v] and capacity > 0:
                    queue.append(v)
                    visited[v] = True

        return not visited.get(node, False)

    def get_grid_segments(self, nodeids):
        h, w = nodeids.shape
        segments = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                segments[y, x] = self.get_segment(nodeids[y, x])
        return segments
