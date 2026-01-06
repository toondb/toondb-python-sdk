# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Semi-GraphDB Overlay for Agent Memory.

Provides a lightweight graph layer on top of ToonDB's KV storage for modeling
agent memory relationships. This is NOT a full graph database - it's optimized
for typical agent memory patterns:

- Entity-to-entity relationships (user <-> conversation <-> message)
- Causal chains (action1 -> action2 -> action3)
- Reference graphs (document <- citation <- quote)

Storage Model:
--------------
Nodes: _graph/{namespace}/nodes/{node_id} -> {type, properties}
Edges: _graph/{namespace}/edges/{from_id}/{edge_type}/{to_id} -> {properties}
Index: _graph/{namespace}/index/{edge_type}/{to_id} -> [from_ids] (reverse lookup)

Performance:
------------
- Add node: O(1)
- Add edge: O(1)
- Get node: O(1)
- Get outgoing edges: O(degree)
- Get incoming edges: O(degree) via reverse index
- BFS/DFS traversal: O(V + E) for reachable subgraph

Example:
--------
    from toondb import Database
    from toondb.graph import GraphOverlay
    
    db = Database.open("./agent_memory")
    graph = GraphOverlay(db, namespace="agent_001")
    
    # Create nodes
    graph.add_node("user_1", "User", {"name": "Alice"})
    graph.add_node("conv_1", "Conversation", {"title": "Planning Session"})
    graph.add_node("msg_1", "Message", {"content": "Let's start planning"})
    
    # Create edges
    graph.add_edge("user_1", "STARTED", "conv_1")
    graph.add_edge("conv_1", "CONTAINS", "msg_1")
    graph.add_edge("user_1", "SENT", "msg_1")
    
    # Query relationships
    conversations = graph.get_edges("user_1", "STARTED")
    # [("conv_1", {"title": "Planning Session"})]
    
    # Traverse graph
    reachable = graph.bfs("user_1", max_depth=2)
    # ["user_1", "conv_1", "msg_1"]
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set, Iterator


class TraversalOrder(Enum):
    """Graph traversal order."""
    BFS = "bfs"  # Breadth-first search
    DFS = "dfs"  # Depth-first search


@dataclass
class GraphNode:
    """A node in the graph."""
    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GraphNode":
        return cls(
            id=data["id"],
            type=data["type"],
            properties=data.get("properties", {}),
        )


@dataclass
class GraphEdge:
    """An edge in the graph."""
    from_id: str
    edge_type: str
    to_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "from_id": self.from_id,
            "edge_type": self.edge_type,
            "to_id": self.to_id,
            "properties": self.properties,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GraphEdge":
        return cls(
            from_id=data["from_id"],
            edge_type=data["edge_type"],
            to_id=data["to_id"],
            properties=data.get("properties", {}),
        )


class GraphOverlay:
    """
    Lightweight graph overlay on ToonDB.
    
    Provides graph operations for agent memory without a full graph database.
    Uses the underlying KV store for persistence with O(1) node/edge operations.
    """
    
    # Key prefixes
    PREFIX = "_graph"
    
    def __init__(self, db, namespace: str = "default"):
        """
        Initialize graph overlay.
        
        Args:
            db: ToonDB Database instance
            namespace: Namespace for graph isolation (e.g., agent_id)
        """
        self._db = db
        self._namespace = namespace
        self._prefix = f"{self.PREFIX}/{namespace}".encode()
    
    def _node_key(self, node_id: str) -> bytes:
        """Key for a node."""
        return f"{self.PREFIX}/{self._namespace}/nodes/{node_id}".encode()
    
    def _edge_key(self, from_id: str, edge_type: str, to_id: str) -> bytes:
        """Key for an edge."""
        return f"{self.PREFIX}/{self._namespace}/edges/{from_id}/{edge_type}/{to_id}".encode()
    
    def _edge_prefix(self, from_id: str, edge_type: Optional[str] = None) -> bytes:
        """Prefix for outgoing edges."""
        if edge_type:
            return f"{self.PREFIX}/{self._namespace}/edges/{from_id}/{edge_type}/".encode()
        return f"{self.PREFIX}/{self._namespace}/edges/{from_id}/".encode()
    
    def _reverse_index_key(self, edge_type: str, to_id: str, from_id: str) -> bytes:
        """Key for reverse edge index."""
        return f"{self.PREFIX}/{self._namespace}/index/{edge_type}/{to_id}/{from_id}".encode()
    
    def _reverse_index_prefix(self, edge_type: str, to_id: str) -> bytes:
        """Prefix for reverse index lookup."""
        return f"{self.PREFIX}/{self._namespace}/index/{edge_type}/{to_id}/".encode()
    
    # =========================================================================
    # Node Operations
    # =========================================================================
    
    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> GraphNode:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique node identifier
            node_type: Node type label (e.g., "User", "Message", "Tool")
            properties: Optional node properties
            
        Returns:
            The created GraphNode
        """
        node = GraphNode(
            id=node_id,
            type=node_type,
            properties=properties or {},
        )
        self._db.put(self._node_key(node_id), json.dumps(node.to_dict()).encode())
        return node
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            GraphNode if found, None otherwise
        """
        data = self._db.get(self._node_key(node_id))
        if data is None:
            return None
        return GraphNode.from_dict(json.loads(data.decode()))
    
    def update_node(
        self,
        node_id: str,
        properties: Optional[Dict[str, Any]] = None,
        node_type: Optional[str] = None,
    ) -> Optional[GraphNode]:
        """
        Update a node's properties or type.
        
        Args:
            node_id: Node identifier
            properties: Properties to merge (None to skip)
            node_type: New type (None to keep existing)
            
        Returns:
            Updated GraphNode if found, None otherwise
        """
        node = self.get_node(node_id)
        if node is None:
            return None
        
        if properties:
            node.properties.update(properties)
        if node_type:
            node.type = node_type
        
        self._db.put(self._node_key(node_id), json.dumps(node.to_dict()).encode())
        return node
    
    def delete_node(self, node_id: str, cascade: bool = False) -> bool:
        """
        Delete a node from the graph.
        
        Args:
            node_id: Node identifier
            cascade: If True, also delete all connected edges
            
        Returns:
            True if deleted, False if not found
        """
        if self.get_node(node_id) is None:
            return False
        
        if cascade:
            # Delete outgoing edges
            for edge in self.get_edges(node_id):
                self.delete_edge(node_id, edge.edge_type, edge.to_id)
            
            # Delete incoming edges (using reverse index)
            for edge in self.get_incoming_edges(node_id):
                self.delete_edge(edge.from_id, edge.edge_type, node_id)
        
        self._db.delete(self._node_key(node_id))
        return True
    
    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists."""
        return self._db.get(self._node_key(node_id)) is not None
    
    # =========================================================================
    # Edge Operations
    # =========================================================================
    
    def add_edge(
        self,
        from_id: str,
        edge_type: str,
        to_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> GraphEdge:
        """
        Add an edge between two nodes.
        
        Args:
            from_id: Source node ID
            edge_type: Edge type label (e.g., "SENT", "REFERENCES", "CAUSED")
            to_id: Target node ID
            properties: Optional edge properties
            
        Returns:
            The created GraphEdge
        """
        edge = GraphEdge(
            from_id=from_id,
            edge_type=edge_type,
            to_id=to_id,
            properties=properties or {},
        )
        
        # Store edge
        self._db.put(
            self._edge_key(from_id, edge_type, to_id),
            json.dumps(edge.to_dict()).encode()
        )
        
        # Store reverse index for incoming edge queries
        self._db.put(
            self._reverse_index_key(edge_type, to_id, from_id),
            from_id.encode()
        )
        
        return edge
    
    def get_edge(
        self,
        from_id: str,
        edge_type: str,
        to_id: str,
    ) -> Optional[GraphEdge]:
        """
        Get a specific edge.
        
        Args:
            from_id: Source node ID
            edge_type: Edge type
            to_id: Target node ID
            
        Returns:
            GraphEdge if found, None otherwise
        """
        data = self._db.get(self._edge_key(from_id, edge_type, to_id))
        if data is None:
            return None
        return GraphEdge.from_dict(json.loads(data.decode()))
    
    def get_edges(
        self,
        from_id: str,
        edge_type: Optional[str] = None,
    ) -> List[GraphEdge]:
        """
        Get all outgoing edges from a node.
        
        Args:
            from_id: Source node ID
            edge_type: Optional filter by edge type
            
        Returns:
            List of GraphEdge objects
        """
        prefix = self._edge_prefix(from_id, edge_type)
        edges = []
        
        for _, value in self._db.scan_prefix_unchecked(prefix):
            edges.append(GraphEdge.from_dict(json.loads(value.decode())))
        
        return edges
    
    def get_incoming_edges(
        self,
        to_id: str,
        edge_type: Optional[str] = None,
    ) -> List[GraphEdge]:
        """
        Get all incoming edges to a node.
        
        Uses reverse index for O(degree) lookup.
        
        Args:
            to_id: Target node ID
            edge_type: Optional filter by edge type
            
        Returns:
            List of GraphEdge objects
        """
        edges = []
        
        if edge_type:
            # Query specific edge type
            prefix = self._reverse_index_prefix(edge_type, to_id)
            for key, value in self._db.scan_prefix_unchecked(prefix):
                from_id = value.decode()
                edge = self.get_edge(from_id, edge_type, to_id)
                if edge:
                    edges.append(edge)
        else:
            # Query all edge types - scan all index entries for to_id
            # This is less efficient but works
            index_prefix = f"{self.PREFIX}/{self._namespace}/index/".encode()
            for key, value in self._db.scan_prefix_unchecked(index_prefix):
                key_str = key.decode()
                parts = key_str.split("/")
                if len(parts) >= 6 and parts[4] == to_id:
                    from_id = value.decode()
                    et = parts[3]
                    edge = self.get_edge(from_id, et, to_id)
                    if edge:
                        edges.append(edge)
        
        return edges
    
    def delete_edge(
        self,
        from_id: str,
        edge_type: str,
        to_id: str,
    ) -> bool:
        """
        Delete an edge.
        
        Args:
            from_id: Source node ID
            edge_type: Edge type
            to_id: Target node ID
            
        Returns:
            True if deleted, False if not found
        """
        if self.get_edge(from_id, edge_type, to_id) is None:
            return False
        
        # Delete edge
        self._db.delete(self._edge_key(from_id, edge_type, to_id))
        
        # Delete reverse index
        self._db.delete(self._reverse_index_key(edge_type, to_id, from_id))
        
        return True
    
    # =========================================================================
    # Traversal Operations
    # =========================================================================
    
    def bfs(
        self,
        start_id: str,
        max_depth: int = 10,
        edge_types: Optional[List[str]] = None,
        node_types: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Breadth-first search from a starting node.
        
        Args:
            start_id: Starting node ID
            max_depth: Maximum traversal depth
            edge_types: Optional filter by edge types
            node_types: Optional filter by node types
            
        Returns:
            List of reachable node IDs in BFS order
        """
        return list(self._traverse(
            start_id,
            max_depth=max_depth,
            edge_types=edge_types,
            node_types=node_types,
            order=TraversalOrder.BFS,
        ))
    
    def dfs(
        self,
        start_id: str,
        max_depth: int = 10,
        edge_types: Optional[List[str]] = None,
        node_types: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Depth-first search from a starting node.
        
        Args:
            start_id: Starting node ID
            max_depth: Maximum traversal depth
            edge_types: Optional filter by edge types
            node_types: Optional filter by node types
            
        Returns:
            List of reachable node IDs in DFS order
        """
        return list(self._traverse(
            start_id,
            max_depth=max_depth,
            edge_types=edge_types,
            node_types=node_types,
            order=TraversalOrder.DFS,
        ))
    
    def _traverse(
        self,
        start_id: str,
        max_depth: int,
        edge_types: Optional[List[str]],
        node_types: Optional[List[str]],
        order: TraversalOrder,
    ) -> Iterator[str]:
        """Internal traversal implementation."""
        visited: Set[str] = set()
        
        if order == TraversalOrder.BFS:
            from collections import deque
            frontier: Any = deque([(start_id, 0)])
        else:
            frontier = [(start_id, 0)]
        
        while frontier:
            if order == TraversalOrder.BFS:
                node_id, depth = frontier.popleft()
            else:
                node_id, depth = frontier.pop()
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            
            # Check node type filter
            if node_types:
                node = self.get_node(node_id)
                if node is None or node.type not in node_types:
                    continue
            
            yield node_id
            
            if depth >= max_depth:
                continue
            
            # Get outgoing edges
            for edge in self.get_edges(node_id):
                if edge_types and edge.edge_type not in edge_types:
                    continue
                if edge.to_id not in visited:
                    frontier.append((edge.to_id, depth + 1))
    
    def shortest_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 10,
        edge_types: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes using BFS.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            max_depth: Maximum path length
            edge_types: Optional filter by edge types
            
        Returns:
            List of node IDs forming the path, or None if not reachable
        """
        from collections import deque
        
        if from_id == to_id:
            return [from_id]
        
        visited: Set[str] = {from_id}
        parent: Dict[str, str] = {}
        frontier: Any = deque([(from_id, 0)])
        
        while frontier:
            node_id, depth = frontier.popleft()
            
            if depth >= max_depth:
                continue
            
            for edge in self.get_edges(node_id):
                if edge_types and edge.edge_type not in edge_types:
                    continue
                
                next_id = edge.to_id
                if next_id in visited:
                    continue
                
                visited.add(next_id)
                parent[next_id] = node_id
                
                if next_id == to_id:
                    # Reconstruct path
                    path = [to_id]
                    current = to_id
                    while current in parent:
                        current = parent[current]
                        path.append(current)
                    return list(reversed(path))
                
                frontier.append((next_id, depth + 1))
        
        return None  # No path found
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        direction: str = "outgoing",
    ) -> List[Tuple[str, GraphEdge]]:
        """
        Get neighboring nodes with their connecting edges.
        
        Args:
            node_id: Node ID
            edge_types: Optional filter by edge types
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of (neighbor_id, edge) tuples
        """
        neighbors = []
        
        if direction in ("outgoing", "both"):
            for edge in self.get_edges(node_id):
                if edge_types and edge.edge_type not in edge_types:
                    continue
                neighbors.append((edge.to_id, edge))
        
        if direction in ("incoming", "both"):
            for edge in self.get_incoming_edges(node_id):
                if edge_types and edge.edge_type not in edge_types:
                    continue
                neighbors.append((edge.from_id, edge))
        
        return neighbors
    
    def get_nodes_by_type(
        self,
        node_type: str,
        limit: int = 100,
    ) -> List[GraphNode]:
        """
        Get all nodes of a specific type.
        
        Note: This scans all nodes, use sparingly for large graphs.
        
        Args:
            node_type: Node type to filter by
            limit: Maximum number of nodes to return
            
        Returns:
            List of GraphNode objects
        """
        prefix = f"{self.PREFIX}/{self._namespace}/nodes/".encode()
        nodes = []
        
        for _, value in self._db.scan_prefix_unchecked(prefix):
            node = GraphNode.from_dict(json.loads(value.decode()))
            if node.type == node_type:
                nodes.append(node)
                if len(nodes) >= limit:
                    break
        
        return nodes
