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
ToonDB Python SDK

A Python client for ToonDB - the database optimized for LLM context retrieval.

Provides two modes of access:
- Embedded: Direct database access via FFI (single process)
- IPC: Client-server access via Unix sockets (multi-process)
- Vector: HNSW vector search (15x faster than ChromaDB)
"""

from .ipc_client import IpcClient
from .database import Database, Transaction
from .query import Query, SQLQueryResult
from .errors import (
    ToonDBError, 
    ConnectionError, 
    TransactionError, 
    ProtocolError,
    # Error taxonomy (Task 11)
    ErrorCode,
    NamespaceError,
    NamespaceNotFoundError,
    NamespaceExistsError,
    CollectionError,
    CollectionNotFoundError,
    CollectionExistsError,
    CollectionConfigError,
    ValidationError,
    DimensionMismatchError,
    QueryError,
)
from .namespace import (
    # Namespace handle (Task 8)
    Namespace,
    NamespaceConfig,
    # Collection (Task 9)
    Collection,
    CollectionConfig,
    DistanceMetric,
    QuantizationType,
    # Search (Task 10)
    SearchRequest,
    SearchResult,
    SearchResults,
)
from .context import (
    # ContextQuery (Task 12)
    ContextQuery,
    ContextResult,
    ContextChunk,
    TokenEstimator,
    DeduplicationStrategy,
    estimate_tokens,
    split_by_tokens,
)
from .graph import (
    # Graph Overlay (Task 10)
    GraphOverlay,
    GraphNode,
    GraphEdge,
    TraversalOrder,
)
from .policy import (
    # Policy & Safety Hooks (Task 11)
    PolicyEngine,
    PolicyAction,
    PolicyTrigger,
    PolicyResult,
    PolicyContext,
    PolicyHandler,
    PatternPolicy,
    RateLimiter,
    PolicyViolation,
    # Built-in policy helpers
    deny_all,
    allow_all,
    require_agent_id,
    redact_value,
    log_and_allow,
)
from .routing import (
    # Tool Routing (Task 12)
    ToolRouter,
    AgentRegistry,
    ToolDispatcher,
    Tool,
    Agent,
    ToolCategory,
    RoutingStrategy,
    AgentStatus,
    RouteResult,
    RoutingContext,
)

# Vector search (optional - requires libtoondb_index)
try:
    from .vector import VectorIndex
except ImportError:
    VectorIndex = None

# Bulk operations (optional - requires toondb-bulk binary)
try:
    from .bulk import bulk_build_index, bulk_query_index, BulkBuildStats, QueryResult
except ImportError:
    bulk_build_index = None
    bulk_query_index = None
    BulkBuildStats = None
    QueryResult = None

# Analytics (optional - requires posthog)
try:
    from .analytics import (
        capture as capture_analytics,
        capture_error,
        shutdown as shutdown_analytics,
        track_database_open,
        track_vector_search,
        track_batch_insert,
        is_analytics_disabled,
    )
except ImportError:
    capture_analytics = None
    capture_error = None
    shutdown_analytics = None
    track_database_open = None
    track_vector_search = None
    track_batch_insert = None
    is_analytics_disabled = lambda: True

__version__ = "0.3.1"


# =============================================================================
# Unified Connection API (Task 9: Standardize Deployment Modes)
# =============================================================================

from enum import Enum
from typing import Optional, Union


class ConnectionMode(Enum):
    """ToonDB connection mode."""
    EMBEDDED = "embedded"    # Direct FFI to Rust library
    IPC = "ipc"              # Unix socket to local server
    GRPC = "grpc"            # gRPC to remote server


def connect(
    path_or_url: str,
    mode: Optional[Union[str, ConnectionMode]] = None,
    config: Optional[dict] = None,
) -> Union[Database, IpcClient]:
    """
    Connect to ToonDB with automatic mode detection.
    
    This is the unified entry point for all ToonDB connection modes.
    If mode is not specified, it auto-detects based on the path/URL:
    
    - Embedded: File paths (./data, /tmp/db, ~/toondb)
    - IPC: Unix socket paths (/tmp/toondb.sock, unix://...)
    - gRPC: URLs with grpc:// scheme or host:port format
    
    Args:
        path_or_url: Database path, socket path, or gRPC URL
        mode: Optional explicit mode ('embedded', 'ipc', 'grpc' or ConnectionMode enum)
        config: Optional configuration dict (passed to underlying client)
        
    Returns:
        Database, IpcClient, or GrpcClient depending on mode
        
    Examples:
        # Embedded mode (auto-detected from file path)
        db = toondb.connect("./my_database")
        db.put(b"key", b"value")
        
        # IPC mode (auto-detected from .sock extension)
        db = toondb.connect("/tmp/toondb.sock")
        
        # gRPC mode (auto-detected from host:port)
        db = toondb.connect("localhost:50051")
        
        # Explicit mode
        db = toondb.connect("./data", mode="embedded", config={
            "sync_mode": "full",
            "index_policy": "scan_optimized",
        })
        
        # Using enum
        db = toondb.connect("localhost:50051", mode=toondb.ConnectionMode.GRPC)
    """
    # Normalize mode to enum
    if mode is None:
        detected_mode = _detect_mode(path_or_url)
    elif isinstance(mode, str):
        try:
            detected_mode = ConnectionMode(mode.lower())
        except ValueError:
            raise ValueError(
                f"Invalid mode '{mode}'. Valid modes: embedded, ipc, grpc"
            )
    else:
        detected_mode = mode
    
    # Create appropriate client
    if detected_mode == ConnectionMode.EMBEDDED:
        return Database.open(path_or_url, config=config)
    
    elif detected_mode == ConnectionMode.IPC:
        socket_path = path_or_url
        if socket_path.startswith("unix://"):
            socket_path = socket_path[7:]  # Strip unix:// prefix
        return IpcClient(socket_path)
    
    elif detected_mode == ConnectionMode.GRPC:
        try:
            from .grpc_client import GrpcClient
            url = path_or_url
            if url.startswith("grpc://"):
                url = url[7:]  # Strip grpc:// prefix
            return GrpcClient(url)
        except ImportError:
            raise ImportError(
                "gRPC mode requires grpc dependencies. "
                "Install with: pip install toondb[grpc]"
            )
    
    else:
        raise ValueError(f"Unknown connection mode: {detected_mode}")


def _detect_mode(path_or_url: str) -> ConnectionMode:
    """Auto-detect connection mode from path/URL format."""
    import os
    
    # Explicit scheme detection
    if path_or_url.startswith("grpc://"):
        return ConnectionMode.GRPC
    if path_or_url.startswith("unix://"):
        return ConnectionMode.IPC
    
    # Socket file detection
    if path_or_url.endswith(".sock"):
        return ConnectionMode.IPC
    if "/tmp/" in path_or_url and "sock" in path_or_url.lower():
        return ConnectionMode.IPC
    
    # Host:port detection (gRPC)
    if ":" in path_or_url:
        parts = path_or_url.rsplit(":", 1)
        if len(parts) == 2:
            try:
                port = int(parts[1])
                if 1 <= port <= 65535:
                    # Looks like host:port - probably gRPC
                    return ConnectionMode.GRPC
            except ValueError:
                pass
    
    # Default to embedded for file paths
    return ConnectionMode.EMBEDDED


__all__ = [
    # Unified API (Task 9)
    "connect",
    "ConnectionMode",
    
    # Core
    "Database",
    "Transaction", 
    "Query",
    "SQLQueryResult",
    "IpcClient",
    "VectorIndex",
    
    # Namespace (Task 8)
    "Namespace",
    "NamespaceConfig",
    
    # Collection (Task 9)
    "Collection",
    "CollectionConfig",
    "DistanceMetric",
    "QuantizationType",
    
    # Search (Task 10)
    "SearchRequest",
    "SearchResult",
    "SearchResults",
    
    # Graph Overlay (Task 10)
    "GraphOverlay",
    "GraphNode",
    "GraphEdge",
    "TraversalOrder",
    
    # Policy & Safety Hooks (Task 11)
    "PolicyEngine",
    "PolicyAction",
    "PolicyTrigger",
    "PolicyResult",
    "PolicyContext",
    "PolicyHandler",
    "PatternPolicy",
    "RateLimiter",
    "PolicyViolation",
    "deny_all",
    "allow_all",
    "require_agent_id",
    "redact_value",
    "log_and_allow",
    
    # Tool Routing (Task 12)
    "ToolRouter",
    "AgentRegistry",
    "ToolDispatcher",
    "Tool",
    "Agent",
    "ToolCategory",
    "RoutingStrategy",
    "AgentStatus",
    "RouteResult",
    "RoutingContext",
    
    # ContextQuery (Task 12)
    "ContextQuery",
    "ContextResult",
    "ContextChunk",
    "TokenEstimator",
    "DeduplicationStrategy",
    "estimate_tokens",
    "split_by_tokens",
    
    # Bulk operations
    "bulk_build_index",
    "bulk_query_index",
    "BulkBuildStats",
    "QueryResult",
    
    # Analytics (disabled with TOONDB_DISABLE_ANALYTICS=true)
    "capture_analytics",
    "capture_error",
    "shutdown_analytics",
    "track_database_open",
    "track_vector_search",
    "track_batch_insert",
    "is_analytics_disabled",
    
    # Errors
    "ToonDBError",
    "ConnectionError",
    "TransactionError",
    "ProtocolError",
    "ErrorCode",
    "NamespaceError",
    "NamespaceNotFoundError",
    "NamespaceExistsError",
    "CollectionError",
    "CollectionNotFoundError",
    "CollectionExistsError",
    "CollectionConfigError",
    "ValidationError",
    "DimensionMismatchError",
    "QueryError",
]
