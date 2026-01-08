# ToonDB Python SDK v0.3.4

**Ultra-thin client for ToonDB server.**  
All business logic runs on the server.

## Architecture: Thick Server / Thin Client

```
┌────────────────────────────────────────────────┐
│         Rust Server (toondb-grpc)              │
├────────────────────────────────────────────────┤
│  • All business logic (Graph, Policy, Search)  │
│  • Vector operations (HNSW)                    │
│  • SQL parsing & execution                     │
│  • Collections & Namespaces                    │
│  • Single source of truth                      │
└────────────────────────────────────────────────┘
                       │ gRPC/IPC
                       ▼
            ┌─────────────────────┐
            │   Python SDK        │
            │   (~200 LOC)        │
            ├─────────────────────┤
            │ • Transport layer   │
            │ • Type definitions  │
            │ • Zero logic        │
            └─────────────────────┘
```

### What This SDK Contains

**This SDK is ~5,400 lines of code, consisting of:**
- **Transport Layer** (~1,000 LOC): gRPC and IPC clients
- **Type Definitions** (~500 LOC): Errors, queries, results
- **Generated Code** (~4,000 LOC): Protobuf-generated files

**This SDK does NOT contain:**
- ❌ No database logic (all server-side)
- ❌ No vector operations (all server-side)
- ❌ No SQL parsing (all server-side)
- ❌ No graph algorithms (all server-side)
- ❌ No policy evaluation (all server-side)

### Why This Design?

**Before (Fat Client - REMOVED):**
```python
# ❌ OLD: Business logic duplicated in every language
from toondb import Database, GraphOverlay, PolicyEngine

db = Database.open("./data")  # 1780 lines of logic
graph = GraphOverlay(db)      # 664 lines of duplicate code
graph.add_node("alice", "person", {"name": "Alice"})
```

**After (Thin Client - CURRENT):**
```python
# ✅ NEW: All logic on server, SDK just sends requests
from toondb import ToonDBClient

client = ToonDBClient("localhost:50051")
client.add_node("alice", "person", {"name": "Alice"})  # → Server handles it
```

**Benefits:**
- 🎯 **Single source of truth**: Fix bugs once in Rust, not 3 times
- 🔧 **3x easier maintenance**: No semantic drift between languages
- 🚀 **Faster development**: Add features once, works everywhere
- 📦 **Smaller SDK size**: 66% code reduction

---

## Installation

```bash
pip install toondb-client
```

Or from source:
```bash
cd toondb-python-sdk
pip install -e .
```

---

## Quick Start

### 1. Start ToonDB Server

```bash
# Start the gRPC server
cd toondb
cargo run -p toondb-grpc --release

# Server listens on localhost:50051
```

### 2. Connect from Python

```python
from toondb import ToonDBClient

# Connect to server
client = ToonDBClient("localhost:50051")

# Create a vector collection
client.create_collection("documents", dimension=384)

# Add documents with embeddings
documents = [
    {
        "id": "doc1",
        "content": "Machine learning tutorial",
        "embedding": [0.1, 0.2, ...],  # 384-dimensional vector
        "metadata": {"category": "AI"}
    }
]
client.add_documents("documents", documents)

# Search for similar documents
query_vector = [0.15, 0.25, ...]  # 384-dimensional
results = client.search_collection("documents", query_vector, k=5)

for result in results:
    print(f"Score: {result.score}, Content: {result.content}")
```

---

## API Reference

### ToonDBClient (gRPC Transport)

**Constructor:**
```python
client = ToonDBClient(address: str = "localhost:50051", secure: bool = False)
```

**Vector Operations:**
```python
# Create vector index
client.create_index(
    name: str,
    dimension: int,
    metric: str = "cosine"  # cosine, euclidean, dot
) -> bool

# Insert vectors
client.insert_vectors(
    index_name: str,
    ids: List[int],
    vectors: List[List[float]]
) -> bool

# Search vectors
client.search(
    index_name: str,
    query: List[float],
    k: int = 10
) -> List[SearchResult]
```

**Collection Operations:**
```python
# Create collection
client.create_collection(
    name: str,
    dimension: int,
    namespace: str = "default"
) -> bool

# Add documents
client.add_documents(
    collection_name: str,
    documents: List[Dict],
    namespace: str = "default"
) -> List[str]

# Search collection
client.search_collection(
    collection_name: str,
    query: List[float],
    k: int = 10,
    namespace: str = "default",
    filter: Optional[Dict] = None
) -> List[Document]
```

**Graph Operations:**
```python
# Add graph node
client.add_node(
    node_id: str,
    node_type: str,
    properties: Optional[Dict] = None,
    namespace: str = "default"
) -> bool

# Add graph edge
client.add_edge(
    from_id: str,
    edge_type: str,
    to_id: str,
    properties: Optional[Dict] = None,
    namespace: str = "default"
) -> bool

# Traverse graph
client.traverse(
    start_node: str,
    max_depth: int = 3,
    edge_types: Optional[List[str]] = None,
    namespace: str = "default"
) -> Tuple[List[GraphNode], List[GraphEdge]]
```

**Namespace Operations:**
```python
# Create namespace
client.create_namespace(
    name: str,
    metadata: Optional[Dict] = None
) -> bool

# List namespaces
client.list_namespaces() -> List[str]
```

**Key-Value Operations:**
```python
# Put key-value
client.put_kv(
    key: str,
    value: bytes,
    namespace: str = "default"
) -> bool

# Get value
client.get_kv(
    key: str,
    namespace: str = "default"
) -> Optional[bytes]

# Batch operations (atomic)
client.batch_put([
    (b"key1", b"value1"),
    (b"key2", b"value2"),
]) -> bool
```

**Temporal Graph Operations:**
```python
# Add time-bounded edge
client.add_temporal_edge(
    namespace: str,
    from_id: str,
    edge_type: str,
    to_id: str,
    valid_from: int,  # Unix timestamp (ms)
    valid_until: int = 0,  # 0 = no expiry
    properties: Optional[Dict] = None
) -> bool

# Query at specific point in time
edges = client.query_temporal_graph(
    namespace: str,
    node_id: str,
    mode: str = "POINT_IN_TIME",  # POINT_IN_TIME, RANGE, CURRENT
    timestamp: int = None,  # For POINT_IN_TIME
    start_time: int = None,  # For RANGE
    end_time: int = None,    # For RANGE
    edge_types: List[str] = None
) -> List[TemporalEdge]
```

**Format Utilities:**
```python
from toondb import WireFormat, ContextFormat, FormatCapabilities

# Parse format from string
wire = WireFormat.from_string("json")  # WireFormat.JSON

# Convert between formats
ctx = FormatCapabilities.wire_to_context(WireFormat.JSON)
# Returns: ContextFormat.JSON

# Check round-trip support
supports = FormatCapabilities.supports_round_trip(WireFormat.TOON)
# Returns: True (TOON and JSON support round-trip)
```

### IpcClient (Unix Socket Transport)

For local inter-process communication:

```python
from toondb import IpcClient

# Connect via Unix socket
client = IpcClient.connect("/tmp/toondb.sock")

# Same API as ToonDBClient
client.put(b"key", b"value")
value = client.get(b"key")
```

---

## Data Types

### SearchResult
```python
@dataclass
class SearchResult:
    id: int           # Vector ID
    distance: float   # Similarity distance
```

### Document
```python
@dataclass
class Document:
    id: str                      # Document ID
    content: str                 # Text content
    embedding: List[float]       # Vector embedding
    metadata: Dict[str, str]     # Metadata
```

### GraphNode
```python
@dataclass
class GraphNode:
    id: str                      # Node ID
    node_type: str               # Node type
    properties: Dict[str, str]   # Properties
```

### GraphEdge
```python
@dataclass
class GraphEdge:
    from_id: str                 # Source node
    edge_type: str               # Edge type
    to_id: str                   # Target node
    properties: Dict[str, str]   # Properties
```

### TemporalEdge
```python
@dataclass
class TemporalEdge:
    from_id: str                 # Source node
    edge_type: str               # Edge type
    to_id: str                   # Target node
    valid_from: int              # Unix timestamp (ms)
    valid_until: int             # Unix timestamp (ms), 0 = no expiry
    properties: Dict[str, str]   # Properties
```

### WireFormat
```python
class WireFormat(Enum):
    TOON = "toon"        # 40-66% fewer tokens than JSON
    JSON = "json"        # Standard compatibility
    COLUMNAR = "columnar"  # Analytics optimized
```

### ContextFormat
```python
class ContextFormat(Enum):
    TOON = "toon"        # Token-efficient for LLMs
    JSON = "json"        # Structured data
    MARKDOWN = "markdown"  # Human-readable
```

---

## Advanced Features

### Temporal Graph Queries

Temporal graphs allow you to query "What did the system know at time T?"

**Use Case: Agent Memory with Time Travel**
```python
import time
from toondb import ToonDBClient

client = ToonDBClient("localhost:50051")

# Record that door was open from 10:00 to 11:00
now = int(time.time() * 1000)
one_hour = 60 * 60 * 1000

client.add_temporal_edge(
    namespace="agent_memory",
    from_id="door_1",
    edge_type="is_open",
    to_id="room_5",
    valid_from=now,
    valid_until=now + one_hour
)

# Query: "Was door_1 open 30 minutes ago?"
thirty_min_ago = now - (30 * 60 * 1000)
edges = client.query_temporal_graph(
    namespace="agent_memory",
    node_id="door_1",
    mode="POINT_IN_TIME",
    timestamp=thirty_min_ago
)

print(f"Door was open: {len(edges) > 0}")

# Query: "What changed in the last hour?"
edges = client.query_temporal_graph(
    namespace="agent_memory",
    node_id="door_1",
    mode="RANGE",
    start_time=now - one_hour,
    end_time=now
)
```

**Query Modes:**
- `POINT_IN_TIME`: Edges valid at specific timestamp
- `RANGE`: Edges overlapping a time range
- `CURRENT`: Edges valid right now

### Atomic Multi-Operation Writes

Ensure all-or-nothing semantics across multiple operations:

```python
from toondb import ToonDBClient

client = ToonDBClient("localhost:50051")

# All operations succeed or all fail atomically
client.batch_put([
    (b"user:alice:email", b"alice@example.com"),
    (b"user:alice:age", b"30"),
    (b"user:alice:created", b"2026-01-07"),
])

# If server crashes mid-batch, none of the writes persist
```

### Format Conversion for LLM Context

Optimize token usage when sending data to LLMs:

```python
from toondb import WireFormat, ContextFormat, FormatCapabilities

# Query results come in WireFormat
query_format = WireFormat.TOON  # 40-66% fewer tokens than JSON

# Convert to ContextFormat for LLM prompt
ctx_format = FormatCapabilities.wire_to_context(query_format)
# Returns: ContextFormat.TOON

# TOON format example:
# user:alice|email:alice@example.com,age:30
# vs JSON:
# {"user":"alice","email":"alice@example.com","age":30}

# Check if format supports decode(encode(x)) = x
is_lossless = FormatCapabilities.supports_round_trip(WireFormat.TOON)
# Returns: True (TOON and JSON are lossless)
```

**Format Benefits:**
- **TOON format**: 40-66% fewer tokens than JSON → Lower LLM API costs
- **Round-trip guarantee**: `decode(encode(x)) = x` for TOON and JSON
- **Columnar format**: Optimized for analytics queries with projections

---

## Error Handling

```python
from toondb import ToonDBError, ConnectionError

try:
    client = ToonDBClient("localhost:50051")
    client.create_collection("test", dimension=128)
except ConnectionError as e:
    print(f"Cannot connect to server: {e}")
except ToonDBError as e:
    print(f"ToonDB error: {e}")
```

**Error Types:**
- `ToonDBError` - Base exception
- `ConnectionError` - Cannot connect to server
- `TransactionError` - Transaction failed
- `ProtocolError` - Protocol mismatch
- `DatabaseError` - Server-side error

---

## Advanced Usage

### Connection with TLS
```python
client = ToonDBClient("api.example.com:50051", secure=True)
```

### Batch Operations
```python
# Insert multiple vectors at once
ids = list(range(1000))
vectors = [[...] for _ in range(1000)]  # 1000 vectors
client.insert_vectors("my_index", ids, vectors)
```

### Filtered Search
```python
# Search with metadata filtering
results = client.search_collection(
    "documents",
    query_vector,
    k=10,
    filter={"category": "AI", "year": "2024"}
)
```

---

## Server Requirements

### Starting the Server

```bash
# Development mode
cd toondb
cargo run -p toondb-grpc

# Production mode (optimized)
cargo build --release -p toondb-grpc
./target/release/toondb-grpc --host 0.0.0.0 --port 50051
```

### Server Configuration

Server runs all business logic including:
- ✅ HNSW vector indexing (15x faster than ChromaDB)
- ✅ SQL query parsing and execution
- ✅ Graph traversal algorithms
- ✅ Policy evaluation
- ✅ Multi-tenant namespace isolation
- ✅ Collection management

---

## Performance

**Network Overhead:**
- gRPC: ~100-200 μs per request (local)
- IPC: ~50-100 μs per request (Unix socket)

**Batch Operations:**
- Vector insert: 50,000 vectors/sec (batch mode)
- Vector search: 20,000 queries/sec (47 μs/query)

**Recommendation:**
- Use **batch operations** for high throughput
- Use **IPC** for same-machine communication
- Use **gRPC** for distributed systems

---

## Comparison with Old Architecture

| Feature | Old (Fat Client) | New (Thin Client) |
|---------|------------------|-------------------|
| SDK Size | 15,872 LOC | 5,400 LOC (-66%) |
| Business Logic | In SDK (Python) | In Server (Rust) |
| Bug Fixes | Per language | Once in server |
| Semantic Drift | High risk | Zero risk |
| Performance | FFI overhead | Network call |
| Maintenance | 3x effort | 1x effort |

---

## Migration Guide

### From Fat Client (v0.3.3 or earlier)

**Old Code:**
```python
from toondb import Database, GraphOverlay

db = Database.open("./data")
graph = GraphOverlay(db)
graph.add_node("alice", "person", {"name": "Alice"})
```

**New Code:**
```python
from toondb import ToonDBClient

client = ToonDBClient("localhost:50051")
client.add_node("alice", "person", {"name": "Alice"})
```

**Key Changes:**
1. Replace `Database.open()` → `ToonDBClient()`
2. Start the gRPC server first
3. All operations now go through client methods
4. No more FFI/native bindings needed

---

## FAQ

**Q: Why remove the embedded Database class?**  
A: To eliminate duplicate business logic. Having SQL parsers, vector indexes, and graph algorithms in every language (Python, JS, Go) creates 3x maintenance burden and semantic drift.

**Q: What if I need offline/embedded mode?**  
A: Use the IPC client with a local server process. The server can run on the same machine with Unix socket communication (50 μs latency).

**Q: Is this slower than the old FFI-based approach?**  
A: Network overhead is ~100-200 μs. For batch operations (1000+ vectors), the throughput is identical. The server's Rust implementation is 15x faster than alternatives, offsetting any network cost.

**Q: Can I use this without a server?**  
A: No. This is a thin client that requires a ToonDB server. To deploy, run one server and connect multiple clients.

**Q: Where is the old Database class?**  
A: Removed in v0.3.4. All database operations now happen server-side via gRPC/IPC.

---

## Support

- **GitHub**: https://github.com/sushanthpy/toondb
- **Issues**: https://github.com/sushanthpy/toondb/issues
- **Docs**: https://toondb.dev

---

## License

Apache License 2.0
