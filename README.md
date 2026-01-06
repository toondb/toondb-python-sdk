# ToonDB Python SDK

[![PyPI version](https://badge.fury.io/py/toondb-client.svg)](https://badge.fury.io/py/toondb-client)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

The official Python SDK for **ToonDB** — a high-performance embedded document database with HNSW vector search, built-in multi-tenancy, and SQL support.

## Features

- ✅ **Key-Value Store** — Simple `get()`/`put()`/`delete()` operations
- ✅ **Path-Native API** — Hierarchical keys like `users/alice/email`
- ✅ **Namespace Isolation** — Type-safe multi-tenancy with `Namespace` and `Collection`
- ✅ **Hybrid Search** — Vector + BM25 keyword search with RRF fusion
- ✅ **ContextQuery Builder** — Token-aware LLM context retrieval with budgeting
- ✅ **Multi-Vector Documents** — Chunk-level embeddings with aggregation
- ✅ **Prefix Scanning** — Fast `scan_prefix()` for safe tenant-scoped iteration
- ✅ **ACID Transactions** — Full snapshot isolation with automatic commit/abort
- ✅ **Vector Search** — HNSW with bulk API (~1,600 vec/s ingestion)
- ✅ **SQL Support** — Full DDL/DML with CREATE, INSERT, SELECT, UPDATE, DELETE
- ✅ **Enhanced Error Taxonomy** — ErrorCode enum with remediation hints
- ✅ **CLI Tools** — `toondb-server`, `toondb-bulk`, `toondb-grpc-server` commands
- ✅ **Dual Mode** — Embedded (FFI) or IPC (multi-process)
- ✅ **Zero Compilation** — Pre-built binaries for Linux/macOS/Windows

## Installation

```bash
pip install toondb-client
```

> **Import Note:** The package is installed as `toondb-client` but imported as `toondb`:
> ```python
> from toondb import Database  # Correct
> ```

**Pre-built binaries included for:**
- Linux x86_64 and aarch64 (glibc ≥ 2.17)
- macOS Intel and Apple Silicon (universal2)
- Windows x64

**No Rust toolchain required!**

> ℹ️ **About the Binaries**: This Python SDK packages pre-compiled binaries from the [main ToonDB repository](https://github.com/toondb/toondb). Each wheel contains platform-specific executables (`toondb-bulk`, `toondb-server`, `toondb-grpc-server`) and native FFI libraries. See [RELEASE.md](RELEASE.md) for details on the release process.

## What's New in v0.3.3

### 🕸️ Graph Overlay for Agent Memory
Build lightweight graph structures on top of ToonDB's KV storage for agent memory:

```python
from toondb import Database, GraphOverlay

db = Database.open("./agent_db")
graph = GraphOverlay(db, namespace="agent_memory")

# Add nodes (entities, concepts, events)
graph.add_node("user_alice", "person", {"name": "Alice", "role": "developer"})
graph.add_node("conv_123", "conversation", {"topic": "ToonDB features"})
graph.add_node("action_456", "action", {"type": "code_commit", "status": "success"})

# Add edges (relationships, causality, references)
graph.add_edge("user_alice", "started", "conv_123", {"timestamp": "2026-01-05"})
graph.add_edge("conv_123", "triggered", "action_456", {"reason": "user request"})

# Retrieve nodes and edges
node = graph.get_node("user_alice")
edges = graph.get_edges("user_alice", edge_type="started")

# Graph traversal
visited = graph.bfs_traversal("user_alice", max_depth=3)  # BFS from Alice
path = graph.shortest_path("user_alice", "action_456")  # Find connection

# Get neighbors
neighbors = graph.get_neighbors("conv_123", direction="both")

# Extract subgraph
subgraph = graph.get_subgraph(["user_alice", "conv_123", "action_456"])
```

**Use Cases:**
- Agent conversation history with causal chains
- Entity relationship tracking across sessions
- Action dependency graphs for planning
- Knowledge graph construction

### 🛡️ Policy & Safety Hooks
Enforce safety policies on agent operations with pre/post triggers:

```python
from toondb import Database, PolicyEngine, PolicyAction

db = Database.open("./agent_data")
policy = PolicyEngine(db)

# Block writes to system keys from agents
@policy.before_write("system/*")
def block_system_writes(key, value, context):
    if context.get("agent_id"):
        return PolicyAction.DENY
    return PolicyAction.ALLOW

# Redact sensitive data on read
@policy.after_read("users/*/email")
def redact_emails(key, value, context):
    if context.get("redact_pii"):
        return b"[REDACTED]"
    return value

# Rate limit writes per agent
policy.add_rate_limit("write", max_per_minute=100, scope="agent_id")

# Enable audit logging
policy.enable_audit()

# Use policy-wrapped operations
policy.put(b"users/alice", b"data", context={"agent_id": "agent_001"})
```

### 🔀 Multi-Agent Tool Routing
Route tool calls to specialized agents with automatic failover:

```python
from toondb import Database, ToolDispatcher, ToolCategory, RoutingStrategy

db = Database.open("./agent_data")
dispatcher = ToolDispatcher(db)

# Register agents with capabilities
dispatcher.register_local_agent(
    "code_agent",
    capabilities=[ToolCategory.CODE, ToolCategory.GIT],
    handler=lambda tool, args: {"result": f"Processed {tool}"},
)

dispatcher.register_remote_agent(
    "search_agent",
    capabilities=[ToolCategory.SEARCH],
    endpoint="http://localhost:8001/invoke",
)

# Register tools
dispatcher.register_tool(
    name="search_code",
    description="Search codebase",
    category=ToolCategory.CODE,
)

# Invoke with automatic routing (priority, round-robin, fastest, etc.)
result = dispatcher.invoke("search_code", {"query": "auth"}, session_id="sess_001")
print(f"Routed to: {result.agent_id}, Success: {result.success}")
```

### 🕸️ Graph Overlay
Lightweight graph layer for agent memory relationships:

```python
from toondb import Database, GraphOverlay, TraversalOrder

db = Database.open("./agent_data")
graph = GraphOverlay(db)

# Add nodes (entities, concepts, events)
graph.add_node("user:alice", node_type="user", properties={"role": "admin"})
graph.add_node("project:toondb", node_type="project", properties={"status": "active"})

# Add relationships
graph.add_edge("user:alice", "project:toondb", edge_type="owns", properties={"since": "2024"})

# Traverse graph (BFS/DFS)
related = graph.bfs("user:alice", max_depth=2, edge_filter=lambda e: e.edge_type == "owns")

# Find shortest path
path = graph.shortest_path("user:alice", "project:toondb")
```

### 🔗 Unified Connection API
Single entry point with auto-detection:

```python
import toondb

# Auto-detects embedded mode from path
db = toondb.connect("./my_database")

# Auto-detects IPC mode from socket
db = toondb.connect("/tmp/toondb.sock")

# Auto-detects gRPC mode from host:port
db = toondb.connect("localhost:50051")

# Explicit mode
db = toondb.connect("./data", mode="embedded", config={"sync_mode": "full"})
```

### 🎯 Namespace Isolation
Logical database namespaces for true multi-tenancy without key prefixing:

```python
from toondb import Database, CollectionConfig, DistanceMetric

db = Database.open("./my_database")

# Create isolated namespace for tenant
ns = db.create_namespace(
    "tenant_acme",
    display_name="Acme Corporation",
    labels={"tier": "enterprise"}
)

# Create collection with immutable config
collection = ns.create_collection(
    CollectionConfig(
        name="documents",
        dimension=384,
        metric=DistanceMetric.COSINE,
        enable_hybrid_search=True,
        content_field="text"
    )
)
```

### 🔍 Hybrid Search
Combine dense vectors (HNSW) with sparse BM25 text search:

```python
# Insert documents with text and vectors
collection.insert(
    id="doc_1",
    vector=[0.1] * 384,
    metadata={"title": "Guide", "text": "ToonDB is fast"},
    content="ToonDB is a fast database"
)

# Hybrid search (vector + keyword)
results = collection.hybrid_search(
    vector=query_embedding,
    text_query="fast database",
    k=10,
    alpha=0.7,  # 70% vector, 30% keyword weight
    rrf_fusion=True  # Reciprocal Rank Fusion
)
```

### 📄 Multi-Vector Documents
Store multiple embeddings per document (e.g., title + content):

```python
# Insert document with multiple vectors
collection.insert_multi_vector(
    id="article_1",
    vectors={
        "title": title_embedding,      # List[float] of dim 384
        "abstract": abstract_embedding, # List[float] of dim 384
        "content": content_embedding    # List[float] of dim 384
    },
    metadata={"title": "Deep Learning Survey"}
)

# Search with aggregation strategy
results = collection.multi_vector_search(
    query_vectors={
        "title": query_title_embedding,
        "content": query_content_embedding
    },
    k=10,
    aggregation="max_pooling"  # or "mean_pooling", "weighted_sum"
)
```

### 🧩 Context-Aware Queries
Optimize retrieval for LLM context windows:

```python
from toondb import ContextQuery, DeduplicationStrategy

# Build context with token budgeting
context = (
    ContextQuery(collection)
    .add_vector_query(query_embedding, weight=0.7)
    .add_keyword_query("machine learning", weight=0.3)
    .with_token_budget(4000)  # Stay within model limit
    .with_deduplication(DeduplicationStrategy.SEMANTIC)
    .execute()
)

# Format for LLM
prompt = f"Context: {context.as_markdown()}\\n\\nQuestion: {user_question}"
```

## Quick Start

### Embedded Mode (Recommended)

```python
from toondb import Database

# Open database (creates if doesn't exist)
with Database.open("./my_database") as db:
    # Simple key-value
    db.put(b"user:123", b'{"name":"Alice","age":30}')
    value = db.get(b"user:123")
    print(value.decode())
    # Output: {"name":"Alice","age":30}
```

### IPC Mode (Multi-Process)

```bash
# Start server (globally available after pip install)
toondb-server --db ./my_database

# Check status
toondb-server status --db ./my_database
# Output: [Server] Running (PID: 12345)

# Stop server
toondb-server stop --db ./my_database
```

```python
from toondb import IpcClient

# Connect to running server
client = IpcClient.connect("./my_database/toondb.sock")

client.put(b"key", b"value")
value = client.get(b"key")
print(value.decode())
# Output: value
```

## SQL Database

### Create Tables and Insert Data

```python
from toondb import Database

with Database.open("./sql_db") as db:
    # Create users table
    db.execute_sql("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER
        )
    """)
    
    # Insert data
    db.execute_sql("""
        INSERT INTO users (id, name, email, age)
        VALUES (1, 'Alice', 'alice@example.com', 30)
    """)
    
    db.execute_sql("""
        INSERT INTO users (id, name, email, age)
        VALUES (2, 'Bob', 'bob@example.com', 25)
    """)
```

**Output:**
```
Table 'users' created
2 rows inserted
```

### Query with SELECT

```python
# Select all users - Returns SQLQueryResult object
result = db.execute_sql("SELECT * FROM users")
print(f"Found {len(result.rows)} users (affected: {result.rows_affected})")
for row in result.rows:
    print(row)

# Output:
# Found 2 users (affected: 0)
# {'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'age': 30}
# {'id': 2, 'name': 'Bob', 'email': 'bob@example.com', 'age': 25}

# WHERE clause
result = db.execute_sql("SELECT name, age FROM users WHERE age > 26")
for row in result.rows:
    print(f"{row['name']}: {row['age']} years old")

# Output:
# Alice: 30 years old
```

**Important:** `execute_sql()` returns a `SQLQueryResult` object with:
- `.rows` - List of dictionaries (for SELECT queries)
- `.columns` - List of column names  
- `.rows_affected` - Number of rows modified (for INSERT/UPDATE/DELETE)

### UPDATE and DELETE

```python
# Update
update_result = db.execute_sql("UPDATE users SET age = 31 WHERE name = 'Alice'")
print(f"Updated {update_result.rows_affected} rows")

# Delete
delete_result = db.execute_sql("DELETE FROM users WHERE age < 26")
print(f"Deleted {delete_result.rows_affected} rows")

# Verify
result = db.execute_sql("SELECT name, age FROM users ORDER BY age")
for row in result.rows:
    print(row)

# Output:
# Updated 1 rows
# Deleted 1 rows
# {'name': 'Alice', 'age': 31}
```

### Complex Queries

```python
# Aggregates
result = db.execute_sql("SELECT COUNT(*) as total FROM users")
print(f"Total users: {result.rows[0]['total']}")

result = db.execute_sql("SELECT AVG(age) as avg_age FROM users")
print(f"Average age: {result.rows[0]['avg_age']}")

# Complex WHERE with AND/OR
result = db.execute_sql("""
    SELECT name, age FROM users
    WHERE age > 25 AND (name = 'Alice' OR name = 'Bob')
    ORDER BY age DESC
""")

for row in result.rows:
    print(f"{row['name']}: {row['age']} years old")
```

**Note:** JOIN operations are not yet supported in the current version. They are planned for a future release.

## High-Performance Scanning

ToonDB provides `scan_batched()` for efficient large-scale scans with dramatically reduced FFI overhead.

### Performance Comparison

For 10,000 results with 500ns FFI overhead per call:
- `scan()`: 10,000 FFI calls = **5ms overhead**
- `scan_batched()`: 10 FFI calls = **5µs overhead (1000x faster!)**

### Usage Example

```python
from toondb import Database

with Database.open("./my_database") as db:
    # Regular scan - fine for small datasets
    for key, value in db.scan_prefix(b"user:"):
        print(key, value)
    
    # Batched scan - optimized for large datasets
    # Fetches 1000 results per FFI call instead of 1
    for key, value in db.scan_batched(start=b"user:", end=b"user:\xff", batch_size=1000):
        print(key, value)
```

### When to Use `scan_batched()`

- ✅ Scanning thousands or millions of records
- ✅ Data export or migration operations
- ✅ Full table scans in analytics workloads
- ✅ Batch processing pipelines

**Output:**
```
Alice bought Laptop for $999.99
Bob bought Keyboard for $75.0
```

### Aggregations

```python
# GROUP BY with aggregations
result = db.execute_sql("""
    SELECT users.name, COUNT(*) as order_count, SUM(orders.amount) as total
    FROM users
    JOIN orders ON users.id = orders.user_id
    GROUP BY users.name
    ORDER BY total DESC
""")

for row in result.rows:
    print(f"{row['name']}: {row['order_count']} orders, ${row['total']} total")
```

**Output:**
```
Alice: 2 orders, $1024.99 total
Bob: 1 orders, $75.0 total
```

## Key-Value Operations

### Basic Operations

```python
# Put
db.put(b"key", b"value")

# Get
value = db.get(b"key")
if value:
    print(value.decode())
else:
    print("Key not found")

# Delete
db.delete(b"key")
```

**Output:**
```
value
Key not found (after delete)
```

### Path Operations

```python
# Hierarchical data storage
db.put_path("users/alice/email", b"alice@example.com")
db.put_path("users/alice/age", b"30")
db.put_path("users/bob/email", b"bob@example.com")

# Retrieve by path
email = db.get_path("users/alice/email")
print(f"Alice's email: {email.decode()}")
```

**Output:**
```
Alice's email: alice@example.com
```

### Prefix Scanning ⭐

The most efficient way to iterate keys with a common prefix:

```python
# Insert multi-tenant data
db.put(b"tenants/acme/users/1", b'{"name":"Alice"}')
db.put(b"tenants/acme/users/2", b'{"name":"Bob"}')
db.put(b"tenants/acme/orders/1", b'{"total":100}')
db.put(b"tenants/globex/users/1", b'{"name":"Charlie"}')

# Scan only ACME Corp data (tenant isolation)
results = list(db.scan(b"tenants/acme/", b"tenants/acme;"))
print(f"ACME Corp has {len(results)} items:")
for key, value in results:
    print(f"  {key.decode()}: {value.decode()}")
```

**Output:**
```
ACME Corp has 3 items:
  tenants/acme/orders/1: {"total":100}
  tenants/acme/users/1: {"name":"Alice"}
  tenants/acme/users/2: {"name":"Bob"}
```

**Why use scan():**
- **Fast**: O(|prefix|) performance
- **Isolated**: Perfect for multi-tenant apps
- **Efficient**: Binary-safe iteration

## Transactions

### Automatic Transactions

```python
# Context manager handles commit/abort
with db.transaction() as txn:
    txn.put(b"account:1:balance", b"1000")
    txn.put(b"account:2:balance", b"500")
    # Commits on success, aborts on exception
```

**Output:**
```
✅ Transaction committed
```

### Manual Transaction Control

```python
txn = db.begin_transaction()
try:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")
    
    # Scan within transaction
    for key, value in txn.scan(b"key", b"key~"):
        print(f"{key.decode()}: {value.decode()}")
    
    txn.commit()
except Exception as e:
    txn.abort()
    raise
```

**Output:**
```
key1: value1
key2: value2
✅ Transaction committed
```

## Query Builder

Returns results in **TOON format** (token-optimized for LLMs):

```python
# Insert structured data
db.put(b"products/laptop", b'{"name":"Laptop","price":999,"stock":5}')
db.put(b"products/mouse", b'{"name":"Mouse","price":25,"stock":20}')

# Query with column selection
results = db.query("products/") \
    .select(["name", "price"]) \
    .limit(10) \
    .to_list()

for key, value in results:
    print(f"{key.decode()}: {value.decode()}")
```

**Output (TOON Format):**
```
products/laptop: result[1]{name,price}:Laptop,999
products/mouse: result[1]{name,price}:Mouse,25
```

**Other query methods:**
```python
first = db.query("products/").first()      # Get first result
count = db.query("products/").count()      # Count results
exists = db.query("products/").exists()    # Check existence
```

## Vector Search

### Bulk HNSW Index Building (Fast!)

```python
from toondb.bulk import bulk_build_index, bulk_query_index
import numpy as np

# Generate embeddings (10K × 768D)
embeddings = np.random.randn(10000, 768).astype(np.float32)

# Build HNSW index at ~1,600 vec/s
stats = bulk_build_index(
    embeddings,
    output="my_index.hnsw",
    m=16,
    ef_construction=100,
    metric="cosine"
)

print(f"Built {stats.vectors} vectors at {stats.rate:.0f} vec/s")
```

**Output:**
```
Built 10000 vectors at 1598 vec/s
Index size: 45.2 MB
```

### Query HNSW Index

```python
# Single query vector
query = np.random.randn(768).astype(np.float32)

results = bulk_query_index(
    index="my_index.hnsw",
    query=query,
    k=10,
    ef_search=64
)

print(f"Top {len(results)} nearest neighbors:")
for i, neighbor in enumerate(results):
    print(f"{i+1}. ID: {neighbor.id}, Distance: {neighbor.distance:.4f}")
```

**Output:**
```
Top 10 nearest neighbors:
1. ID: 3421, Distance: 0.1234
2. ID: 7892, Distance: 0.1456
3. ID: 1205, Distance: 0.1678
...
```

**Performance Comparison:**

| Method | Throughput | Use Case |
|--------|------------|----------|
| Python FFI | ~130 vec/s | Small datasets |
| Bulk API | ~1,600 vec/s | Large-scale ingestion |

---

## CLI Tools

Three CLI tools are globally available after `pip install toondb-client`:

### toondb-server

Multi-process database access via Unix domain sockets.

```bash
# Start server
toondb-server --db ./my_database

# Check status
toondb-server status --db ./my_database
# Output: [Server] Running (PID: 12345)

# Stop server
toondb-server stop --db ./my_database
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--db PATH` | `./toondb_data` | Database directory |
| `--socket PATH` | `<db>/toondb.sock` | Unix socket path |
| `--max-clients N` | `100` | Max concurrent connections |
| `--timeout-ms MS` | `30000` | Connection timeout |
| `--log-level` | `info` | trace/debug/info/warn/error |

**Wire Protocol (14 Operations):**

| OpCode | Operation | Description |
|--------|-----------|-------------|
| `0x01` | PUT | Store key-value |
| `0x02` | GET | Retrieve value |
| `0x03` | DELETE | Delete key |
| `0x04` | BEGIN_TXN | Start transaction |
| `0x05` | COMMIT_TXN | Commit transaction |
| `0x06` | ABORT_TXN | Abort transaction |
| `0x07` | QUERY | Execute query (TOON format) |
| `0x08` | CREATE_TABLE | Create table schema |
| `0x09` | PUT_PATH | Store at path |
| `0x0A` | GET_PATH | Get by path |
| `0x0B` | SCAN | Scan key range |
| `0x0C` | CHECKPOINT | Force durability |
| `0x0D` | STATS | Get server statistics |
| `0x0E` | PING | Health check |

### toondb-bulk

High-performance vector operations (~1,600 vec/s).

```bash
# Build HNSW index
toondb-bulk build-index \
    --input embeddings.npy \
    --output index.hnsw \
    --dimension 768

# Query k-NN
toondb-bulk query --index index.hnsw --query vec.raw --k 10

# Get index metadata
toondb-bulk info --index index.hnsw

# Convert formats
toondb-bulk convert --input vec.npy --output vec.raw --to-format raw_f32 --dimension 768
```

### toondb-grpc-server

gRPC server for remote vector search.

```bash
# Start gRPC server
toondb-grpc-server --host 0.0.0.0 --port 50051

# Check status
toondb-grpc-server status --port 50051
```

**gRPC Service Methods:**

| Method | Description |
|--------|-------------|
| `CreateIndex` | Create HNSW index (dimension, metric, M, ef_construction) |
| `DropIndex` | Delete an index |
| `InsertBatch` | Batch vector insertion (flat format) |
| `InsertStream` | Stream vectors for insertion |
| `Search` | Single k-NN query |
| `SearchBatch` | Batch k-NN queries |
| `GetStats` | Index statistics (num_vectors, layers, connections) |
| `HealthCheck` | Server health + version |

### Environment Variables

```bash
export TOONDB_SERVER_PATH=/path/to/toondb-server
export TOONDB_BULK_PATH=/path/to/toondb-bulk
export TOONDB_GRPC_SERVER_PATH=/path/to/toondb-grpc-server
```

---

## Complete Example: Multi-Tenant SaaS App

```python
from toondb import Database
import json

def main():
    with Database.open("./saas_db") as db:
        # Create SQL schema
        db.execute_sql("""
            CREATE TABLE IF NOT EXISTS tenants (
                id INTEGER PRIMARY KEY,
                name TEXT,
                created_at TEXT
            )
        """)
        
        # Insert tenants
        db.execute_sql("INSERT INTO tenants VALUES (1, 'ACME Corp', '2026-01-01')")
        db.execute_sql("INSERT INTO tenants VALUES (2, 'Globex Inc', '2026-01-01')")
        
        # Store tenant-specific K-V data
        db.put(b"tenants/1/users/alice", b'{"role":"admin","email":"alice@acme.com"}')
        db.put(b"tenants/1/users/bob", b'{"role":"user","email":"bob@acme.com"}')
        db.put(b"tenants/2/users/charlie", b'{"role":"admin","email":"charlie@globex.com"}')
        
        # Query SQL
        tenants = db.execute_sql("SELECT * FROM tenants ORDER BY name")
        
        for tenant in tenants:
            tenant_id = tenant['id']
            tenant_name = tenant['name']
            
            # Scan tenant-specific data (isolation)
            prefix = f"tenants/{tenant_id}/".encode()
            end = f"tenants/{tenant_id};".encode()
            users = list(db.scan(prefix, end))
            
            print(f"\n{tenant_name} ({len(users)} users):")
            for key, value in users:
                user_data = json.loads(value.decode())
                print(f"  {key.decode()}: {user_data['email']} ({user_data['role']})")

if __name__ == "__main__":
    main()
```

**Output:**
```
ACME Corp (2 users):
  tenants/1/users/alice: alice@acme.com (admin)
  tenants/1/users/bob: bob@acme.com (user)

Globex Inc (1 users):
  tenants/2/users/charlie: charlie@globex.com (admin)
```

## API Reference

### Database (Embedded Mode)

| Method | Description |
|--------|-------------|
| `Database.open(path)` | Open/create database |
| `put(key: bytes, value: bytes)` | Store key-value pair |
| `get(key: bytes) -> bytes \| None` | Retrieve value |
| `delete(key: bytes)` | Delete a key |
| `put_path(path: str, value: bytes)` | Store at hierarchical path |
| `get_path(path: str) -> bytes \| None` | Retrieve by path |
| `scan(start: bytes, end: bytes)` | Iterate key range |
| `transaction()` | Begin ACID transaction |
| `execute_sql(query: str)` | Execute SQL statement |
| `checkpoint()` | Force durability checkpoint |
| `stats()` | Get storage statistics |

### IpcClient

| Method | Description |
|--------|-------------|
| `IpcClient.connect(path)` | Connect to IPC server |
| `ping() -> float` | Check latency (ms) |
| `query(prefix: str)` | Create query builder |
| `scan(prefix: str)` | Scan keys with prefix |
| `begin_transaction()` | Start transaction |
| `commit(txn_id)` | Commit transaction |
| `abort(txn_id)` | Abort transaction |

### Bulk API

| Function | Description |
|----------|-------------|
| `bulk_build_index(embeddings, output, m, ef_construction)` | Build HNSW index (~1,600 vec/s) |
| `bulk_query_index(index, query, k, ef_search)` | Query for k nearest neighbors |
| `bulk_info(index)` | Get index metadata |

## Configuration

```python
# Custom configuration
db = Database.open("./my_db", config={
    "create_if_missing": True,
    "wal_enabled": True,
    "sync_mode": "normal",  # "full", "normal", "off"
    "memtable_size_bytes": 64 * 1024 * 1024,  # 64MB
})
```

## Error Handling

```python
from toondb import Database, ToonDBError

try:
    with Database.open("./db") as db:
        value = db.get(b"key")
        if value is None:
            print("Key not found (not an error)")
except ToonDBError as e:
    print(f"Database error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Features

### TOON Format (Token-Optimized Output Notation)

Achieve **40-66% token reduction** compared to JSON for LLM context efficiency.

#### Convert to TOON

```python
from toondb import Database

records = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
]

# Convert to TOON format
toon_str = Database.to_toon("users", records, ["name", "email"])
print(toon_str)
# Output: users[2]{name,email}:Alice,alice@example.com;Bob,bob@example.com

# Token comparison:
# JSON (compact): ~165 tokens
# TOON format:    ~70 tokens (59% reduction!)
```

#### Parse from TOON

```python
toon_str = "users[2]{name,email}:Alice,alice@ex.com;Bob,bob@ex.com"

table_name, fields, records = Database.from_toon(toon_str)
print(table_name)  # "users"
print(fields)      # ["name", "email"]
print(records)     # [{"name": "Alice", "email": "alice@ex.com"}, ...]
```

### High-Performance Batched Scanning

**1000× fewer FFI calls** for large dataset scans.

```python
with Database.open("./my_db") as db:
    # Insert test data
    for i in range(10000):
        db.put(f"item:{i:05d}".encode(), f"value:{i}".encode())
    
    txn = db.transaction()
    
    # Batched scan - dramatically faster!
    count = 0
    for key, value in txn.scan_batched(
        start=b"item:",
        end=b"item;",
        batch_size=1000  # Fetch 1000 results per FFI call
    ):
        count += 1
    
    print(f"Scanned {count} items")
    txn.abort()
```

**Performance Comparison (10,000 results, 500ns FFI overhead):**

| Method | FFI Calls | Overhead |
|--------|-----------|----------|
| `scan()` | 10,000 | 5ms |
| `scan_batched()` | 10 | 5µs (1000× faster) |

### Database Statistics & Monitoring

```python
with Database.open("./my_db") as db:
    # Perform operations
    for i in range(100):
        db.put(f"key:{i}".encode(), f"value:{i}".encode())
    
    # Get runtime statistics
    stats = db.stats()
    print(f"Keys: {stats['keys_count']}")
    print(f"Bytes written: {stats['bytes_written']}")
    print(f"Bytes read: {stats['bytes_read']}")
    print(f"Transactions committed: {stats['transactions_committed']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
```

**Available Statistics:**
- `keys_count` - Total number of keys
- `bytes_written` - Total bytes written
- `bytes_read` - Total bytes read
- `transactions_committed` - Number of committed transactions
- `transactions_aborted` - Number of aborted transactions
- `queries_executed` - Number of queries executed
- `cache_hits` - Cache hit count
- `cache_misses` - Cache miss count

### Manual Checkpoint

Force a durability checkpoint to flush all in-memory data to disk.

```python
with Database.open("./my_db") as db:
    # Bulk import
    for i in range(10000):
        db.put(f"bulk:{i}".encode(), f"data:{i}".encode())
    
    # Force checkpoint for durability
    db.checkpoint()
    print("All data flushed to disk!")
```

**Use Cases:**
- Before backup operations
- After bulk imports
- Before system shutdown
- To reduce recovery time after crash

### Python Plugin System

Full trigger system for database events with Python code.

```python
from toondb.plugins import PythonPlugin, PluginRegistry, TriggerEvent, TriggerAbort

# Define a validation plugin
plugin = PythonPlugin(
    name="user_validator",
    code='''
def on_before_insert(row: dict) -> dict:
    """Validate and transform data before insert."""
    # Normalize email
    if "email" in row:
        row["email"] = row["email"].lower().strip()
    
    # Validate age
    if row.get("age", 0) < 0:
        raise TriggerAbort("Age cannot be negative", code="INVALID_AGE")
    
    # Add timestamp
    import time
    row["created_at"] = time.time()
    
    return row

def on_after_delete(row: dict) -> dict:
    """Audit log on delete."""
    print(f"[AUDIT] Deleted: {row}")
    return row
''',
    version="1.0.0",
    packages=["numpy", "pandas"],  # Optional: required packages
    triggers={"users": ["BEFORE INSERT", "AFTER DELETE"]}
)

# Register and use
registry = PluginRegistry()
registry.register(plugin)

# Fire trigger
row = {"name": "Alice", "email": "  ALICE@EXAMPLE.COM  ", "age": 30}
result = registry.fire("users", TriggerEvent.BEFORE_INSERT, row)
print(result["email"])  # "alice@example.com"
```

**Available Trigger Events:**
- `BEFORE_INSERT`, `AFTER_INSERT`
- `BEFORE_UPDATE`, `AFTER_UPDATE`
- `BEFORE_DELETE`, `AFTER_DELETE`
- `ON_BATCH`

### Transaction Advanced Features

```python
with Database.open("./my_db") as db:
    txn = db.transaction()
    
    # Get transaction ID
    print(f"Transaction ID: {txn.id}")
    
    # Perform operations
    txn.put(b"key", b"value")
    
    # Commit returns LSN (Log Sequence Number)
    lsn = txn.commit()
    print(f"Committed at LSN: {lsn}")
    
    # Execute SQL within transaction
    txn2 = db.transaction()
    result = txn2.execute("INSERT INTO users VALUES (1, 'Alice')")
    txn2.commit()
```

### IPC Server & Multi-Process Access

Start the bundled IPC server for multi-process access:

```bash
# Start server
toondb-server --db ./my_database

# Options:
#   -d, --db <PATH>           Database directory [default: ./toondb_data]
#   -s, --socket <PATH>       Unix socket path [default: <db>/toondb.sock]
#   --max-clients <N>         Max connections [default: 100]
#   --timeout-ms <MS>         Connection timeout [default: 30000]
#   --log-level <LEVEL>       trace/debug/info/warn/error [default: info]
```

Connect from Python:

```python
from toondb import IpcClient

client = IpcClient.connect("./my_database/toondb.sock")
client.put(b"key", b"value")
value = client.get(b"key")
latency = client.ping()  # Round-trip latency in seconds
client.close()
```






## Best Practices

✅ **Use SQL for structured data** — Tables, relationships, complex queries
✅ **Use K-V for unstructured data** — JSON documents, blobs, caching
✅ **Use scan_prefix() for multi-tenancy** — Efficient prefix-based isolation
✅ **Use scan_batched() for large scans** — 1000× faster than regular scan()
✅ **Use transactions** — Atomic multi-key/multi-table operations
✅ **Use bulk API for vectors** — 12× faster than FFI for HNSW building
✅ **Use TOON format for LLMs** — 40-66% token reduction vs JSON
✅ **Always use context managers** — `with Database.open()` ensures cleanup

## Platform Support

| Platform | Wheel Tag | Notes |
|----------|-----------|-------|
| Linux x86_64 | `manylinux_2_17_x86_64` | glibc ≥ 2.17 (CentOS 7+) |
| Linux aarch64 | `manylinux_2_17_aarch64` | ARM servers (AWS Graviton) |
| macOS | `macosx_11_0_universal2` | Intel + Apple Silicon |
| Windows | `win_amd64` | Windows 10+ x64 |

## Development

```bash
# Clone repo
git clone https://github.com/toondb/toondb-python-sdk
cd toondb-python-sdk

# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v
```

## Requirements

- Python 3.9+
- NumPy (for vector operations)
- No Rust toolchain required for installation

## License

Apache License 2.0

## Links

- [Documentation](https://docs.toondb.dev/)
- [GitHub](https://github.com/toondb/toondb-python-sdk)
- [PyPI Package](https://pypi.org/project/toondb-client/)
- [Changelog](https://github.com/toondb/toondb-python-sdk/blob/main/CHANGELOG.md)
- [examples](https://github.com/toondb/toondb-python-examples)

## Support

- GitHub Issues: https://github.com/toondb/toondb-python-sdk/issues
- Email: sushanth@toondb.dev

## Author

**Sushanth** - [GitHub](https://github.com/sushanthpy)
