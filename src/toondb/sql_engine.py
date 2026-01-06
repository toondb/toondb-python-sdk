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
SQL Engine for ToonDB Python SDK.

Provides SQL support on top of the KV storage backend.
Tables are stored as:
  - Schema: _sql/tables/{table_name}/schema -> JSON schema definition
  - Rows: _sql/tables/{table_name}/rows/{row_id} -> JSON row data
  - Indexes: _sql/tables/{table_name}/indexes/{index_name} -> index data
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from .query import SQLQueryResult


@dataclass
class Column:
    """SQL column definition."""
    name: str
    type: str  # INT, TEXT, FLOAT, BOOL, BLOB
    nullable: bool = True
    primary_key: bool = False
    default: Any = None


@dataclass
class TableSchema:
    """SQL table schema."""
    name: str
    columns: List[Column] = field(default_factory=list)
    primary_key: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "columns": [
                {
                    "name": c.name,
                    "type": c.type,
                    "nullable": c.nullable,
                    "primary_key": c.primary_key,
                    "default": c.default
                }
                for c in self.columns
            ],
            "primary_key": self.primary_key
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TableSchema":
        columns = [
            Column(
                name=c["name"],
                type=c["type"],
                nullable=c.get("nullable", True),
                primary_key=c.get("primary_key", False),
                default=c.get("default")
            )
            for c in data.get("columns", [])
        ]
        return cls(
            name=data["name"],
            columns=columns,
            primary_key=data.get("primary_key")
        )


class SQLParser:
    """Simple SQL parser for basic DDL and DML."""
    
    @staticmethod
    def parse(sql: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse SQL and return (operation, parsed_data).
        
        Returns:
            Tuple of (operation_type, parsed_info)
            operation_type: CREATE_TABLE, DROP_TABLE, CREATE_INDEX, DROP_INDEX,
                            INSERT, SELECT, UPDATE, DELETE
        """
        sql = sql.strip()
        upper = sql.upper()
        
        if upper.startswith("CREATE TABLE"):
            return SQLParser._parse_create_table(sql)
        elif upper.startswith("CREATE INDEX"):
            return SQLParser._parse_create_index(sql)
        elif upper.startswith("DROP TABLE"):
            return SQLParser._parse_drop_table(sql)
        elif upper.startswith("DROP INDEX"):
            return SQLParser._parse_drop_index(sql)
        elif upper.startswith("INSERT"):
            return SQLParser._parse_insert(sql)
        elif upper.startswith("SELECT"):
            return SQLParser._parse_select(sql)
        elif upper.startswith("UPDATE"):
            return SQLParser._parse_update(sql)
        elif upper.startswith("DELETE"):
            return SQLParser._parse_delete(sql)
        else:
            raise ValueError(f"Unsupported SQL statement: {sql[:50]}")
    
    @staticmethod
    def _parse_create_index(sql: str) -> Tuple[str, Dict]:
        """
        Parse CREATE INDEX statement.
        
        Syntax: CREATE INDEX idx_name ON table_name(column_name)
        """
        match = re.match(
            r'CREATE\s+INDEX\s+(\w+)\s+ON\s+(\w+)\s*\(\s*(\w+)\s*\)',
            sql,
            re.IGNORECASE
        )
        if not match:
            raise ValueError(f"Invalid CREATE INDEX syntax: {sql}")
        
        index_name = match.group(1)
        table = match.group(2)
        column = match.group(3)
        
        return "CREATE_INDEX", {
            "index_name": index_name,
            "table": table,
            "column": column
        }
    
    @staticmethod
    def _parse_drop_index(sql: str) -> Tuple[str, Dict]:
        """
        Parse DROP INDEX statement.
        
        Syntax: DROP INDEX idx_name ON table_name
        """
        match = re.match(
            r'DROP\s+INDEX\s+(\w+)\s+ON\s+(\w+)',
            sql,
            re.IGNORECASE
        )
        if not match:
            raise ValueError(f"Invalid DROP INDEX syntax: {sql}")
        
        index_name = match.group(1)
        table = match.group(2)
        
        return "DROP_INDEX", {
            "index_name": index_name,
            "table": table
        }
    
    @staticmethod
    def _parse_create_table(sql: str) -> Tuple[str, Dict]:
        """Parse CREATE TABLE statement."""
        # CREATE TABLE table_name (col1 TYPE, col2 TYPE, ...)
        match = re.match(
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*)\)',
            sql,
            re.IGNORECASE | re.DOTALL
        )
        if not match:
            raise ValueError(f"Invalid CREATE TABLE: {sql}")
        
        table_name = match.group(1)
        cols_str = match.group(2)
        
        columns = []
        primary_key = None
        
        # Split by comma, but not inside parentheses
        col_defs = SQLParser._split_columns(cols_str)
        
        for col_def in col_defs:
            col_def = col_def.strip()
            if not col_def:
                continue
            
            # Check for PRIMARY KEY constraint
            if col_def.upper().startswith("PRIMARY KEY"):
                pk_match = re.match(r'PRIMARY\s+KEY\s*\((\w+)\)', col_def, re.IGNORECASE)
                if pk_match:
                    primary_key = pk_match.group(1)
                continue
            
            # Parse column: name TYPE [PRIMARY KEY] [NOT NULL] [DEFAULT value]
            parts = col_def.split()
            if len(parts) < 2:
                continue
            
            col_name = parts[0]
            col_type = parts[1].upper()
            
            # Normalize types
            if col_type in ("INTEGER", "INT", "BIGINT", "SMALLINT"):
                col_type = "INT"
            elif col_type in ("VARCHAR", "CHAR", "STRING", "TEXT"):
                col_type = "TEXT"
            elif col_type in ("REAL", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC"):
                col_type = "FLOAT"
            elif col_type in ("BOOLEAN", "BOOL"):
                col_type = "BOOL"
            elif col_type in ("BLOB", "BYTES", "BINARY"):
                col_type = "BLOB"
            
            col_upper = col_def.upper()
            is_pk = "PRIMARY KEY" in col_upper
            nullable = "NOT NULL" not in col_upper
            
            if is_pk:
                primary_key = col_name
            
            columns.append(Column(
                name=col_name,
                type=col_type,
                nullable=nullable,
                primary_key=is_pk
            ))
        
        return "CREATE_TABLE", {
            "table": table_name,
            "columns": columns,
            "primary_key": primary_key
        }
    
    @staticmethod
    def _split_columns(cols_str: str) -> List[str]:
        """Split column definitions, handling parentheses."""
        result = []
        current = ""
        depth = 0
        
        for char in cols_str:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                result.append(current)
                current = ""
            else:
                current += char
        
        if current.strip():
            result.append(current)
        
        return result
    
    @staticmethod
    def _parse_drop_table(sql: str) -> Tuple[str, Dict]:
        """Parse DROP TABLE statement."""
        match = re.match(
            r'DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)',
            sql,
            re.IGNORECASE
        )
        if not match:
            raise ValueError(f"Invalid DROP TABLE: {sql}")
        
        return "DROP_TABLE", {"table": match.group(1)}
    
    @staticmethod
    def _parse_insert(sql: str) -> Tuple[str, Dict]:
        """Parse INSERT statement."""
        # INSERT INTO table (col1, col2) VALUES (val1, val2)
        # or INSERT INTO table VALUES (val1, val2)
        
        # With column names
        match = re.match(
            r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\((.+)\)',
            sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if match:
            table = match.group(1)
            columns = [c.strip() for c in match.group(2).split(',')]
            values = SQLParser._parse_values(match.group(3))
            return "INSERT", {"table": table, "columns": columns, "values": values}
        
        # Without column names
        match = re.match(
            r'INSERT\s+INTO\s+(\w+)\s+VALUES\s*\((.+)\)',
            sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if match:
            table = match.group(1)
            values = SQLParser._parse_values(match.group(2))
            return "INSERT", {"table": table, "columns": None, "values": values}
        
        raise ValueError(f"Invalid INSERT: {sql}")
    
    @staticmethod
    def _parse_values(values_str: str) -> List[Any]:
        """Parse value list from VALUES clause."""
        values = []
        current = ""
        in_string = False
        string_char = None
        
        for char in values_str:
            if char in ('"', "'") and not in_string:
                in_string = True
                string_char = char
                current += char
            elif char == string_char and in_string:
                in_string = False
                string_char = None
                current += char
            elif char == ',' and not in_string:
                values.append(SQLParser._parse_value(current.strip()))
                current = ""
            else:
                current += char
        
        if current.strip():
            values.append(SQLParser._parse_value(current.strip()))
        
        return values
    
    @staticmethod
    def _parse_value(val_str: str) -> Any:
        """Parse a single value."""
        val_str = val_str.strip()
        
        if not val_str or val_str.upper() == "NULL":
            return None
        
        # String literals
        if (val_str.startswith("'") and val_str.endswith("'")) or \
           (val_str.startswith('"') and val_str.endswith('"')):
            return val_str[1:-1]
        
        # Boolean
        if val_str.upper() == "TRUE":
            return True
        if val_str.upper() == "FALSE":
            return False
        
        # Numbers
        try:
            if '.' in val_str:
                return float(val_str)
            return int(val_str)
        except ValueError:
            return val_str
    
    @staticmethod
    def _parse_select(sql: str) -> Tuple[str, Dict]:
        """Parse SELECT statement."""
        # SELECT cols FROM table [WHERE ...] [ORDER BY ...] [LIMIT ...]
        
        # Extract main parts using regex
        pattern = r'''
            SELECT\s+(.+?)           # columns
            \s+FROM\s+(\w+)          # table
            (?:\s+WHERE\s+(.+?))?    # optional WHERE
            (?:\s+ORDER\s+BY\s+(.+?))?  # optional ORDER BY
            (?:\s+LIMIT\s+(\d+))?    # optional LIMIT
            (?:\s+OFFSET\s+(\d+))?   # optional OFFSET
            \s*$
        '''
        
        match = re.match(pattern, sql, re.IGNORECASE | re.DOTALL | re.VERBOSE)
        
        if not match:
            # Simpler pattern for basic SELECT
            simple_match = re.match(
                r'SELECT\s+(.+?)\s+FROM\s+(\w+)',
                sql,
                re.IGNORECASE | re.DOTALL
            )
            if not simple_match:
                raise ValueError(f"Invalid SELECT: {sql}")
            
            columns_str = simple_match.group(1)
            table = simple_match.group(2)
            
            # Parse rest of the query
            rest = sql[simple_match.end():].strip()
            where_clause = None
            order_by = None
            limit = None
            offset = None
            
            # Extract WHERE
            where_match = re.search(r'\bWHERE\s+(.+?)(?:\s+ORDER|\s+LIMIT|\s+OFFSET|$)', rest, re.IGNORECASE)
            if where_match:
                where_clause = where_match.group(1).strip()
            
            # Extract ORDER BY
            order_match = re.search(r'\bORDER\s+BY\s+(.+?)(?:\s+LIMIT|\s+OFFSET|$)', rest, re.IGNORECASE)
            if order_match:
                order_by = order_match.group(1).strip()
            
            # Extract LIMIT
            limit_match = re.search(r'\bLIMIT\s+(\d+)', rest, re.IGNORECASE)
            if limit_match:
                limit = int(limit_match.group(1))
            
            # Extract OFFSET
            offset_match = re.search(r'\bOFFSET\s+(\d+)', rest, re.IGNORECASE)
            if offset_match:
                offset = int(offset_match.group(1))
        else:
            columns_str = match.group(1)
            table = match.group(2)
            where_clause = match.group(3)
            order_by = match.group(4)
            limit = int(match.group(5)) if match.group(5) else None
            offset = int(match.group(6)) if match.group(6) else None
        
        # Parse columns
        if columns_str.strip() == "*":
            columns = ["*"]
        else:
            columns = [c.strip() for c in columns_str.split(',')]
        
        # Parse WHERE clause
        conditions = []
        if where_clause:
            conditions = SQLParser._parse_where(where_clause)
        
        # Parse ORDER BY
        order = []
        if order_by:
            for part in order_by.split(','):
                part = part.strip()
                if part.upper().endswith(" DESC"):
                    order.append((part[:-5].strip(), "DESC"))
                elif part.upper().endswith(" ASC"):
                    order.append((part[:-4].strip(), "ASC"))
                else:
                    order.append((part, "ASC"))
        
        return "SELECT", {
            "table": table,
            "columns": columns,
            "where": conditions,
            "order_by": order,
            "limit": limit,
            "offset": offset
        }
    
    @staticmethod
    def _parse_where(where_clause: str) -> List[Tuple[str, str, Any]]:
        """Parse WHERE clause into list of (column, operator, value)."""
        conditions = []
        
        # Split by AND (simple case, doesn't handle nested OR)
        parts = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip()
            
            # Match: column operator value
            match = re.match(r'(\w+)\s*(=|!=|<>|>=|<=|>|<|LIKE|NOT\s+LIKE)\s*(.+)', part, re.IGNORECASE)
            if match:
                col = match.group(1)
                op = match.group(2).upper().replace(" ", "_")
                if op == "<>":
                    op = "!="
                val = SQLParser._parse_value(match.group(3))
                conditions.append((col, op, val))
        
        return conditions
    
    @staticmethod
    def _parse_update(sql: str) -> Tuple[str, Dict]:
        """Parse UPDATE statement."""
        # UPDATE table SET col1=val1, col2=val2 [WHERE ...]
        match = re.match(
            r'UPDATE\s+(\w+)\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?$',
            sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if not match:
            raise ValueError(f"Invalid UPDATE: {sql}")
        
        table = match.group(1)
        set_clause = match.group(2)
        where_clause = match.group(3)
        
        # Parse SET clause
        updates = {}
        for part in set_clause.split(','):
            eq_match = re.match(r'\s*(\w+)\s*=\s*(.+)\s*', part)
            if eq_match:
                col = eq_match.group(1)
                val = SQLParser._parse_value(eq_match.group(2))
                updates[col] = val
        
        conditions = []
        if where_clause:
            conditions = SQLParser._parse_where(where_clause)
        
        return "UPDATE", {"table": table, "updates": updates, "where": conditions}
    
    @staticmethod
    def _parse_delete(sql: str) -> Tuple[str, Dict]:
        """Parse DELETE statement."""
        match = re.match(
            r'DELETE\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+))?$',
            sql,
            re.IGNORECASE | re.DOTALL
        )
        
        if not match:
            raise ValueError(f"Invalid DELETE: {sql}")
        
        table = match.group(1)
        where_clause = match.group(2)
        
        conditions = []
        if where_clause:
            conditions = SQLParser._parse_where(where_clause)
        
        return "DELETE", {"table": table, "where": conditions}


class SQLExecutor:
    """Execute SQL operations using the KV backend."""
    
    # Key prefixes for SQL data
    TABLE_PREFIX = b"_sql/tables/"
    SCHEMA_SUFFIX = b"/schema"
    ROWS_PREFIX = b"/rows/"
    INDEXES_PREFIX = b"/indexes/"
    INDEX_META_SUFFIX = b"/meta"
    
    def __init__(self, db):
        """Initialize with a Database instance."""
        self._db = db
    
    # =========================================================================
    # Index Infrastructure
    # =========================================================================
    
    def _index_key(self, table: str, index_name: str, value: Any, row_id: str) -> bytes:
        """Get key for an index entry: _sql/tables/{table}/indexes/{idx}/{value}/{row_id}."""
        # Encode value as sortable string for range queries
        value_str = self._encode_index_value(value)
        return (
            self.TABLE_PREFIX + table.encode() + 
            self.INDEXES_PREFIX + index_name.encode() + 
            b"/" + value_str.encode() + b"/" + row_id.encode()
        )
    
    def _index_prefix(self, table: str, index_name: str) -> bytes:
        """Get prefix for all entries in an index."""
        return (
            self.TABLE_PREFIX + table.encode() + 
            self.INDEXES_PREFIX + index_name.encode() + b"/"
        )
    
    def _index_value_prefix(self, table: str, index_name: str, value: Any) -> bytes:
        """Get prefix for all entries with a specific value in an index."""
        value_str = self._encode_index_value(value)
        return (
            self.TABLE_PREFIX + table.encode() + 
            self.INDEXES_PREFIX + index_name.encode() + 
            b"/" + value_str.encode() + b"/"
        )
    
    def _encode_index_value(self, value: Any) -> str:
        """Encode a value for use in index keys (sortable string format)."""
        if value is None:
            return "__null__"
        elif isinstance(value, bool):
            return f"b:{1 if value else 0}"
        elif isinstance(value, int):
            # Zero-pad integers for proper string sorting
            return f"i:{value:020d}"
        elif isinstance(value, float):
            return f"f:{value:030.15f}"
        else:
            # String values - escape slashes
            return f"s:{str(value).replace('/', '__SLASH__')}"
    
    def _index_meta_key(self, table: str, index_name: str) -> bytes:
        """Get key for index metadata."""
        return (
            self.TABLE_PREFIX + table.encode() + 
            self.INDEXES_PREFIX + index_name.encode() + 
            self.INDEX_META_SUFFIX
        )
    
    def _get_indexes(self, table: str) -> Dict[str, str]:
        """Get all indexes for a table. Returns {index_name: column_name}."""
        prefix = self.TABLE_PREFIX + table.encode() + self.INDEXES_PREFIX
        indexes = {}
        
        for key, value in self._db.scan_prefix(prefix):
            if key.endswith(self.INDEX_META_SUFFIX):
                # Extract index name from key
                parts = key.decode().split("/")
                if len(parts) >= 5:  # _sql/tables/{table}/indexes/{idx}/meta
                    idx_name = parts[4]
                    meta = json.loads(value.decode())
                    indexes[idx_name] = meta.get("column", idx_name)
        
        return indexes
    
    def _update_index(self, table: str, index_name: str, column: str, 
                      old_row: Optional[Dict], new_row: Optional[Dict], row_id: str):
        """Update index entries when a row changes."""
        old_value = old_row.get(column) if old_row else None
        new_value = new_row.get(column) if new_row else None
        
        # Remove old index entry if value changed
        if old_row and (new_row is None or old_value != new_value):
            old_key = self._index_key(table, index_name, old_value, row_id)
            self._db.delete(old_key)
        
        # Add new index entry if there's a new row
        if new_row and (old_row is None or old_value != new_value):
            new_key = self._index_key(table, index_name, new_value, row_id)
            self._db.put(new_key, row_id.encode())
    
    def _lookup_by_index(self, table: str, column: str, value: Any) -> List[str]:
        """Look up row IDs by index value. Returns list of row_ids."""
        indexes = self._get_indexes(table)
        
        # Find index for this column
        index_name = None
        for idx_name, col in indexes.items():
            if col == column:
                index_name = idx_name
                break
        
        if index_name is None:
            return []  # No index available
        
        # Scan index entries for this value
        prefix = self._index_value_prefix(table, index_name, value)
        row_ids = []
        
        for key, value_bytes in self._db.scan_prefix(prefix):
            row_ids.append(value_bytes.decode())
        
        return row_ids
    
    def _has_index_for_column(self, table: str, column: str) -> bool:
        """Check if an index exists for a column."""
        indexes = self._get_indexes(table)
        return any(col == column for col in indexes.values())
    
    def _find_indexed_equality_condition(self, table: str, 
                                          conditions: List[Tuple]) -> Optional[Tuple[str, Any]]:
        """
        Find a WHERE condition that can use an index.
        Returns (column, value) if found, None otherwise.
        Only considers equality conditions (=).
        """
        for condition in conditions:
            col, op, val = condition
            if op == "=" and self._has_index_for_column(table, col):
                return (col, val)
        return None
    
    def execute(self, sql: str) -> SQLQueryResult:
        """Execute a SQL statement."""
        operation, data = SQLParser.parse(sql)
        
        if operation == "CREATE_TABLE":
            return self._create_table(data)
        elif operation == "DROP_TABLE":
            return self._drop_table(data)
        elif operation == "CREATE_INDEX":
            return self._create_index(data)
        elif operation == "DROP_INDEX":
            return self._drop_index(data)
        elif operation == "INSERT":
            return self._insert(data)
        elif operation == "SELECT":
            return self._select(data)
        elif operation == "UPDATE":
            return self._update(data)
        elif operation == "DELETE":
            return self._delete(data)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _schema_key(self, table: str) -> bytes:
        """Get the key for a table's schema."""
        return self.TABLE_PREFIX + table.encode() + self.SCHEMA_SUFFIX
    
    def _row_key(self, table: str, row_id: str) -> bytes:
        """Get the key for a specific row."""
        return self.TABLE_PREFIX + table.encode() + self.ROWS_PREFIX + row_id.encode()
    
    def _row_prefix(self, table: str) -> bytes:
        """Get the prefix for all rows in a table."""
        return self.TABLE_PREFIX + table.encode() + self.ROWS_PREFIX
    
    def _get_schema(self, table: str) -> Optional[TableSchema]:
        """Get table schema."""
        data = self._db.get(self._schema_key(table))
        if data is None:
            return None
        return TableSchema.from_dict(json.loads(data.decode()))
    
    def _create_table(self, data: Dict) -> SQLQueryResult:
        """Create a new table."""
        table = data["table"]
        columns = data["columns"]
        primary_key = data.get("primary_key")
        
        # Check if table exists
        if self._get_schema(table) is not None:
            raise ValueError(f"Table '{table}' already exists")
        
        schema = TableSchema(name=table, columns=columns, primary_key=primary_key)
        
        # Store schema
        self._db.put(
            self._schema_key(table),
            json.dumps(schema.to_dict()).encode()
        )
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=0)
    
    def _drop_table(self, data: Dict) -> SQLQueryResult:
        """Drop a table."""
        table = data["table"]
        
        # Delete all indexes first
        indexes = self._get_indexes(table)
        for idx_name in indexes:
            idx_prefix = self._index_prefix(table, idx_name)
            for key, _ in self._db.scan_prefix(idx_prefix):
                self._db.delete(key)
            self._db.delete(self._index_meta_key(table, idx_name))
        
        # Delete all rows
        prefix = self._row_prefix(table)
        rows_deleted = 0
        for key, _ in self._db.scan_prefix(prefix):
            self._db.delete(key)
            rows_deleted += 1
        
        # Delete schema
        self._db.delete(self._schema_key(table))
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=rows_deleted)
    
    def _create_index(self, data: Dict) -> SQLQueryResult:
        """
        Create a secondary index on a column.
        
        CREATE INDEX idx_name ON table(column)
        
        This builds the index by scanning existing rows.
        """
        index_name = data["index_name"]
        table = data["table"]
        column = data["column"]
        
        schema = self._get_schema(table)
        if schema is None:
            raise ValueError(f"Table '{table}' does not exist")
        
        # Check column exists
        if not any(c.name == column for c in schema.columns):
            raise ValueError(f"Column '{column}' does not exist in table '{table}'")
        
        # Check index doesn't already exist
        if self._db.get(self._index_meta_key(table, index_name)) is not None:
            raise ValueError(f"Index '{index_name}' already exists on table '{table}'")
        
        # Store index metadata
        meta = {"column": column, "table": table}
        self._db.put(
            self._index_meta_key(table, index_name),
            json.dumps(meta).encode()
        )
        
        # Build index from existing rows
        prefix = self._row_prefix(table)
        indexed_count = 0
        
        for _, value in self._db.scan_prefix(prefix):
            row = json.loads(value.decode())
            row_id = row.get("_id", "")
            col_value = row.get(column)
            
            # Add index entry
            idx_key = self._index_key(table, index_name, col_value, row_id)
            self._db.put(idx_key, row_id.encode())
            indexed_count += 1
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=indexed_count)
    
    def _drop_index(self, data: Dict) -> SQLQueryResult:
        """Drop a secondary index."""
        index_name = data["index_name"]
        table = data.get("table")
        
        # If table not specified, find it from index meta
        if table is None:
            # Search for index in all tables - not ideal but works
            raise ValueError("DROP INDEX requires table name: DROP INDEX idx_name ON table")
        
        # Delete all index entries
        idx_prefix = self._index_prefix(table, index_name)
        deleted = 0
        for key, _ in self._db.scan_prefix(idx_prefix):
            self._db.delete(key)
            deleted += 1
        
        # Delete index metadata
        self._db.delete(self._index_meta_key(table, index_name))
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=deleted)
    
    def _insert(self, data: Dict) -> SQLQueryResult:
        """Insert a row and maintain secondary indexes."""
        table = data["table"]
        columns = data["columns"]
        values = data["values"]
        
        schema = self._get_schema(table)
        if schema is None:
            raise ValueError(f"Table '{table}' does not exist")
        
        # If no columns specified, use schema order
        if columns is None:
            columns = [c.name for c in schema.columns]
        
        if len(columns) != len(values):
            raise ValueError(f"Column count ({len(columns)}) doesn't match value count ({len(values)})")
        
        # Create row dict
        row = dict(zip(columns, values))
        
        # Generate row ID (use primary key value or UUID)
        if schema.primary_key and schema.primary_key in row:
            row_id = str(row[schema.primary_key])
        else:
            row_id = str(uuid.uuid4())
        
        # Add row ID to row data
        row["_id"] = row_id
        
        # Store row
        self._db.put(
            self._row_key(table, row_id),
            json.dumps(row).encode()
        )
        
        # Update secondary indexes
        indexes = self._get_indexes(table)
        for idx_name, idx_col in indexes.items():
            self._update_index(table, idx_name, idx_col, None, row, row_id)
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=1)
    
    def _select(self, data: Dict) -> SQLQueryResult:
        """Select rows."""
        table = data["table"]
        columns = data["columns"]
        conditions = data.get("where", [])
        order_by = data.get("order_by", [])
        limit = data.get("limit")
        offset = data.get("offset", 0)
        
        schema = self._get_schema(table)
        if schema is None:
            raise ValueError(f"Table '{table}' does not exist")
        
        # Get column names
        if columns == ["*"]:
            columns = [c.name for c in schema.columns]
        
        # Scan all rows
        prefix = self._row_prefix(table)
        rows = []
        
        for key, value in self._db.scan_prefix(prefix):
            row = json.loads(value.decode())
            
            # Apply WHERE conditions
            if self._matches_conditions(row, conditions):
                # Project columns
                projected = {col: row.get(col) for col in columns if col in row}
                rows.append(projected)
        
        # Apply ORDER BY
        if order_by:
            for col, direction in reversed(order_by):
                reverse = direction == "DESC"
                rows.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=reverse)
        
        # Apply OFFSET and LIMIT
        if offset:
            rows = rows[offset:]
        if limit:
            rows = rows[:limit]
        
        return SQLQueryResult(rows=rows, columns=columns, rows_affected=0)
    
    def _matches_conditions(self, row: Dict, conditions: List[Tuple]) -> bool:
        """Check if a row matches all conditions."""
        for col, op, val in conditions:
            row_val = row.get(col)
            
            if op == "=":
                if row_val != val:
                    return False
            elif op == "!=":
                if row_val == val:
                    return False
            elif op == ">":
                if row_val is None or row_val <= val:
                    return False
            elif op == ">=":
                if row_val is None or row_val < val:
                    return False
            elif op == "<":
                if row_val is None or row_val >= val:
                    return False
            elif op == "<=":
                if row_val is None or row_val > val:
                    return False
            elif op == "LIKE":
                if row_val is None:
                    return False
                # Convert SQL LIKE to regex
                pattern = val.replace("%", ".*").replace("_", ".")
                if not re.match(f"^{pattern}$", str(row_val), re.IGNORECASE):
                    return False
            elif op == "NOT_LIKE":
                if row_val is None:
                    return True
                pattern = val.replace("%", ".*").replace("_", ".")
                if re.match(f"^{pattern}$", str(row_val), re.IGNORECASE):
                    return False
        
        return True
    
    def _update(self, data: Dict) -> SQLQueryResult:
        """
        Update rows. Uses index lookup when WHERE clause has indexed equality condition.
        
        Index-accelerated path: O(k) where k = matching rows
        Fallback path: O(n) full table scan
        """
        table = data["table"]
        updates = data["updates"]
        conditions = data.get("where", [])
        
        schema = self._get_schema(table)
        if schema is None:
            raise ValueError(f"Table '{table}' does not exist")
        
        indexes = self._get_indexes(table)
        rows_affected = 0
        
        # Try index-accelerated path
        indexed_cond = self._find_indexed_equality_condition(table, conditions)
        
        if indexed_cond:
            # Index-accelerated UPDATE: lookup matching row IDs directly
            col, val = indexed_cond
            row_ids = self._lookup_by_index(table, col, val)
            
            for row_id in row_ids:
                key = self._row_key(table, row_id)
                value = self._db.get(key)
                if value is None:
                    continue
                
                old_row = json.loads(value.decode())
                
                # Apply all WHERE conditions (not just the indexed one)
                if not self._matches_conditions(old_row, conditions):
                    continue
                
                # Apply updates
                new_row = old_row.copy()
                for ucol, uval in updates.items():
                    new_row[ucol] = uval
                
                # Update indexes for changed columns
                for idx_name, idx_col in indexes.items():
                    if idx_col in updates:
                        self._update_index(table, idx_name, idx_col, old_row, new_row, row_id)
                
                # Save updated row
                self._db.put(key, json.dumps(new_row).encode())
                rows_affected += 1
        else:
            # Fallback: full table scan
            prefix = self._row_prefix(table)
            
            for key, value in self._db.scan_prefix(prefix):
                old_row = json.loads(value.decode())
                
                # Apply WHERE conditions
                if self._matches_conditions(old_row, conditions):
                    # Apply updates
                    new_row = old_row.copy()
                    for ucol, uval in updates.items():
                        new_row[ucol] = uval
                    
                    row_id = old_row.get("_id", "")
                    
                    # Update indexes for changed columns
                    for idx_name, idx_col in indexes.items():
                        if idx_col in updates:
                            self._update_index(table, idx_name, idx_col, old_row, new_row, row_id)
                    
                    # Save updated row
                    self._db.put(key, json.dumps(new_row).encode())
                    rows_affected += 1
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=rows_affected)
    
    def _delete(self, data: Dict) -> SQLQueryResult:
        """
        Delete rows. Uses index lookup when WHERE clause has indexed equality condition.
        
        Index-accelerated path: O(k) where k = matching rows
        Fallback path: O(n) full table scan
        """
        table = data["table"]
        conditions = data.get("where", [])
        
        schema = self._get_schema(table)
        if schema is None:
            raise ValueError(f"Table '{table}' does not exist")
        
        indexes = self._get_indexes(table)
        rows_affected = 0
        
        # Try index-accelerated path
        indexed_cond = self._find_indexed_equality_condition(table, conditions)
        
        if indexed_cond:
            # Index-accelerated DELETE: lookup matching row IDs directly
            col, val = indexed_cond
            row_ids = self._lookup_by_index(table, col, val)
            
            rows_to_delete = []  # (key, row) pairs
            
            for row_id in row_ids:
                key = self._row_key(table, row_id)
                value = self._db.get(key)
                if value is None:
                    continue
                
                row = json.loads(value.decode())
                
                # Apply all WHERE conditions (not just the indexed one)
                if self._matches_conditions(row, conditions):
                    rows_to_delete.append((key, row, row_id))
            
            # Delete rows and update indexes
            for key, row, row_id in rows_to_delete:
                # Remove from all indexes
                for idx_name, idx_col in indexes.items():
                    self._update_index(table, idx_name, idx_col, row, None, row_id)
                
                self._db.delete(key)
                rows_affected += 1
        else:
            # Fallback: full table scan
            prefix = self._row_prefix(table)
            rows_to_delete = []
            
            # Collect keys to delete (don't modify while iterating)
            for key, value in self._db.scan_prefix(prefix):
                row = json.loads(value.decode())
                
                # Apply WHERE conditions
                if self._matches_conditions(row, conditions):
                    row_id = row.get("_id", "")
                    rows_to_delete.append((key, row, row_id))
            
            # Delete collected rows and update indexes
            for key, row, row_id in rows_to_delete:
                # Remove from all indexes
                for idx_name, idx_col in indexes.items():
                    self._update_index(table, idx_name, idx_col, row, None, row_id)
                
                self._db.delete(key)
                rows_affected += 1
        
        return SQLQueryResult(rows=[], columns=[], rows_affected=rows_affected)
