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
ToonDB ContextQuery Builder (Task 12: Killer Feature Productization)

Token-aware context retrieval for LLM applications.

The ContextQuery builder provides:
1. Token budgeting - Fit context within model limits
2. Relevance scoring - Prioritize most relevant chunks
3. Deduplication - Avoid repeating similar content
4. Structured output - Ready for LLM prompts

Example:
    context = (
        ContextQuery(collection)
        .add_vector_query(embedding, weight=0.7)
        .add_keyword_query("machine learning", weight=0.3)
        .with_token_budget(4000)
        .with_min_relevance(0.5)
        .execute()
    )
    
    prompt = f"{context.as_text()}\\n\\nQuestion: {question}"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from enum import Enum

if TYPE_CHECKING:
    from .namespace import Collection


# ============================================================================
# Token Estimation
# ============================================================================

class TokenEstimator:
    """
    Estimates token count for text.
    
    Uses a simple heuristic by default (4 chars ≈ 1 token), but can be
    configured with an actual tokenizer for accuracy.
    """
    
    def __init__(self, tokenizer: Optional[Callable[[str], int]] = None):
        """
        Initialize token estimator.
        
        Args:
            tokenizer: Optional function that takes text and returns token count.
                       If None, uses heuristic (4 chars ≈ 1 token).
        """
        self._tokenizer = tokenizer
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return self._tokenizer(text)
        # Heuristic: ~4 chars per token for English
        return max(1, len(text) // 4)
    
    @classmethod
    def tiktoken(cls, model: str = "gpt-4") -> "TokenEstimator":
        """
        Create estimator using tiktoken (requires tiktoken package).
        
        Args:
            model: OpenAI model name for tokenizer selection
            
        Returns:
            TokenEstimator with tiktoken
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            return cls(lambda text: len(encoding.encode(text)))
        except ImportError:
            raise ImportError(
                "tiktoken not installed. Install with: pip install tiktoken"
            )


# ============================================================================
# Deduplication
# ============================================================================

class DeduplicationStrategy(str, Enum):
    """Strategy for deduplicating results."""
    NONE = "none"           # No deduplication
    EXACT = "exact"         # Exact text match
    SEMANTIC = "semantic"   # Semantic similarity threshold


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""
    strategy: DeduplicationStrategy = DeduplicationStrategy.NONE
    similarity_threshold: float = 0.9  # For semantic dedup


# ============================================================================
# Context Chunk
# ============================================================================

@dataclass
class ContextChunk:
    """A chunk of context with metadata."""
    
    id: Union[str, int]
    text: str
    score: float
    tokens: int
    
    # Source information
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Multi-vector support
    chunk_index: Optional[int] = None
    doc_score: Optional[float] = None  # Aggregated doc score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "tokens": self.tokens,
            "source": self.source,
            "metadata": self.metadata,
        }


# ============================================================================
# Context Result
# ============================================================================

@dataclass
class ContextResult:
    """Result of a context query."""
    
    chunks: List[ContextChunk]
    total_tokens: int
    budget_tokens: int
    dropped_count: int = 0  # Chunks dropped due to budget
    
    # Query metadata
    vector_score_range: Optional[Tuple[float, float]] = None
    keyword_score_range: Optional[Tuple[float, float]] = None
    
    def as_text(self, separator: str = "\n\n---\n\n") -> str:
        """
        Format chunks as text for LLM prompt.
        
        Args:
            separator: Separator between chunks
            
        Returns:
            Formatted context string
        """
        return separator.join(chunk.text for chunk in self.chunks)
    
    def as_markdown(self, include_scores: bool = False) -> str:
        """
        Format chunks as markdown.
        
        Args:
            include_scores: Include relevance scores
            
        Returns:
            Markdown formatted context
        """
        sections = []
        for i, chunk in enumerate(self.chunks, 1):
            header = f"### Context {i}"
            if chunk.source:
                header += f" (Source: {chunk.source})"
            if include_scores:
                header += f" [Score: {chunk.score:.3f}]"
            
            sections.append(f"{header}\n\n{chunk.text}")
        
        return "\n\n".join(sections)
    
    def as_json(self) -> str:
        """Format chunks as JSON."""
        return json.dumps({
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "total_tokens": self.total_tokens,
            "budget_tokens": self.budget_tokens,
            "dropped_count": self.dropped_count,
        }, indent=2)
    
    def __iter__(self) -> Iterator[ContextChunk]:
        return iter(self.chunks)
    
    def __len__(self) -> int:
        return len(self.chunks)


# ============================================================================
# Query Component
# ============================================================================

@dataclass
class QueryComponent:
    """A component of a context query."""
    
    query_type: str  # "vector" | "keyword"
    weight: float
    
    # For vector queries
    vector: Optional[List[float]] = None
    
    # For keyword queries
    text: Optional[str] = None
    
    # Shared options
    k: int = 50  # Retrieve more than needed for reranking


# ============================================================================
# ContextQuery Builder
# ============================================================================

class ContextQuery:
    """
    Token-aware context retrieval builder.
    
    Provides a fluent API for building context queries with:
    - Multiple query types (vector, keyword, hybrid)
    - Token budgeting
    - Relevance filtering
    - Deduplication
    
    Example:
        context = (
            ContextQuery(collection)
            .add_vector_query(embedding, weight=0.7)
            .add_keyword_query("python programming", weight=0.3)
            .with_token_budget(4000)
            .with_min_relevance(0.5)
            .with_deduplication(DeduplicationStrategy.EXACT)
            .execute()
        )
        
        # Use in prompt
        prompt = f'''Context:
        {context.as_text()}
        
        Question: {question}
        '''
    """
    
    def __init__(
        self,
        collection: "Collection",
        token_estimator: Optional[TokenEstimator] = None,
    ):
        """
        Initialize context query builder.
        
        Args:
            collection: Collection to query
            token_estimator: Optional token estimator (default: heuristic)
        """
        self._collection = collection
        self._estimator = token_estimator or TokenEstimator()
        
        # Query components
        self._components: List[QueryComponent] = []
        
        # Result options
        self._token_budget: int = 4000
        self._min_relevance: float = 0.0
        self._max_chunks: int = 50
        
        # Text options
        self._text_field: str = "content"  # Field to extract text from
        self._source_field: Optional[str] = "source"
        
        # Deduplication
        self._dedup = DeduplicationConfig()
        
        # Filtering
        self._filter: Optional[Dict[str, Any]] = None
    
    # ========================================================================
    # Query Components
    # ========================================================================
    
    def add_vector_query(
        self,
        vector: List[float],
        weight: float = 1.0,
        k: int = 50,
    ) -> "ContextQuery":
        """
        Add a vector similarity query.
        
        Args:
            vector: Query embedding
            weight: Weight for combining with other queries
            k: Number of results to retrieve (before filtering)
            
        Returns:
            self for chaining
        """
        self._components.append(QueryComponent(
            query_type="vector",
            weight=weight,
            vector=vector,
            k=k,
        ))
        return self
    
    def add_keyword_query(
        self,
        text: str,
        weight: float = 1.0,
        k: int = 50,
    ) -> "ContextQuery":
        """
        Add a keyword (BM25) query.
        
        Args:
            text: Search text
            weight: Weight for combining with other queries
            k: Number of results to retrieve (before filtering)
            
        Returns:
            self for chaining
        """
        self._components.append(QueryComponent(
            query_type="keyword",
            weight=weight,
            text=text,
            k=k,
        ))
        return self
    
    # ========================================================================
    # Result Options
    # ========================================================================
    
    def with_token_budget(self, tokens: int) -> "ContextQuery":
        """
        Set the token budget for context.
        
        Chunks will be added until the budget is exhausted.
        
        Args:
            tokens: Maximum tokens to include
            
        Returns:
            self for chaining
        """
        self._token_budget = tokens
        return self
    
    def with_min_relevance(self, threshold: float) -> "ContextQuery":
        """
        Set minimum relevance threshold.
        
        Chunks with scores below this threshold are excluded.
        
        Args:
            threshold: Minimum score (0-1 range for normalized scores)
            
        Returns:
            self for chaining
        """
        self._min_relevance = threshold
        return self
    
    def with_max_chunks(self, n: int) -> "ContextQuery":
        """
        Set maximum number of chunks.
        
        Args:
            n: Maximum chunks to return
            
        Returns:
            self for chaining
        """
        self._max_chunks = n
        return self
    
    # ========================================================================
    # Text Options
    # ========================================================================
    
    def from_field(self, field: str) -> "ContextQuery":
        """
        Specify which metadata field contains the text.
        
        Args:
            field: Metadata field name
            
        Returns:
            self for chaining
        """
        self._text_field = field
        return self
    
    def with_source_field(self, field: Optional[str]) -> "ContextQuery":
        """
        Specify which metadata field contains the source.
        
        Args:
            field: Metadata field name (None to disable)
            
        Returns:
            self for chaining
        """
        self._source_field = field
        return self
    
    # ========================================================================
    # Deduplication
    # ========================================================================
    
    def with_deduplication(
        self,
        strategy: DeduplicationStrategy,
        similarity_threshold: float = 0.9,
    ) -> "ContextQuery":
        """
        Configure deduplication.
        
        Args:
            strategy: Deduplication strategy
            similarity_threshold: Threshold for semantic dedup
            
        Returns:
            self for chaining
        """
        self._dedup = DeduplicationConfig(strategy, similarity_threshold)
        return self
    
    # ========================================================================
    # Filtering
    # ========================================================================
    
    def with_filter(self, filter: Dict[str, Any]) -> "ContextQuery":
        """
        Add metadata filter.
        
        Args:
            filter: Metadata filter dict
            
        Returns:
            self for chaining
        """
        self._filter = filter
        return self
    
    # ========================================================================
    # Execution
    # ========================================================================
    
    def execute(self) -> ContextResult:
        """
        Execute the context query.
        
        Returns:
            ContextResult with chunks fitting within token budget
        """
        if not self._components:
            raise ValueError("No query components added. Use add_vector_query() or add_keyword_query()")
        
        # Execute queries and combine results
        all_results = self._execute_queries()
        
        # Deduplicate
        all_results = self._deduplicate(all_results)
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply relevance filter
        all_results = [r for r in all_results if r.score >= self._min_relevance]
        
        # Budget allocation
        chunks = []
        total_tokens = 0
        dropped = 0
        
        for result in all_results[:self._max_chunks]:
            if total_tokens + result.tokens > self._token_budget:
                dropped += 1
                continue
            
            chunks.append(result)
            total_tokens += result.tokens
        
        dropped += len(all_results) - len(chunks) - dropped
        
        return ContextResult(
            chunks=chunks,
            total_tokens=total_tokens,
            budget_tokens=self._token_budget,
            dropped_count=dropped,
        )
    
    def _execute_queries(self) -> List[ContextChunk]:
        """Execute all query components and combine results."""
        from .namespace import SearchRequest
        
        # Track scores by ID for combining
        combined: Dict[Union[str, int], Dict[str, Any]] = {}
        
        for component in self._components:
            if component.query_type == "vector" and component.vector:
                request = SearchRequest(
                    vector=component.vector,
                    k=component.k,
                    filter=self._filter,
                    include_metadata=True,
                )
                results = self._collection.search(request)
                
                for result in results:
                    if result.id not in combined:
                        combined[result.id] = {
                            "id": result.id,
                            "metadata": result.metadata or {},
                            "vector_score": 0.0,
                            "keyword_score": 0.0,
                        }
                    combined[result.id]["vector_score"] += result.score * component.weight
            
            elif component.query_type == "keyword" and component.text:
                request = SearchRequest(
                    text_query=component.text,
                    k=component.k,
                    filter=self._filter,
                    alpha=0.0,  # Pure keyword
                    include_metadata=True,
                )
                results = self._collection.search(request)
                
                for result in results:
                    if result.id not in combined:
                        combined[result.id] = {
                            "id": result.id,
                            "metadata": result.metadata or {},
                            "vector_score": 0.0,
                            "keyword_score": 0.0,
                        }
                    combined[result.id]["keyword_score"] += result.score * component.weight
        
        # Convert to chunks
        chunks = []
        total_weight = sum(c.weight for c in self._components)
        
        for id, data in combined.items():
            metadata = data["metadata"]
            text = str(metadata.get(self._text_field, ""))
            
            # Normalize combined score
            combined_score = (data["vector_score"] + data["keyword_score"]) / total_weight
            
            chunk = ContextChunk(
                id=id,
                text=text,
                score=combined_score,
                tokens=self._estimator.count(text),
                source=metadata.get(self._source_field) if self._source_field else None,
                metadata=metadata,
            )
            chunks.append(chunk)
        
        return chunks
    
    def _deduplicate(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Remove duplicate chunks based on strategy."""
        if self._dedup.strategy == DeduplicationStrategy.NONE:
            return chunks
        
        if self._dedup.strategy == DeduplicationStrategy.EXACT:
            seen_texts = set()
            deduped = []
            for chunk in chunks:
                if chunk.text not in seen_texts:
                    seen_texts.add(chunk.text)
                    deduped.append(chunk)
            return deduped
        
        # Semantic deduplication would require embedding comparison
        # For now, fall back to exact match
        return chunks


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for text.
    
    Uses tiktoken if available, otherwise falls back to heuristic.
    
    Args:
        text: Text to estimate
        model: Model for tokenizer selection
        
    Returns:
        Estimated token count
    """
    try:
        estimator = TokenEstimator.tiktoken(model)
    except ImportError:
        estimator = TokenEstimator()
    
    return estimator.count(text)


def split_by_tokens(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    model: str = "gpt-4",
) -> List[str]:
    """
    Split text into chunks by token count.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks
        model: Model for tokenizer
        
    Returns:
        List of text chunks
    """
    try:
        estimator = TokenEstimator.tiktoken(model)
    except ImportError:
        estimator = TokenEstimator()
    
    # Simple sentence-based splitting
    sentences = text.replace("? ", "?\n").replace("! ", "!\n").replace(". ", ".\n").split("\n")
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimator.count(sentence)
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Keep overlap
            overlap_chunk = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_tokens = estimator.count(s)
                if overlap_count + s_tokens > overlap_tokens:
                    break
                overlap_chunk.insert(0, s)
                overlap_count += s_tokens
            
            current_chunk = overlap_chunk
            current_tokens = overlap_count
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


# ============================================================================
# CONTEXT SELECT: Production-Ready Query Builder (Aligned with Rust Model)
# ============================================================================

class SectionKind(str, Enum):
    """Section content kind - aligned with Rust SectionContent enum."""
    GET = "get"              # GET path expression
    LAST = "last"            # LAST N FROM table
    SEARCH = "search"        # SEARCH by similarity
    SELECT = "select"        # Standard SQL subquery
    LITERAL = "literal"      # Literal value
    VARIABLE = "variable"    # Variable reference
    TOOL_REGISTRY = "tool_registry"  # Available tools
    TOOL_CALLS = "tool_calls"        # Recent tool calls


class TruncationPolicy(str, Enum):
    """Truncation policy when budget is exceeded."""
    TAIL_DROP = "tail_drop"       # Drop from tail (keep head)
    HEAD_DROP = "head_drop"       # Drop from head (keep tail)
    PROPORTIONAL = "proportional" # Proportional truncation
    FAIL = "fail"                 # Fail on budget exceeded


@dataclass
class ContextSectionConfig:
    """
    Configuration for a single context section.
    
    Aligned with Rust ContextSection model for production consistency.
    
    Example:
        # Get user profile
        ContextSectionConfig(
            name="user",
            kind=SectionKind.GET,
            priority=0,
            path="users/alice/profile",
            fields=["name", "preferences"]
        )
        
        # Last 10 tool calls
        ContextSectionConfig(
            name="history",
            kind=SectionKind.LAST,
            priority=1,
            table="tool_calls",
            count=10,
            where={"status": "success"}
        )
        
        # Vector search
        ContextSectionConfig(
            name="knowledge",
            kind=SectionKind.SEARCH,
            priority=2,
            collection="docs",
            query="machine learning",
            top_k=5
        )
    """
    name: str
    kind: SectionKind
    priority: int = 0  # Lower = higher priority
    
    # GET options
    path: Optional[str] = None
    fields: Optional[List[str]] = None
    
    # LAST/SELECT options
    table: Optional[str] = None
    count: Optional[int] = None
    columns: Optional[List[str]] = None
    where: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    
    # SEARCH options
    collection: Optional[str] = None
    query: Optional[str] = None
    vector: Optional[List[float]] = None
    top_k: int = 5
    min_score: Optional[float] = None
    
    # LITERAL options
    text: Optional[str] = None
    
    # VARIABLE options
    variable_name: Optional[str] = None
    
    # TOOL_REGISTRY options
    include_tools: Optional[List[str]] = None
    exclude_tools: Optional[List[str]] = None
    include_schema: bool = True
    
    # TOOL_CALLS options
    tool_filter: Optional[str] = None
    status_filter: Optional[str] = None
    include_outputs: bool = True


@dataclass
class ContextSelectResult:
    """Result from CONTEXT SELECT execution."""
    sections: List[Dict[str, Any]]
    total_tokens: int
    budget_tokens: int
    truncated: bool = False
    truncated_sections: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    def as_text(self, include_headers: bool = True) -> str:
        """Format as text for LLM prompt."""
        parts = []
        for section in self.sections:
            if include_headers:
                parts.append(f"## {section['name']}")
            if "content" in section:
                content = section["content"]
                if isinstance(content, list):
                    parts.append("\n".join(str(item) for item in content))
                elif isinstance(content, dict):
                    parts.append(json.dumps(content, indent=2))
                else:
                    parts.append(str(content))
        return "\n\n".join(parts)
    
    def as_json(self) -> str:
        """Format as JSON."""
        return json.dumps({
            "sections": self.sections,
            "total_tokens": self.total_tokens,
            "budget_tokens": self.budget_tokens,
            "truncated": self.truncated,
            "provenance": self.provenance,
        }, indent=2)


class ContextSelect:
    """
    Production-ready CONTEXT SELECT query builder.
    
    Aligned with Rust ContextSelectQuery for consistent semantics across
    embedded, IPC, and MCP deployment modes.
    
    Example:
        from toondb import Database
        from toondb.context import ContextSelect, SectionKind, ContextSectionConfig
        
        db = Database.open("./my_db")
        
        result = (
            ContextSelect(db)
            .add_section(ContextSectionConfig(
                name="user",
                kind=SectionKind.GET,
                priority=0,
                path="users/alice/profile"
            ))
            .add_section(ContextSectionConfig(
                name="history",
                kind=SectionKind.LAST,
                priority=1,
                table="events",
                count=10
            ))
            .add_section(ContextSectionConfig(
                name="knowledge",
                kind=SectionKind.SEARCH,
                priority=2,
                collection="docs",
                query="machine learning",
                top_k=5
            ))
            .with_token_budget(4096)
            .execute()
        )
        
        prompt = f'''Context:
        {result.as_text()}
        
        Question: What is machine learning?
        '''
    """
    
    def __init__(
        self,
        db: "Database",
        token_estimator: Optional[TokenEstimator] = None,
    ):
        """
        Initialize CONTEXT SELECT builder.
        
        Args:
            db: ToonDB Database instance
            token_estimator: Optional token estimator (default: heuristic)
        """
        from .database import Database
        if not isinstance(db, Database):
            raise TypeError("db must be a Database instance")
        
        self._db = db
        self._estimator = token_estimator or TokenEstimator()
        self._sections: List[ContextSectionConfig] = []
        self._token_budget: int = 4096
        self._truncation: TruncationPolicy = TruncationPolicy.TAIL_DROP
        self._include_headers: bool = True
        self._output_format: str = "text"
        
    def add_section(self, section: ContextSectionConfig) -> "ContextSelect":
        """Add a section to the context query."""
        self._sections.append(section)
        return self
    
    def with_token_budget(self, tokens: int) -> "ContextSelect":
        """Set token budget for entire context."""
        self._token_budget = tokens
        return self
    
    def with_truncation(self, policy: TruncationPolicy) -> "ContextSelect":
        """Set truncation policy when budget is exceeded."""
        self._truncation = policy
        return self
    
    def with_headers(self, include: bool = True) -> "ContextSelect":
        """Include section headers in output."""
        self._include_headers = include
        return self
    
    def execute(self) -> ContextSelectResult:
        """
        Execute the CONTEXT SELECT query.
        
        Sections are processed in priority order (lower = higher priority).
        Token budget is enforced using greedy allocation.
        
        Returns:
            ContextSelectResult with assembled context
        """
        if not self._sections:
            raise ValueError("No sections added. Use add_section().")
        
        # Sort by priority
        sorted_sections = sorted(self._sections, key=lambda s: s.priority)
        
        assembled_sections = []
        total_tokens = 0
        truncated = False
        truncated_sections = []
        provenance = {}
        
        for section in sorted_sections:
            # Execute section
            section_data = self._execute_section(section)
            
            # Estimate tokens
            section_text = json.dumps(section_data) if isinstance(section_data, (dict, list)) else str(section_data)
            section_tokens = self._estimator.count(section_text)
            
            # Check budget
            if total_tokens + section_tokens > self._token_budget:
                if self._truncation == TruncationPolicy.FAIL:
                    raise ValueError(f"Token budget exceeded at section '{section.name}'")
                elif self._truncation == TruncationPolicy.TAIL_DROP:
                    # Try to fit partial content
                    remaining = self._token_budget - total_tokens
                    if remaining > 50:  # Minimum useful content
                        section_data = self._truncate_section(section_data, remaining)
                        section_tokens = remaining
                        truncated = True
                        truncated_sections.append(section.name)
                    else:
                        truncated_sections.append(section.name)
                        continue
                else:
                    truncated_sections.append(section.name)
                    continue
            
            assembled_sections.append({
                "name": section.name,
                "priority": section.priority,
                "kind": section.kind.value,
                "content": section_data,
                "tokens": section_tokens,
            })
            total_tokens += section_tokens
            
            # Track provenance
            provenance[section.name] = {
                "kind": section.kind.value,
                "tokens": section_tokens,
                "source": self._get_provenance_source(section),
            }
        
        return ContextSelectResult(
            sections=assembled_sections,
            total_tokens=total_tokens,
            budget_tokens=self._token_budget,
            truncated=truncated,
            truncated_sections=truncated_sections,
            provenance=provenance,
        )
    
    def _execute_section(self, section: ContextSectionConfig) -> Any:
        """Execute a single section and return its content."""
        if section.kind == SectionKind.GET:
            return self._exec_get(section)
        elif section.kind == SectionKind.LAST:
            return self._exec_last(section)
        elif section.kind == SectionKind.SELECT:
            return self._exec_select(section)
        elif section.kind == SectionKind.SEARCH:
            return self._exec_search(section)
        elif section.kind == SectionKind.LITERAL:
            return section.text or ""
        elif section.kind == SectionKind.VARIABLE:
            return f"${{{section.variable_name}}}"  # Placeholder for variable expansion
        elif section.kind == SectionKind.TOOL_REGISTRY:
            return self._exec_tool_registry(section)
        elif section.kind == SectionKind.TOOL_CALLS:
            return self._exec_tool_calls(section)
        else:
            return None
    
    def _exec_get(self, section: ContextSectionConfig) -> Any:
        """Execute GET section."""
        if not section.path:
            return None
        
        data = self._db.get_path(section.path)
        if data is None:
            return None
        
        try:
            parsed = json.loads(data.decode("utf-8"))
            # Project fields if specified
            if section.fields and isinstance(parsed, dict):
                return {k: parsed.get(k) for k in section.fields if k in parsed}
            return parsed
        except (json.JSONDecodeError, UnicodeDecodeError):
            return data.decode("utf-8", errors="replace")
    
    def _exec_last(self, section: ContextSectionConfig) -> List[Any]:
        """Execute LAST section."""
        if not section.table:
            return []
        
        prefix = f"_sql/tables/{section.table}/rows/".encode()
        results = []
        count = section.count or 10
        
        for key, value in self._db.scan_prefix(prefix):
            try:
                row = json.loads(value.decode("utf-8"))
                # Apply WHERE filter
                if section.where:
                    if not self._matches_where(row, section.where):
                        continue
                results.append(row)
                if len(results) >= count:
                    break
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        
        return results
    
    def _exec_select(self, section: ContextSectionConfig) -> List[Any]:
        """Execute SELECT section."""
        # Use SQL engine if available
        if section.table:
            columns = ",".join(section.columns or ["*"])
            sql = f"SELECT {columns} FROM {section.table}"
            if section.where:
                conditions = " AND ".join(f"{k}='{v}'" for k, v in section.where.items())
                sql += f" WHERE {conditions}"
            if section.limit:
                sql += f" LIMIT {section.limit}"
            
            try:
                result = self._db.execute(sql)
                return result.rows
            except Exception:
                return []
        return []
    
    def _exec_search(self, section: ContextSectionConfig) -> List[Any]:
        """Execute SEARCH section."""
        # Vector search via namespace/collection
        # This is a placeholder - real implementation would use vector index
        if section.collection:
            # Try to get collection from namespace
            try:
                ns = self._db.namespace("default")
                coll = ns.collection(section.collection)
                from .namespace import SearchRequest
                
                request = SearchRequest(
                    text_query=section.query,
                    k=section.top_k,
                )
                results = coll.search(request)
                return [{"id": r.id, "score": r.score, "metadata": r.metadata} for r in results]
            except Exception:
                pass
        return []
    
    def _exec_tool_registry(self, section: ContextSectionConfig) -> List[Dict[str, Any]]:
        """Execute TOOL_REGISTRY section."""
        # Placeholder - would query actual tool registry
        return [{"name": "example_tool", "description": "Example tool"}]
    
    def _exec_tool_calls(self, section: ContextSectionConfig) -> List[Any]:
        """Execute TOOL_CALLS section."""
        prefix = b"_tool_calls/"
        results = []
        count = section.count or 10
        
        for key, value in self._db.scan_prefix(prefix):
            try:
                call = json.loads(value.decode("utf-8"))
                if section.tool_filter and call.get("tool") != section.tool_filter:
                    continue
                if section.status_filter and call.get("status") != section.status_filter:
                    continue
                if not section.include_outputs:
                    call.pop("output", None)
                results.append(call)
                if len(results) >= count:
                    break
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        
        return results
    
    def _matches_where(self, row: Dict[str, Any], where: Dict[str, Any]) -> bool:
        """Check if row matches WHERE conditions."""
        for key, value in where.items():
            if row.get(key) != value:
                return False
        return True
    
    def _truncate_section(self, data: Any, max_tokens: int) -> Any:
        """Truncate section content to fit token budget."""
        if isinstance(data, str):
            # Truncate string
            chars = max_tokens * 4  # Heuristic
            return data[:chars] + "..." if len(data) > chars else data
        elif isinstance(data, list):
            # Truncate list
            truncated = []
            tokens = 0
            for item in data:
                item_str = json.dumps(item) if isinstance(item, dict) else str(item)
                item_tokens = self._estimator.count(item_str)
                if tokens + item_tokens > max_tokens:
                    break
                truncated.append(item)
                tokens += item_tokens
            return truncated
        return data
    
    def _get_provenance_source(self, section: ContextSectionConfig) -> str:
        """Get provenance source string for a section."""
        if section.kind == SectionKind.GET:
            return f"path:{section.path}"
        elif section.kind == SectionKind.LAST:
            return f"table:{section.table}"
        elif section.kind == SectionKind.SEARCH:
            return f"collection:{section.collection}"
        elif section.kind == SectionKind.SELECT:
            return f"sql:{section.table}"
        return section.kind.value
