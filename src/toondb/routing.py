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
Tool Routing Primitive for Multi-Agent Scenarios.

Provides a first-class system for routing tool calls to agents based on:
- Agent capabilities
- Tool requirements
- Load balancing
- Agent availability

This enables multi-agent architectures where specialized agents handle
different tool domains (e.g., code, search, database, email).

Example:
--------
    from toondb import Database
    from toondb.routing import (
        ToolRouter,
        AgentRegistry,
        Tool,
        ToolCategory,
        RoutingStrategy,
    )
    
    db = Database.open("./agent_data")
    registry = AgentRegistry(db)
    router = ToolRouter(registry)
    
    # Register agents with capabilities
    registry.register_agent(
        agent_id="code_agent",
        capabilities=[ToolCategory.CODE, ToolCategory.GIT],
        endpoint="http://localhost:8001/invoke",
    )
    registry.register_agent(
        agent_id="search_agent",
        capabilities=[ToolCategory.SEARCH, ToolCategory.WEB],
        endpoint="http://localhost:8002/invoke",
    )
    
    # Register tools with categories
    router.register_tool(Tool(
        name="search_code",
        description="Search codebase",
        category=ToolCategory.CODE,
        schema={"type": "object", "properties": {"query": {"type": "string"}}},
    ))
    
    # Route a tool call to the best agent
    result = router.route(
        tool_name="search_code",
        args={"query": "authentication handler"},
        context={"session_id": "sess_001"},
    )
"""

import time
import json
import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from threading import Lock
import random


class ToolCategory(Enum):
    """Standard tool categories for routing."""
    CODE = "code"           # Code analysis, generation, editing
    SEARCH = "search"       # Search operations (semantic, keyword)
    DATABASE = "database"   # Database operations (CRUD, queries)
    WEB = "web"             # Web browsing, API calls
    FILE = "file"           # File system operations
    GIT = "git"             # Git/VCS operations
    SHELL = "shell"         # Shell command execution
    EMAIL = "email"         # Email operations
    CALENDAR = "calendar"   # Calendar/scheduling
    MEMORY = "memory"       # Agent memory operations
    VECTOR = "vector"       # Vector search and embeddings
    GRAPH = "graph"         # Graph operations
    CUSTOM = "custom"       # User-defined category


class RoutingStrategy(Enum):
    """How to select among multiple capable agents."""
    ROUND_ROBIN = "round_robin"      # Cycle through agents
    RANDOM = "random"                 # Random selection
    LEAST_LOADED = "least_loaded"     # Prefer less busy agents
    STICKY = "sticky"                 # Keep same agent for session
    PRIORITY = "priority"             # Use agent priority scores
    FASTEST = "fastest"               # Prefer historically fastest


class AgentStatus(Enum):
    """Agent availability status."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    DEGRADED = "degraded"


@dataclass
class Tool:
    """Definition of a routable tool."""
    name: str
    description: str
    category: ToolCategory
    schema: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: List[ToolCategory] = field(default_factory=list)
    timeout_seconds: float = 30.0
    retries: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """Definition of an agent that can handle tools."""
    agent_id: str
    capabilities: List[ToolCategory]
    endpoint: Optional[str] = None
    handler: Optional[Callable] = None  # Local function handler
    priority: int = 100  # Higher = preferred
    max_concurrent: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    status: AgentStatus = AgentStatus.AVAILABLE
    current_load: int = 0
    total_calls: int = 0
    total_latency_ms: float = 0.0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None


@dataclass
class RouteResult:
    """Result of a tool routing decision."""
    agent_id: str
    tool_name: str
    result: Any
    latency_ms: float
    success: bool
    error: Optional[str] = None
    retries_used: int = 0


@dataclass
class RoutingContext:
    """Context for routing decisions."""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    priority: int = 100
    timeout_override: Optional[float] = None
    preferred_agent: Optional[str] = None
    excluded_agents: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """
    Registry of agents and their capabilities.
    
    Persists agent registrations to ToonDB for durability across restarts.
    """
    
    PREFIX = "/_routing/agents/"
    
    def __init__(self, db):
        """
        Create an agent registry.
        
        Args:
            db: ToonDB Database instance
        """
        self._db = db
        self._agents: Dict[str, Agent] = {}
        self._lock = Lock()
        self._load_agents()
    
    def _load_agents(self):
        """Load agent registrations from database."""
        try:
            results = self._db.scan_prefix(self.PREFIX.encode())
            for key, value in results:
                try:
                    data = json.loads(value.decode())
                    agent = Agent(
                        agent_id=data["agent_id"],
                        capabilities=[ToolCategory(c) for c in data["capabilities"]],
                        endpoint=data.get("endpoint"),
                        priority=data.get("priority", 100),
                        max_concurrent=data.get("max_concurrent", 10),
                        metadata=data.get("metadata", {}),
                    )
                    self._agents[agent.agent_id] = agent
                except (json.JSONDecodeError, KeyError):
                    pass
        except Exception:
            pass  # No existing agents
    
    def register_agent(
        self,
        agent_id: str,
        capabilities: List[Union[ToolCategory, str]],
        endpoint: Optional[str] = None,
        handler: Optional[Callable] = None,
        priority: int = 100,
        max_concurrent: int = 10,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """
        Register an agent with capabilities.
        
        Args:
            agent_id: Unique agent identifier
            capabilities: List of tool categories this agent handles
            endpoint: HTTP endpoint for remote agents
            handler: Local function handler for in-process agents
            priority: Routing priority (higher = preferred)
            max_concurrent: Max concurrent tool calls
            metadata: Additional metadata
            
        Returns:
            The registered Agent
        """
        caps = [
            c if isinstance(c, ToolCategory) else ToolCategory(c)
            for c in capabilities
        ]
        
        agent = Agent(
            agent_id=agent_id,
            capabilities=caps,
            endpoint=endpoint,
            handler=handler,
            priority=priority,
            max_concurrent=max_concurrent,
            metadata=metadata or {},
        )
        
        with self._lock:
            self._agents[agent_id] = agent
            
            # Persist to database (skip handler as it's not serializable)
            data = {
                "agent_id": agent_id,
                "capabilities": [c.value for c in caps],
                "endpoint": endpoint,
                "priority": priority,
                "max_concurrent": max_concurrent,
                "metadata": metadata or {},
            }
            key = f"{self.PREFIX}{agent_id}".encode()
            self._db.put(key, json.dumps(data).encode())
        
        return agent
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent registration."""
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                key = f"{self.PREFIX}{agent_id}".encode()
                self._db.delete(key)
                return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[Agent]:
        """List all registered agents."""
        return list(self._agents.values())
    
    def find_capable_agents(
        self,
        required: List[ToolCategory],
        exclude: Optional[List[str]] = None,
    ) -> List[Agent]:
        """
        Find agents capable of handling the required categories.
        
        Args:
            required: Required tool categories
            exclude: Agent IDs to exclude
            
        Returns:
            List of capable agents
        """
        exclude_set = set(exclude or [])
        capable = []
        
        for agent in self._agents.values():
            if agent.agent_id in exclude_set:
                continue
            if agent.status == AgentStatus.OFFLINE:
                continue
            
            # Check if agent has all required capabilities
            agent_caps = set(agent.capabilities)
            if all(req in agent_caps for req in required):
                capable.append(agent)
        
        return capable
    
    def update_agent_status(self, agent_id: str, status: AgentStatus):
        """Update an agent's status."""
        if agent := self._agents.get(agent_id):
            agent.status = status
    
    def record_call(self, agent_id: str, latency_ms: float, success: bool):
        """Record a tool call result for an agent."""
        if agent := self._agents.get(agent_id):
            agent.total_calls += 1
            agent.total_latency_ms += latency_ms
            if success:
                agent.last_success = time.time()
            else:
                agent.last_failure = time.time()


class ToolRouter:
    """
    Routes tool calls to appropriate agents.
    
    Supports multiple routing strategies and automatic failover.
    """
    
    TOOLS_PREFIX = "/_routing/tools/"
    SESSIONS_PREFIX = "/_routing/sessions/"
    
    def __init__(
        self,
        registry: AgentRegistry,
        default_strategy: RoutingStrategy = RoutingStrategy.PRIORITY,
    ):
        """
        Create a tool router.
        
        Args:
            registry: Agent registry
            default_strategy: Default routing strategy
        """
        self._registry = registry
        self._db = registry._db
        self._default_strategy = default_strategy
        self._tools: Dict[str, Tool] = {}
        self._round_robin_idx: Dict[str, int] = defaultdict(int)
        self._session_affinity: Dict[str, str] = {}  # session_id -> agent_id
        self._lock = Lock()
        self._load_tools()
    
    def _load_tools(self):
        """Load tool registrations from database."""
        try:
            results = self._db.scan_prefix(self.TOOLS_PREFIX.encode())
            for key, value in results:
                try:
                    data = json.loads(value.decode())
                    tool = Tool(
                        name=data["name"],
                        description=data["description"],
                        category=ToolCategory(data["category"]),
                        schema=data.get("schema", {}),
                        required_capabilities=[
                            ToolCategory(c) for c in data.get("required_capabilities", [])
                        ],
                        timeout_seconds=data.get("timeout_seconds", 30.0),
                        retries=data.get("retries", 1),
                        metadata=data.get("metadata", {}),
                    )
                    self._tools[tool.name] = tool
                except (json.JSONDecodeError, KeyError):
                    pass
        except Exception:
            pass
    
    def register_tool(self, tool: Tool) -> Tool:
        """
        Register a tool for routing.
        
        Args:
            tool: Tool definition
            
        Returns:
            The registered tool
        """
        with self._lock:
            self._tools[tool.name] = tool
            
            # Persist to database
            data = {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "schema": tool.schema,
                "required_capabilities": [c.value for c in tool.required_capabilities],
                "timeout_seconds": tool.timeout_seconds,
                "retries": tool.retries,
                "metadata": tool.metadata,
            }
            key = f"{self.TOOLS_PREFIX}{tool.name}".encode()
            self._db.put(key, json.dumps(data).encode())
        
        return tool
    
    def unregister_tool(self, name: str) -> bool:
        """Remove a tool registration."""
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                key = f"{self.TOOLS_PREFIX}{name}".encode()
                self._db.delete(key)
                return True
        return False
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def route(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Union[Dict[str, Any], RoutingContext]] = None,
        strategy: Optional[RoutingStrategy] = None,
    ) -> RouteResult:
        """
        Route a tool call to the best agent.
        
        Args:
            tool_name: Name of the tool to call
            args: Tool arguments
            context: Routing context
            strategy: Override routing strategy
            
        Returns:
            RouteResult with the call outcome
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return RouteResult(
                agent_id="",
                tool_name=tool_name,
                result=None,
                latency_ms=0,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )
        
        # Normalize context
        if context is None:
            ctx = RoutingContext()
        elif isinstance(context, dict):
            ctx = RoutingContext(
                session_id=context.get("session_id"),
                user_id=context.get("user_id"),
                priority=context.get("priority", 100),
                timeout_override=context.get("timeout"),
                preferred_agent=context.get("preferred_agent"),
                excluded_agents=context.get("excluded_agents", []),
                custom=context,
            )
        else:
            ctx = context
        
        # Determine required capabilities
        required = tool.required_capabilities or [tool.category]
        
        # Find capable agents
        capable = self._registry.find_capable_agents(required, ctx.excluded_agents)
        if not capable:
            return RouteResult(
                agent_id="",
                tool_name=tool_name,
                result=None,
                latency_ms=0,
                success=False,
                error=f"No capable agents for tool '{tool_name}' (requires: {[c.value for c in required]})",
            )
        
        # Select agent using strategy
        use_strategy = strategy or self._default_strategy
        agent = self._select_agent(capable, use_strategy, ctx)
        
        # Execute with retries
        timeout = ctx.timeout_override or tool.timeout_seconds
        retries = tool.retries
        last_error = None
        
        for attempt in range(retries + 1):
            start_time = time.time()
            try:
                result = self._invoke_agent(agent, tool, args, timeout)
                latency_ms = (time.time() - start_time) * 1000
                
                # Record success
                self._registry.record_call(agent.agent_id, latency_ms, True)
                
                # Update session affinity
                if ctx.session_id:
                    self._session_affinity[ctx.session_id] = agent.agent_id
                
                return RouteResult(
                    agent_id=agent.agent_id,
                    tool_name=tool_name,
                    result=result,
                    latency_ms=latency_ms,
                    success=True,
                    retries_used=attempt,
                )
            
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                self._registry.record_call(agent.agent_id, latency_ms, False)
                last_error = str(e)
                
                # Try next capable agent on failure
                capable = [a for a in capable if a.agent_id != agent.agent_id]
                if capable:
                    agent = self._select_agent(capable, use_strategy, ctx)
        
        return RouteResult(
            agent_id=agent.agent_id if agent else "",
            tool_name=tool_name,
            result=None,
            latency_ms=0,
            success=False,
            error=last_error or "All retries exhausted",
            retries_used=retries,
        )
    
    def _select_agent(
        self,
        capable: List[Agent],
        strategy: RoutingStrategy,
        ctx: RoutingContext,
    ) -> Agent:
        """Select an agent using the specified strategy."""
        if not capable:
            raise ValueError("No capable agents")
        
        # Preferred agent override
        if ctx.preferred_agent:
            for agent in capable:
                if agent.agent_id == ctx.preferred_agent:
                    return agent
        
        # Session affinity (sticky routing)
        if strategy == RoutingStrategy.STICKY and ctx.session_id:
            if prev_agent := self._session_affinity.get(ctx.session_id):
                for agent in capable:
                    if agent.agent_id == prev_agent:
                        return agent
        
        if strategy == RoutingStrategy.ROUND_ROBIN:
            with self._lock:
                key = ",".join(sorted(a.agent_id for a in capable))
                idx = self._round_robin_idx[key] % len(capable)
                self._round_robin_idx[key] = idx + 1
                return capable[idx]
        
        elif strategy == RoutingStrategy.RANDOM:
            return random.choice(capable)
        
        elif strategy == RoutingStrategy.LEAST_LOADED:
            return min(capable, key=lambda a: a.current_load)
        
        elif strategy == RoutingStrategy.PRIORITY:
            # Sort by priority descending, then by load ascending
            return max(capable, key=lambda a: (a.priority, -a.current_load))
        
        elif strategy == RoutingStrategy.FASTEST:
            # Sort by average latency
            def avg_latency(a: Agent) -> float:
                if a.total_calls == 0:
                    return float("inf")
                return a.total_latency_ms / a.total_calls
            return min(capable, key=avg_latency)
        
        # Default to priority
        return max(capable, key=lambda a: a.priority)
    
    def _invoke_agent(
        self,
        agent: Agent,
        tool: Tool,
        args: Dict[str, Any],
        timeout: float,
    ) -> Any:
        """Invoke a tool on an agent."""
        agent.current_load += 1
        try:
            if agent.handler:
                # Local function handler
                return agent.handler(tool.name, args)
            
            elif agent.endpoint:
                # Remote HTTP invocation
                import urllib.request
                import urllib.error
                
                request_data = json.dumps({
                    "tool": tool.name,
                    "args": args,
                    "metadata": tool.metadata,
                }).encode("utf-8")
                
                req = urllib.request.Request(
                    agent.endpoint,
                    data=request_data,
                    headers={"Content-Type": "application/json"},
                )
                
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    response_data = resp.read().decode("utf-8")
                    return json.loads(response_data)
            
            else:
                raise ValueError(f"Agent {agent.agent_id} has no handler or endpoint")
        
        finally:
            agent.current_load -= 1


class ToolDispatcher:
    """
    High-level dispatcher for multi-agent tool orchestration.
    
    Provides a simple interface for agents to register and invoke tools.
    """
    
    def __init__(self, db):
        """
        Create a tool dispatcher.
        
        Args:
            db: ToonDB Database instance
        """
        self._db = db
        self._registry = AgentRegistry(db)
        self._router = ToolRouter(self._registry)
    
    @property
    def registry(self) -> AgentRegistry:
        """Get the agent registry."""
        return self._registry
    
    @property
    def router(self) -> ToolRouter:
        """Get the tool router."""
        return self._router
    
    def register_local_agent(
        self,
        agent_id: str,
        capabilities: List[Union[ToolCategory, str]],
        handler: Callable[[str, Dict[str, Any]], Any],
        priority: int = 100,
    ) -> Agent:
        """
        Register a local (in-process) agent.
        
        Args:
            agent_id: Unique agent identifier
            capabilities: Tool categories this agent handles
            handler: Function(tool_name, args) -> result
            priority: Routing priority
            
        Returns:
            The registered agent
        """
        return self._registry.register_agent(
            agent_id=agent_id,
            capabilities=capabilities,
            handler=handler,
            priority=priority,
        )
    
    def register_remote_agent(
        self,
        agent_id: str,
        capabilities: List[Union[ToolCategory, str]],
        endpoint: str,
        priority: int = 100,
    ) -> Agent:
        """
        Register a remote (HTTP) agent.
        
        Args:
            agent_id: Unique agent identifier
            capabilities: Tool categories this agent handles
            endpoint: HTTP endpoint URL
            priority: Routing priority
            
        Returns:
            The registered agent
        """
        return self._registry.register_agent(
            agent_id=agent_id,
            capabilities=capabilities,
            endpoint=endpoint,
            priority=priority,
        )
    
    def register_tool(
        self,
        name: str,
        description: str,
        category: Union[ToolCategory, str],
        schema: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Tool:
        """
        Register a tool for routing.
        
        Args:
            name: Tool name
            description: Tool description
            category: Tool category
            schema: JSON schema for arguments
            timeout: Call timeout in seconds
            
        Returns:
            The registered tool
        """
        cat = category if isinstance(category, ToolCategory) else ToolCategory(category)
        tool = Tool(
            name=name,
            description=description,
            category=cat,
            schema=schema or {},
            timeout_seconds=timeout,
        )
        return self._router.register_tool(tool)
    
    def invoke(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        strategy: Optional[RoutingStrategy] = None,
    ) -> RouteResult:
        """
        Invoke a tool with automatic routing.
        
        Args:
            tool_name: Name of the tool to call
            args: Tool arguments
            session_id: Optional session for sticky routing
            strategy: Override routing strategy
            
        Returns:
            RouteResult with the call outcome
        """
        ctx = RoutingContext(session_id=session_id)
        return self._router.route(tool_name, args or {}, ctx, strategy)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents with their status."""
        agents = self._registry.list_agents()
        return [
            {
                "agent_id": a.agent_id,
                "capabilities": [c.value for c in a.capabilities],
                "status": a.status.value,
                "priority": a.priority,
                "current_load": a.current_load,
                "total_calls": a.total_calls,
                "avg_latency_ms": (a.total_latency_ms / a.total_calls) if a.total_calls > 0 else None,
                "has_endpoint": a.endpoint is not None,
                "has_handler": a.handler is not None,
            }
            for a in agents
        ]
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        tools = self._router.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
                "schema": t.schema,
                "timeout_seconds": t.timeout_seconds,
                "retries": t.retries,
            }
            for t in tools
        ]
