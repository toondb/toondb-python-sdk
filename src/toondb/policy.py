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
Policy & Safety Hooks for Agent Operations.

Provides a trigger system for enforcing policies on agent actions:

- Pre-write validation (block dangerous operations)
- Post-read filtering (redact sensitive data)
- Rate limiting (prevent runaway agents)
- Audit logging (track all operations)

This is designed for AI agent safety and compliance scenarios where you need
to enforce guardrails on what agents can read/write.

Example:
--------
    from toondb import Database
    from toondb.policy import PolicyEngine, Policy, PolicyAction, PolicyTrigger
    
    db = Database.open("./agent_data")
    policy = PolicyEngine(db)
    
    # Block writes to system keys
    @policy.before_write("system/*")
    def block_system_writes(key, value, context):
        if context.get("agent_id"):
            return PolicyAction.DENY
        return PolicyAction.ALLOW
    
    # Redact sensitive data on read
    @policy.after_read("users/*/email")
    def redact_emails(key, value, context):
        if context.get("agent_id") and not context.get("has_pii_access"):
            return "[REDACTED]".encode()
        return value
    
    # Rate limit writes per agent
    policy.add_rate_limit("writes", max_per_minute=100, scope="agent_id")
    
    # Use policy-wrapped operations
    policy.put(b"users/alice/name", b"Alice", context={"agent_id": "agent_001"})
    value = policy.get(b"users/alice/email", context={"agent_id": "agent_001"})
"""

import time
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
from threading import Lock


class PolicyAction(Enum):
    """Action to take when a policy is triggered."""
    ALLOW = "allow"       # Allow the operation
    DENY = "deny"         # Block the operation
    MODIFY = "modify"     # Allow with modifications
    LOG = "log"           # Allow but log the operation
    RATE_LIMIT = "rate_limit"  # Apply rate limiting


class PolicyTrigger(Enum):
    """When the policy is triggered."""
    BEFORE_READ = "before_read"
    AFTER_READ = "after_read"
    BEFORE_WRITE = "before_write"
    AFTER_WRITE = "after_write"
    BEFORE_DELETE = "before_delete"
    AFTER_DELETE = "after_delete"


@dataclass
class PolicyResult:
    """Result of a policy evaluation."""
    action: PolicyAction
    modified_value: Optional[bytes] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyContext:
    """Context passed to policy handlers."""
    operation: str  # "read", "write", "delete"
    key: bytes
    value: Optional[bytes] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        if key == "agent_id":
            return self.agent_id
        if key == "session_id":
            return self.session_id
        return self.custom.get(key, default)


class PolicyHandler(ABC):
    """Abstract base class for policy handlers."""
    
    @abstractmethod
    def evaluate(self, context: PolicyContext) -> PolicyResult:
        """Evaluate the policy for the given context."""
        pass


class PatternPolicy(PolicyHandler):
    """Policy that applies to keys matching a pattern."""
    
    def __init__(
        self,
        pattern: str,
        trigger: PolicyTrigger,
        handler: Callable[[bytes, Optional[bytes], Dict[str, Any]], Union[PolicyAction, Tuple[PolicyAction, bytes]]],
    ):
        """
        Create a pattern-based policy.
        
        Args:
            pattern: Key pattern (glob-style: users/*/email, system/*)
            trigger: When to trigger
            handler: Function(key, value, context) -> PolicyAction or (PolicyAction, modified_value)
        """
        self.pattern = pattern
        self.trigger = trigger
        self.handler = handler
        self._regex = self._pattern_to_regex(pattern)
    
    def _pattern_to_regex(self, pattern: str) -> Pattern:
        """Convert glob pattern to regex."""
        regex = pattern.replace(".", r"\.").replace("*", r"[^/]*").replace("**", r".*")
        return re.compile(f"^{regex}$")
    
    def matches(self, key: bytes) -> bool:
        """Check if key matches the pattern."""
        try:
            key_str = key.decode("utf-8")
            return bool(self._regex.match(key_str))
        except UnicodeDecodeError:
            return False
    
    def evaluate(self, context: PolicyContext) -> PolicyResult:
        """Evaluate the policy."""
        if not self.matches(context.key):
            return PolicyResult(action=PolicyAction.ALLOW)
        
        result = self.handler(context.key, context.value, context.custom)
        
        if isinstance(result, PolicyAction):
            return PolicyResult(action=result)
        elif isinstance(result, tuple):
            action, value = result
            return PolicyResult(action=action, modified_value=value)
        elif isinstance(result, bytes):
            return PolicyResult(action=PolicyAction.MODIFY, modified_value=result)
        else:
            return PolicyResult(action=PolicyAction.ALLOW)


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.tokens = max_per_minute
        self.last_refill = time.time()
        self._lock = Lock()
    
    def try_acquire(self) -> bool:
        """Try to acquire a token. Returns True if allowed."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens based on elapsed time
            refill = int(elapsed * self.max_per_minute / 60)
            if refill > 0:
                self.tokens = min(self.max_per_minute, self.tokens + refill)
                self.last_refill = now
            
            if self.tokens > 0:
                self.tokens -= 1
                return True
            return False
    
    def remaining(self) -> int:
        """Get remaining tokens."""
        return self.tokens


class PolicyEngine:
    """
    Policy engine for enforcing safety rules on database operations.
    
    Wraps a ToonDB Database instance and applies policies to all operations.
    """
    
    def __init__(self, db):
        """
        Create a policy engine.
        
        Args:
            db: ToonDB Database instance
        """
        self._db = db
        self._policies: Dict[PolicyTrigger, List[PolicyHandler]] = defaultdict(list)
        self._rate_limiters: Dict[str, Dict[str, RateLimiter]] = defaultdict(dict)
        self._rate_limit_configs: List[Dict] = []
        self._audit_log: List[Dict] = []
        self._audit_enabled = False
        self._max_audit_entries = 10000
        self._lock = Lock()
    
    # =========================================================================
    # Decorator API
    # =========================================================================
    
    def before_write(self, pattern: str):
        """
        Decorator for pre-write policies.
        
        Example:
            @policy.before_write("system/*")
            def block_system_writes(key, value, context):
                return PolicyAction.DENY
        """
        def decorator(handler):
            policy = PatternPolicy(pattern, PolicyTrigger.BEFORE_WRITE, handler)
            self._policies[PolicyTrigger.BEFORE_WRITE].append(policy)
            return handler
        return decorator
    
    def after_write(self, pattern: str):
        """Decorator for post-write policies (e.g., audit logging)."""
        def decorator(handler):
            policy = PatternPolicy(pattern, PolicyTrigger.AFTER_WRITE, handler)
            self._policies[PolicyTrigger.AFTER_WRITE].append(policy)
            return handler
        return decorator
    
    def before_read(self, pattern: str):
        """Decorator for pre-read policies (e.g., access control)."""
        def decorator(handler):
            policy = PatternPolicy(pattern, PolicyTrigger.BEFORE_READ, handler)
            self._policies[PolicyTrigger.BEFORE_READ].append(policy)
            return handler
        return decorator
    
    def after_read(self, pattern: str):
        """
        Decorator for post-read policies (e.g., redaction).
        
        Example:
            @policy.after_read("users/*/email")
            def redact_emails(key, value, context):
                if context.get("redact_pii"):
                    return b"[REDACTED]"
                return value
        """
        def decorator(handler):
            policy = PatternPolicy(pattern, PolicyTrigger.AFTER_READ, handler)
            self._policies[PolicyTrigger.AFTER_READ].append(policy)
            return handler
        return decorator
    
    def before_delete(self, pattern: str):
        """Decorator for pre-delete policies."""
        def decorator(handler):
            policy = PatternPolicy(pattern, PolicyTrigger.BEFORE_DELETE, handler)
            self._policies[PolicyTrigger.BEFORE_DELETE].append(policy)
            return handler
        return decorator
    
    # =========================================================================
    # Rate Limiting API
    # =========================================================================
    
    def add_rate_limit(
        self,
        operation: str,  # "read", "write", "delete", or "all"
        max_per_minute: int,
        scope: str = "global",  # "global", "agent_id", "session_id"
    ):
        """
        Add a rate limit policy.
        
        Args:
            operation: Which operation to limit
            max_per_minute: Maximum operations per minute
            scope: Scope for the limit (global, per-agent, per-session)
            
        Example:
            # Global write limit
            policy.add_rate_limit("write", max_per_minute=1000)
            
            # Per-agent read limit
            policy.add_rate_limit("read", max_per_minute=100, scope="agent_id")
        """
        self._rate_limit_configs.append({
            "operation": operation,
            "max_per_minute": max_per_minute,
            "scope": scope,
        })
    
    def _check_rate_limit(self, operation: str, context: PolicyContext) -> bool:
        """Check if operation is allowed under rate limits."""
        for config in self._rate_limit_configs:
            if config["operation"] not in (operation, "all"):
                continue
            
            # Determine scope key
            scope = config["scope"]
            if scope == "global":
                scope_key = "global"
            elif scope == "agent_id":
                scope_key = context.agent_id or "unknown"
            elif scope == "session_id":
                scope_key = context.session_id or "unknown"
            else:
                scope_key = context.get(scope, "unknown")
            
            # Get or create rate limiter
            limiter_key = f"{config['operation']}:{scope}"
            if scope_key not in self._rate_limiters[limiter_key]:
                self._rate_limiters[limiter_key][scope_key] = RateLimiter(
                    config["max_per_minute"]
                )
            
            if not self._rate_limiters[limiter_key][scope_key].try_acquire():
                return False
        
        return True
    
    # =========================================================================
    # Audit API
    # =========================================================================
    
    def enable_audit(self, max_entries: int = 10000):
        """Enable audit logging."""
        self._audit_enabled = True
        self._max_audit_entries = max_entries
    
    def disable_audit(self):
        """Disable audit logging."""
        self._audit_enabled = False
    
    def get_audit_log(
        self,
        limit: int = 100,
        agent_id: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get audit log entries.
        
        Args:
            limit: Maximum entries to return
            agent_id: Filter by agent ID
            operation: Filter by operation type
            
        Returns:
            List of audit log entries
        """
        with self._lock:
            entries = self._audit_log[-limit:]
            
            if agent_id:
                entries = [e for e in entries if e.get("agent_id") == agent_id]
            if operation:
                entries = [e for e in entries if e.get("operation") == operation]
            
            return entries
    
    def _audit(self, operation: str, key: bytes, context: PolicyContext, result: str):
        """Add an audit log entry."""
        if not self._audit_enabled:
            return
        
        with self._lock:
            entry = {
                "timestamp": time.time(),
                "operation": operation,
                "key": key.decode("utf-8", errors="replace"),
                "agent_id": context.agent_id,
                "session_id": context.session_id,
                "result": result,
            }
            self._audit_log.append(entry)
            
            # Trim if too many entries
            if len(self._audit_log) > self._max_audit_entries:
                self._audit_log = self._audit_log[-self._max_audit_entries:]
    
    # =========================================================================
    # Evaluation Logic
    # =========================================================================
    
    def _evaluate_policies(
        self,
        trigger: PolicyTrigger,
        context: PolicyContext,
    ) -> PolicyResult:
        """Evaluate all policies for a trigger."""
        for policy in self._policies[trigger]:
            result = policy.evaluate(context)
            if result.action in (PolicyAction.DENY, PolicyAction.MODIFY):
                return result
        return PolicyResult(action=PolicyAction.ALLOW)
    
    # =========================================================================
    # Wrapped Database Operations
    # =========================================================================
    
    def put(
        self,
        key: bytes,
        value: bytes,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Put a value with policy enforcement.
        
        Args:
            key: Key bytes
            value: Value bytes
            context: Policy context (agent_id, session_id, etc.)
            
        Returns:
            True if write succeeded, False if blocked by policy
            
        Raises:
            PolicyViolation: If policy blocks the write
        """
        ctx = self._make_context("write", key, value, context)
        
        # Check rate limits
        if not self._check_rate_limit("write", ctx):
            self._audit("write", key, ctx, "rate_limited")
            raise PolicyViolation("Rate limit exceeded")
        
        # Evaluate before-write policies
        result = self._evaluate_policies(PolicyTrigger.BEFORE_WRITE, ctx)
        if result.action == PolicyAction.DENY:
            self._audit("write", key, ctx, "denied")
            raise PolicyViolation(result.reason or "Write blocked by policy")
        
        # Apply modifications if any
        write_value = result.modified_value if result.action == PolicyAction.MODIFY else value
        
        # Perform the write
        self._db.put(key, write_value)
        
        # Evaluate after-write policies
        ctx.value = write_value
        self._evaluate_policies(PolicyTrigger.AFTER_WRITE, ctx)
        
        self._audit("write", key, ctx, "allowed")
        return True
    
    def get(
        self,
        key: bytes,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[bytes]:
        """
        Get a value with policy enforcement.
        
        Args:
            key: Key bytes
            context: Policy context
            
        Returns:
            Value bytes (possibly modified by policy) or None
        """
        ctx = self._make_context("read", key, None, context)
        
        # Check rate limits
        if not self._check_rate_limit("read", ctx):
            self._audit("read", key, ctx, "rate_limited")
            raise PolicyViolation("Rate limit exceeded")
        
        # Evaluate before-read policies
        result = self._evaluate_policies(PolicyTrigger.BEFORE_READ, ctx)
        if result.action == PolicyAction.DENY:
            self._audit("read", key, ctx, "denied")
            raise PolicyViolation(result.reason or "Read blocked by policy")
        
        # Perform the read
        value = self._db.get(key)
        if value is None:
            return None
        
        # Evaluate after-read policies
        ctx.value = value
        result = self._evaluate_policies(PolicyTrigger.AFTER_READ, ctx)
        
        if result.action == PolicyAction.MODIFY:
            value = result.modified_value
        elif result.action == PolicyAction.DENY:
            self._audit("read", key, ctx, "redacted")
            return None
        
        self._audit("read", key, ctx, "allowed")
        return value
    
    def delete(
        self,
        key: bytes,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Delete a value with policy enforcement.
        
        Args:
            key: Key bytes
            context: Policy context
            
        Returns:
            True if delete succeeded
        """
        ctx = self._make_context("delete", key, None, context)
        
        # Check rate limits
        if not self._check_rate_limit("delete", ctx):
            self._audit("delete", key, ctx, "rate_limited")
            raise PolicyViolation("Rate limit exceeded")
        
        # Evaluate before-delete policies
        result = self._evaluate_policies(PolicyTrigger.BEFORE_DELETE, ctx)
        if result.action == PolicyAction.DENY:
            self._audit("delete", key, ctx, "denied")
            raise PolicyViolation(result.reason or "Delete blocked by policy")
        
        # Perform the delete
        self._db.delete(key)
        
        self._audit("delete", key, ctx, "allowed")
        return True
    
    def _make_context(
        self,
        operation: str,
        key: bytes,
        value: Optional[bytes],
        context: Optional[Dict[str, Any]],
    ) -> PolicyContext:
        """Create a policy context."""
        ctx = context or {}
        return PolicyContext(
            operation=operation,
            key=key,
            value=value,
            agent_id=ctx.get("agent_id"),
            session_id=ctx.get("session_id"),
            custom=ctx,
        )


class PolicyViolation(Exception):
    """Raised when a policy blocks an operation."""
    pass


# =============================================================================
# Built-in Policy Helpers
# =============================================================================

def deny_all(key: bytes, value: Optional[bytes], context: Dict) -> PolicyAction:
    """Policy that denies all matching operations."""
    return PolicyAction.DENY


def allow_all(key: bytes, value: Optional[bytes], context: Dict) -> PolicyAction:
    """Policy that allows all matching operations."""
    return PolicyAction.ALLOW


def require_agent_id(key: bytes, value: Optional[bytes], context: Dict) -> PolicyAction:
    """Policy that requires an agent_id in context."""
    if context.get("agent_id"):
        return PolicyAction.ALLOW
    return PolicyAction.DENY


def redact_value(replacement: bytes = b"[REDACTED]"):
    """Policy factory that redacts values."""
    def handler(key: bytes, value: Optional[bytes], context: Dict) -> bytes:
        return replacement
    return handler


def log_and_allow(logger=None):
    """Policy factory that logs operations but allows them."""
    def handler(key: bytes, value: Optional[bytes], context: Dict) -> PolicyAction:
        msg = f"Operation on {key.decode('utf-8', errors='replace')}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return PolicyAction.ALLOW
    return handler
