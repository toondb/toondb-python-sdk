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
Unified Output Format Semantics

Provides format enums for query results and LLM context packaging.
This mirrors the Rust toondb-client format module for consistency.
"""

from enum import Enum
from typing import Optional


class FormatConversionError(Exception):
    """Error when format conversion fails."""
    
    def __init__(self, from_format: str, to_format: str, reason: str):
        self.from_format = from_format
        self.to_format = to_format
        self.reason = reason
        super().__init__(f"Cannot convert {from_format} to {to_format}: {reason}")


class WireFormat(Enum):
    """Output format for query results sent to clients.
    
    These formats are optimized for transmission efficiency and
    client-side processing.
    """
    
    TOON = "toon"
    """TOON format (default, 40-66% fewer tokens than JSON).
    Optimized for LLM consumption."""
    
    JSON = "json"
    """Standard JSON for compatibility."""
    
    COLUMNAR = "columnar"
    """Raw columnar format for analytics.
    More efficient for large result sets with projection pushdown."""
    
    @classmethod
    def from_string(cls, s: str) -> "WireFormat":
        """Parse format from string."""
        s_lower = s.lower()
        if s_lower == "toon":
            return cls.TOON
        elif s_lower == "json":
            return cls.JSON
        elif s_lower in ("columnar", "column"):
            return cls.COLUMNAR
        else:
            raise FormatConversionError(
                s, "WireFormat",
                f"Unknown format '{s}'. Valid: toon, json, columnar"
            )
    
    def __str__(self) -> str:
        return self.value


class ContextFormat(Enum):
    """Output format for LLM context packaging.
    
    These formats are optimized for readability and token efficiency
    when constructing prompts for language models.
    """
    
    TOON = "toon"
    """TOON format (default, token-efficient).
    Structured data with minimal syntax overhead."""
    
    JSON = "json"
    """JSON format.
    Widely understood by LLMs, good for structured data."""
    
    MARKDOWN = "markdown"
    """Markdown format.
    Best for human-readable context with formatting."""
    
    @classmethod
    def from_string(cls, s: str) -> "ContextFormat":
        """Parse format from string."""
        s_lower = s.lower()
        if s_lower == "toon":
            return cls.TOON
        elif s_lower == "json":
            return cls.JSON
        elif s_lower in ("markdown", "md"):
            return cls.MARKDOWN
        else:
            raise FormatConversionError(
                s, "ContextFormat",
                f"Unknown format '{s}'. Valid: toon, json, markdown"
            )
    
    def __str__(self) -> str:
        return self.value


class CanonicalFormat(Enum):
    """Canonical storage format (server-side only).
    
    This is the format used for internal storage and is optimized
    for storage efficiency and query performance.
    """
    
    TOON = "toon"
    """TOON canonical format."""
    
    def __str__(self) -> str:
        return self.value


class FormatCapabilities:
    """Helper to check format capabilities and conversions."""
    
    @staticmethod
    def wire_to_context(wire: WireFormat) -> Optional[ContextFormat]:
        """Convert WireFormat to ContextFormat if compatible."""
        if wire == WireFormat.TOON:
            return ContextFormat.TOON
        elif wire == WireFormat.JSON:
            return ContextFormat.JSON
        else:
            return None
    
    @staticmethod
    def context_to_wire(ctx: ContextFormat) -> Optional[WireFormat]:
        """Convert ContextFormat to WireFormat if compatible."""
        if ctx == ContextFormat.TOON:
            return WireFormat.TOON
        elif ctx == ContextFormat.JSON:
            return WireFormat.JSON
        else:
            return None
    
    @staticmethod
    def supports_round_trip(fmt: WireFormat) -> bool:
        """Check if format supports round-trip: decode(encode(x)) = x."""
        return fmt in (WireFormat.TOON, WireFormat.JSON)


__all__ = [
    "FormatConversionError",
    "WireFormat",
    "ContextFormat", 
    "CanonicalFormat",
    "FormatCapabilities",
]
