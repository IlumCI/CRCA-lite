"""Canonicalization utilities for deterministic hashing.

This module provides functions to convert objects to deterministic byte representations
and compute stable hashes. This is critical for ensuring reproducible decision-making
in the policy engine.

All decision artifacts must use stable_hash() to ensure identical inputs produce
identical decision hashes (R0 requirement).
"""

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Union


def quantize_float(x: float, mode: str = "decimal", p: float = 1e-6) -> float:
    """
    Quantize a float to ensure numeric stability.
    
    Args:
        x: The float value to quantize
        mode: Quantization mode ("decimal" rounds to nearest p, "truncate" truncates)
        p: Precision parameter (default: 1e-6)
        
    Returns:
        float: Quantized value
    """
    if mode == "decimal":
        # Round to nearest multiple of p
        return round(x / p) * p
    elif mode == "truncate":
        # Truncate to p decimal places
        return float(f"{x:.{len(str(p).split('.')[-1])}f}")
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")


def canonical_bytes(obj: Any) -> bytes:
    """
    Convert an object to a deterministic byte representation.
    
    Handles:
    - Dicts: sorted keys, recursive canonicalization
    - Lists: preserve order, recursive canonicalization
    - Sets: convert to sorted list
    - Floats: quantize for stability
    - Basic types: str, int, bool, None
    
    Args:
        obj: The object to canonicalize
        
    Returns:
        bytes: Deterministic byte representation
    """
    if obj is None:
        return b"null"
    elif isinstance(obj, bool):
        return b"true" if obj else b"false"
    elif isinstance(obj, int):
        return str(obj).encode("utf-8")
    elif isinstance(obj, float):
        # Quantize float for stability
        quantized = quantize_float(obj)
        return str(quantized).encode("utf-8")
    elif isinstance(obj, str):
        return obj.encode("utf-8")
    elif isinstance(obj, bytes):
        return obj
    elif isinstance(obj, dict):
        # Sort keys deterministically
        sorted_items = sorted(obj.items(), key=lambda x: x[0])
        parts = []
        for key, value in sorted_items:
            key_bytes = canonical_bytes(key)
            value_bytes = canonical_bytes(value)
            parts.append(key_bytes + b":" + value_bytes)
        return b"{" + b",".join(parts) + b"}"
    elif isinstance(obj, (list, tuple)):
        # Preserve order, canonicalize each element
        parts = [canonical_bytes(item) for item in obj]
        return b"[" + b",".join(parts) + b"]"
    elif isinstance(obj, set):
        # Convert set to sorted list
        sorted_list = sorted(obj, key=lambda x: (str(type(x).__name__), str(x)))
        return canonical_bytes(sorted_list)
    else:
        # For other types, try JSON serialization as fallback
        try:
            json_str = json.dumps(obj, sort_keys=True, default=str)
            return json_str.encode("utf-8")
        except (TypeError, ValueError) as e:
            # Last resort: use string representation
            return str(obj).encode("utf-8")


def stable_hash(obj: Any) -> str:
    """
    Compute a stable SHA256 hash of an object.
    
    This function ensures that identical objects (after canonicalization)
    produce identical hashes. Used for all decision artifacts in the policy engine.
    
    Args:
        obj: The object to hash
        
    Returns:
        str: Hexadecimal SHA256 hash (64 characters)
    """
    canonical = canonical_bytes(obj)
    hash_obj = hashlib.sha256(canonical)
    return hash_obj.hexdigest()


def stable_sort(
    items: List[Any],
    key: Optional[Callable[[Any], Any]] = None,
    tie_break: str = "hash"
) -> List[Any]:
    """
    Sort items deterministically with hash-based tie-breaking.
    
    Args:
        items: List of items to sort
        key: Optional key function (like sorted() key parameter)
        tie_break: Tie-breaking method ("hash" uses stable_hash, "repr" uses repr)
        
    Returns:
        List[Any]: Sorted list
    """
    if key is None:
        key = lambda x: x
    
    def tie_breaker(item: Any) -> str:
        if tie_break == "hash":
            return stable_hash(item)
        elif tie_break == "repr":
            return repr(item)
        else:
            raise ValueError(f"Unknown tie_break method: {tie_break}")
    
    # Sort by key, then by tie-breaker for stability
    return sorted(items, key=lambda x: (key(x), tie_breaker(x)))

