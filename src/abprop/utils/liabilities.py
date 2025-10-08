"""Heuristics for antibody liability motif detection."""

from __future__ import annotations

import re
from typing import Dict, Mapping

DEFAULT_PATTERNS = {
    "nglyc": r"N[^P][ST]",  # N-X-S/T with proline exclusion
    "deamidation": r"NG",
    "isomerization": r"DG",
    "oxidation": r"M",
}

# Canonical disulfide spacing in antibodies is typically between 2 and 20 residues.
_CANONICAL_DISULFIDE_SPAN = (2, 20)


def _count_overlapping(pattern: str, sequence: str) -> int:
    """Count overlapping occurrences of a regex pattern."""
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    # Use lookahead to capture overlapping matches.
    lookahead = re.compile(f"(?=({pattern}))", flags=re.IGNORECASE)
    return sum(1 for _ in lookahead.finditer(sequence))


def _estimate_free_cysteines(sequence: str) -> int:
    """Estimate number of cysteines that do not participate in canonical disulfide pairs."""
    indices = [idx for idx, residue in enumerate(sequence) if residue == "C"]
    if not indices:
        return 0

    min_span, max_span = _CANONICAL_DISULFIDE_SPAN
    used = set()
    for i, start in enumerate(indices):
        if start in used:
            continue
        for end in indices[i + 1 :]:
            if end in used:
                continue
            span = end - start
            if span < min_span:
                continue
            if span > max_span:
                break
            used.add(start)
            used.add(end)
            break

    paired = len(used)
    free = len(indices) - paired
    return free


def find_motifs(sequence: str, extra_patterns: Mapping[str, str] | None = None) -> Dict[str, int]:
    """
    Count liability motifs in a sequence.

    Args:
        sequence: Amino-acid sequence (case-insensitive).
        extra_patterns: Optional mapping of {name: regex} for additional motif detection.
    """
    sequence = sequence.upper()
    counts: Dict[str, int] = {}

    for name, pattern in DEFAULT_PATTERNS.items():
        if name == "oxidation":
            counts[name] = sequence.count("M")
        else:
            counts[name] = _count_overlapping(pattern, sequence)

    counts["free_cysteines"] = _estimate_free_cysteines(sequence)

    if extra_patterns:
        for name, pattern in extra_patterns.items():
            counts[name] = _count_overlapping(pattern, sequence)

    return counts


def normalize_by_length(counts: Mapping[str, int], length: int) -> Dict[str, float]:
    """Return length-normalized liability scores."""
    if length <= 0:
        return {key: 0.0 for key in counts}
    return {key: value / length for key, value in counts.items()}


__all__ = ["find_motifs", "normalize_by_length"]
