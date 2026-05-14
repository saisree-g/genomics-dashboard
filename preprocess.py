"""
preprocess.py — Sequence cleaning and validation pipeline.

clean_sequence()      → uppercase, strip whitespace, keep only ACGT
is_valid()            → checks ACGT ratio and minimum length
remove_duplicates()   → exact deduplication
preprocess_pool()     → full pipeline: clean → validate → deduplicate
preprocessing_report() → summary stats before vs after cleaning
"""

from collections import Counter

from config import MIN_ACGT_RATIO, MIN_SEQ_LEN

VALID_BASES = frozenset("ACGT")


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_sequence(seq: str) -> str:
    """
    Uppercase, strip whitespace and digits, remove everything that isn't
    a nucleotide letter. Returns only ACGT characters (non-ACGT letters
    like N, R, Y, W etc. are dropped entirely — not replaced).
    """
    seq = seq.upper().strip()
    return "".join(c for c in seq if c in VALID_BASES)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def is_valid(seq: str, min_len: int = MIN_SEQ_LEN) -> bool:
    """
    A sequence is valid if:
      1. Length >= min_len after cleaning
      2. ACGT ratio >= MIN_ACGT_RATIO (already guaranteed after clean_sequence,
         but checked here in case a raw sequence is passed directly)
      3. Not a low-complexity repeat (e.g. AAAAAAA… or ATATATATAT…)
    """
    if len(seq) < min_len:
        return False

    # ACGT ratio check (for raw/uncleaned sequences)
    acgt = sum(c in VALID_BASES for c in seq)
    if acgt / len(seq) < MIN_ACGT_RATIO:
        return False

    # Low-complexity filter: if any single base > 80% → likely artifact
    counts = Counter(seq)
    if max(counts.values()) / len(seq) > 0.80:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(sequences: list[str]) -> list[str]:
    """Exact deduplication preserving order."""
    seen: set[str] = set()
    unique: list[str] = []
    for s in sequences:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_pool(
    raw_sequences: list[str],
    seq_length: int,
    min_len: int = MIN_SEQ_LEN,
) -> tuple[list[str], dict]:
    """
    Full preprocessing pipeline:
      1. Clean  → strip non-ACGT characters
      2. Filter → drop sequences that are too short or low-complexity
      3. Deduplicate → remove exact duplicates
      4. Truncate → cut to seq_length

    Returns (processed_sequences, report_dict)
    """
    n_raw = len(raw_sequences)

    # Step 1 — clean
    cleaned = [clean_sequence(s) for s in raw_sequences]

    # Step 2 — validate (use seq_length as the effective min_len if larger)
    effective_min = max(min_len, seq_length)
    valid = [s for s in cleaned if is_valid(s, min_len=effective_min)]
    n_after_filter = len(valid)

    # Step 3 — deduplicate
    unique = remove_duplicates(valid)
    n_after_dedup = len(unique)

    # Step 4 — truncate to fixed length
    truncated = [s[:seq_length] for s in unique]

    report = {
        "raw":          n_raw,
        "after_filter": n_after_filter,
        "after_dedup":  n_after_dedup,
        "dropped_low_quality": n_raw - n_after_filter,
        "dropped_duplicates":  n_after_filter - n_after_dedup,
        "final":        len(truncated),
    }

    return truncated, report


# ─────────────────────────────────────────────────────────────────────────────
# REPORT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def preprocessing_report(reports: dict[str, dict]) -> dict:
    """Merge per-class preprocessing reports into a summary dict."""
    total_raw   = sum(r["raw"]   for r in reports.values())
    total_final = sum(r["final"] for r in reports.values())
    total_dropped_quality = sum(r["dropped_low_quality"] for r in reports.values())
    total_dropped_dupes   = sum(r["dropped_duplicates"]  for r in reports.values())
    return {
        "per_class":            reports,
        "total_raw":            total_raw,
        "total_final":          total_final,
        "total_dropped_quality": total_dropped_quality,
        "total_dropped_dupes":  total_dropped_dupes,
        "retention_rate":       round(total_final / total_raw * 100, 1) if total_raw else 0,
    }
