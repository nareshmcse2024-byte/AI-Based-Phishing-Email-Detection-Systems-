"""Exports analysis metadata only; no raw message, sender address or URL values."""
from __future__ import annotations
import pandas as pd

SAFE_COLUMNS = ["id", "created_at", "category", "score", "url_count", "signals", "reviewed_category", "reviewed_at"]

def sanitized_report(records: list[dict]) -> bytes:
    df = pd.DataFrame(records)
    if df.empty: df = pd.DataFrame(columns=SAFE_COLUMNS)
    else:
        df = df[[c for c in SAFE_COLUMNS if c in df.columns]]
    return df.to_csv(index=False).encode("utf-8")

def sanitize_training_csv(file_obj, max_bytes: int, max_rows: int) -> pd.DataFrame:
    content = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read(max_bytes + 1)
    if len(content) > max_bytes: raise ValueError(f"CSV exceeds the {max_bytes // (1024*1024)} MB upload limit")
    from io import BytesIO
    df = pd.read_csv(BytesIO(content))
    if len(df) > max_rows: raise ValueError(f"CSV exceeds the {max_rows:,}-row limit")
    required = {"subject", "body", "label"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"CSV must include columns: {', '.join(sorted(required))}; missing: {', '.join(sorted(missing))}")
    return df
