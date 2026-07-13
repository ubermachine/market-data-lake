import pytest
import pandas as pd
import numpy as np
from update_data import merge_and_deduplicate

def test_merge_and_deduplicate_basic():
    """Verify that merging old and new data removes duplicates and keeps the newest row."""
    # 1. Mock old data (dates 2026-01-01 to 2026-01-03)
    old_dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    df_old = pd.DataFrame({
        "Ticker": ["TCS.NS"] * 3,
        "Date": old_dates,
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [95.0, 96.0, 97.0],
        "Close": [100.0, 101.0, 102.0],
        "Volume": [1000, 1100, 1200]
    })
    
    # 2. Mock new data (dates 2026-01-03 to 2026-01-05, with overlapping date 2026-01-03 having updated volume/close)
    new_dates = pd.to_datetime(["2026-01-03", "2026-01-04", "2026-01-05"])
    df_new = pd.DataFrame({
        "Ticker": ["TCS.NS"] * 3,
        "Date": new_dates,
        "Open": [102.0, 103.0, 104.0],
        "High": [107.0, 108.0, 109.0],
        "Low": [97.0, 98.0, 99.0],
        "Close": [102.5, 103.0, 104.0],  # 102.5 is the updated close for 2026-01-03
        "Volume": [1500, 1300, 1400]     # 1500 is the updated volume for 2026-01-03
    })
    
    # Merge
    merged = merge_and_deduplicate(df_old, df_new)
    
    # Assertions
    assert len(merged) == 5  # No duplicates
    assert list(merged["Date"]) == list(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]))
    
    # Verify that the overlapping date (2026-01-03) has the updated values from df_new
    overlap_row = merged[merged["Date"] == "2026-01-03"]
    assert overlap_row["Close"].values[0] == 102.5
    assert overlap_row["Volume"].values[0] == 1500

def test_merge_and_deduplicate_empty():
    """Verify merging works correctly if one of the DataFrames is empty."""
    df_data = pd.DataFrame({
        "Ticker": ["TCS.NS"],
        "Date": pd.to_datetime(["2026-01-01"]),
        "Open": [100.0], "High": [100.0], "Low": [100.0], "Close": [100.0], "Volume": [100]
    })
    
    assert len(merge_and_deduplicate(df_data, pd.DataFrame())) == 1
    assert len(merge_and_deduplicate(pd.DataFrame(), df_data)) == 1
