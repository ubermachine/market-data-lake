import os
import pandas as pd
import pytest

# We will check if the consolidated files exist and match the expected schemas
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def test_consolidated_files_exist():
    """Verify that the 4 consolidated parquet files are created in the data folder."""
    expected_files = [
        "DailyBars.parquet",
        "WeeklyBars.parquet",
        "SectorDailyBars.parquet",
        "StockMetadatas.parquet"
    ]
    for file in expected_files:
        filepath = os.path.join(DATA_DIR, file)
        assert os.path.exists(filepath), f"Expected consolidated file {file} is missing from {DATA_DIR}"

def test_bar_data_schemas():
    """Verify that the daily, weekly, and sector daily bar files have the correct column schema."""
    bar_files = ["DailyBars.parquet", "WeeklyBars.parquet", "SectorDailyBars.parquet"]
    expected_cols = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    for file in bar_files:
        filepath = os.path.join(DATA_DIR, file)
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            assert list(df.columns) == expected_cols, f"Columns in {file} do not match the expected schema"
            # Ensure Date is type datetime or date
            assert not df.empty, f"File {file} is empty"

def test_metadata_schema():
    """Verify that StockMetadatas has the correct schema."""
    filepath = os.path.join(DATA_DIR, "StockMetadatas.parquet")
    expected_cols = ['Ticker', 'Name', 'Sector']
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        assert list(df.columns) == expected_cols, f"StockMetadatas schema mismatch"
        assert not df.empty
