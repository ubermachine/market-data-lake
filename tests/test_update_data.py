import os
import json
import pytest
import pandas as pd
import numpy as np

# We import the functions we intend to implement in update_data.py
# Even if they don't exist yet, we import them so the tests will fail initially.
from update_data import load_config, download_symbol_data, preprocess_data, save_to_parquet

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "tickers.json")

def test_load_config():
    """Verify that configuration file is correctly parsed."""
    tickers, intervals = load_config(CONFIG_PATH)
    assert isinstance(tickers, list)
    assert len(tickers) > 0
    assert "^NSEI" in tickers
    assert "1d" in intervals


def test_download_symbol_data():
    """Verify downloading data from Yahoo Finance yields correct schema."""
    # We mock or download a small sample (e.g. 5 days of AAPL or ^NSEI)
    df = download_symbol_data("TCS.NS", interval="1d", period="1mo")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    
    # Expected columns
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in expected_cols:
        assert col in df.columns


def test_preprocess_data():
    """Verify preprocessing correctly fills gaps up to 2 days and drops larger gaps."""
    # Create mock dataframe with NaNs
    dates = pd.date_range(start="2026-01-01", periods=10).date
    data = {
        "Date": dates,
        "Open": [10.0, 11.0, np.nan, 13.0, 14.0, np.nan, np.nan, 17.0, np.nan, np.nan],
        "High": [10.5, 11.5, np.nan, 13.5, 14.5, np.nan, np.nan, 17.5, np.nan, np.nan],
        "Low": [9.5, 10.5, np.nan, 12.5, 13.5, np.nan, np.nan, 16.5, np.nan, np.nan],
        "Close": [10.0, 11.0, np.nan, 13.0, 14.0, np.nan, np.nan, 17.0, np.nan, np.nan],
        "Volume": [100, 110, np.nan, 130, 140, np.nan, np.nan, 170, np.nan, np.nan]
    }
    df = pd.DataFrame(data)
    
    # Preprocess
    cleaned = preprocess_data(df)
    
    # Assertions:
    # 1. 2026-01-03 (index 2) had 1 day NaN. It should be forward filled from index 1.
    assert cleaned.loc[cleaned["Date"] == dates[2], "Close"].values[0] == 11.0
    
    # 2. 2026-01-06 & 07 (index 5 & 6) had 2 days of NaN. It should be forward filled.
    assert cleaned.loc[cleaned["Date"] == dates[6], "Close"].values[0] == 14.0
    
    # 3. 2026-01-09 & 10 (index 8 & 9) are at the end, and if they represent >2 days gap or remaining NaNs, they should be dropped.
    # Actually, let's create a 3-day gap to test dropping
    data_large_gap = {
        "Date": pd.date_range(start="2026-01-01", periods=5).date,
        "Open": [10.0, np.nan, np.nan, np.nan, 14.0],
        "High": [10.5, np.nan, np.nan, np.nan, 14.5],
        "Low": [9.5, np.nan, np.nan, np.nan, 13.5],
        "Close": [10.0, np.nan, np.nan, np.nan, 14.0],
        "Volume": [100, np.nan, np.nan, np.nan, 140]
    }
    df_large = pd.DataFrame(data_large_gap)
    cleaned_large = preprocess_data(df_large)
    
    # Since gap is 3 days, it cannot be fully filled, so remaining NaNs are dropped.
    # The middle rows (index 1, 2, 3) had 3 NaNs. ffill(limit=2) leaves index 3 NaN.
    # dropna should drop it. So rows with NaNs should be deleted.
    assert not cleaned_large.isnull().values.any()
    assert len(cleaned_large) < 5


def test_save_parquet(tmp_path):
    """Verify that saving data to Parquet works correctly."""
    dates = pd.date_range(start="2026-01-01", periods=5).date
    data = {
        "Date": dates,
        "Open": [10.0, 11.0, 12.0, 13.0, 14.0],
        "High": [10.5, 11.5, 12.5, 13.5, 14.5],
        "Low": [9.5, 10.5, 11.5, 12.5, 13.5],
        "Close": [10.0, 11.0, 12.0, 13.0, 14.0],
        "Volume": [100, 110, 120, 130, 140]
    }
    df = pd.DataFrame(data)
    
    output_dir = tmp_path / "data"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = save_to_parquet(df, "TEST", "1d", output_dir)
    assert os.path.exists(filepath)
    
    # Read it back and verify
    loaded = pd.read_parquet(filepath)
    # Date in Parquet can be saved as datetime64, check values
    assert len(loaded) == 5
    assert list(loaded.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
