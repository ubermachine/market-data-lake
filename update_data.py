import os
import json
import pandas as pd
import yfinance as yf

def load_config(config_path: str) -> tuple[list[str], list[str]]:
    """Loads tickers and intervals from the json configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get("tickers", []), config.get("intervals", [])

def download_symbol_data(symbol: str, interval: str, period: str = "max") -> pd.DataFrame:
    """Downloads historical OHLCV data from Yahoo Finance for a symbol and interval."""
    # yfinance uses '1wk' for weekly and '1mo' for monthly, map them if needed
    yf_interval = interval
    if interval == '1w':
        yf_interval = '1wk'
    elif interval == '1m':
        yf_interval = '1mo'
        
    df = yf.download(symbol, interval=yf_interval, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for symbol: {symbol}")
        
    # Flatten MultiIndex columns if present (common in newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    
    # Rename columns to standard schema
    rename_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == 'date' or col_lower == 'datetime':
            rename_map[col] = 'Date'
        elif col_lower == 'open':
            rename_map[col] = 'Open'
        elif col_lower == 'high':
            rename_map[col] = 'High'
        elif col_lower == 'low':
            rename_map[col] = 'Low'
        elif col_lower == 'close':
            rename_map[col] = 'Close'
        elif col_lower == 'volume':
            rename_map[col] = 'Volume'
            
    df = df.rename(columns=rename_map)
    
    # Keep only the standard OHLCV columns
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    # Ensure they exist in df
    for col in expected_cols:
        if col not in df.columns:
            # Create dummy or raise
            df[col] = None
            
    df = df[expected_cols]
    
    # Ensure Date is simple date (no timezone or time if daily/weekly/monthly)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans gaps by forward-filling at most 2 consecutive days and dropping remaining NaNs."""
    df = df.copy()
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Forward fill up to 2 days
    cols_to_fill = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[cols_to_fill] = df[cols_to_fill].ffill(limit=2)
    
    # Drop rows containing NaNs
    df = df.dropna(subset=['Close'])
    df = df.reset_index(drop=True)
    return df

def save_to_parquet(df: pd.DataFrame, symbol: str, interval: str, output_dir: str) -> str:
    """Saves the DataFrame to a Parquet file named {symbol}_{interval}.parquet."""
    os.makedirs(output_dir, exist_ok=True)
    # Sanitize symbol for filename
    safe_symbol = symbol.replace("^", "").replace(".", "_").lower()
    filename = f"{safe_symbol}_{interval}.parquet"
    filepath = os.path.join(output_dir, filename)
    
    df_to_save = df.copy()
    # Convert date to timestamp for Parquet compatibility
    df_to_save['Date'] = pd.to_datetime(df_to_save['Date'])
    df_to_save.to_parquet(filepath, index=False, engine='pyarrow')
    return filepath

if __name__ == "__main__":
    # If run directly, read config, download all, and save to a 'data' folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "tickers.json")
    output_dir = os.path.join(base_dir, "data")
    
    try:
        tickers, intervals = load_config(config_path)
        for interval in intervals:
            print(f"Downloading {len(tickers)} tickers in batch for interval {interval}...")
            yf_interval = interval
            if interval == '1w':
                yf_interval = '1wk'
            elif interval == '1m':
                yf_interval = '1mo'
                
            try:
                # Batch download all tickers at once
                bulk_df = yf.download(tickers, interval=yf_interval, period="max", group_by='ticker', progress=False)
            except Exception as e:
                print(f"Bulk download failed: {e}")
                continue
                
            for ticker in tickers:
                try:
                    if isinstance(bulk_df.columns, pd.MultiIndex):
                        if ticker in bulk_df.columns.levels[0]:
                            df = bulk_df[ticker].dropna(how='all').reset_index()
                        else:
                            continue
                    else:
                        # Fallback for single ticker query or non-MultiIndex output
                        df = bulk_df.dropna(how='all').reset_index()
                        
                    if df.empty:
                        continue
                        
                    # Standardize columns
                    rename_map = {}
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if col_lower in ['date', 'datetime', 'index']:
                            rename_map[col] = 'Date'
                        elif col_lower == 'open':
                            rename_map[col] = 'Open'
                        elif col_lower == 'high':
                            rename_map[col] = 'High'
                        elif col_lower == 'low':
                            rename_map[col] = 'Low'
                        elif col_lower == 'close':
                            rename_map[col] = 'Close'
                        elif col_lower == 'volume':
                            rename_map[col] = 'Volume'
                            
                    df = df.rename(columns=rename_map)
                    
                    expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
                    for col in expected_cols:
                        if col not in df.columns:
                            df[col] = None
                            
                    df = df[expected_cols]
                    df['Date'] = pd.to_datetime(df['Date']).dt.date
                    
                    df_cleaned = preprocess_data(df)
                    path = save_to_parquet(df_cleaned, ticker, interval, output_dir)
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
            print(f"Finished processing all tickers for interval {interval}")
    except Exception as e:
        print(f"Failed to run data ingestion: {e}")
