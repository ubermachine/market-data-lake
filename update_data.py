import os
import json
import pandas as pd
import numpy as np
import yfinance as yf

# Sector indices to be separated into SectorDailyBars.parquet
SECTOR_TICKERS = [
    "^NSEI", "^NSEBANK", "^CNXAUTO", "^CNXIT", "^CNXPHARMA", 
    "^CNXMETAL", "^CNXENERGY", "^CNXFMCG", "^CNXMEDIA", "^CNXREALTY", 
    "^CNXPSUBANK", "^CNXINFRA", "NIFTY_FIN_SERVICE.NS", 
    "NIFTY_OIL_AND_GAS.NS", "^CNXCONSUM"
]

def load_config(config_path: str) -> tuple[list[str], list[str]]:
    """Loads tickers and intervals from the json configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get("tickers", []), config.get("intervals", [])

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans gaps by forward-filling at most 2 consecutive days and dropping remaining NaNs."""
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Forward fill up to 2 days
    cols_to_fill = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[cols_to_fill] = df[cols_to_fill].ffill(limit=2)
    
    # Drop rows containing NaNs
    df = df.dropna(subset=['Close'])
    df = df.reset_index(drop=True)
    return df

def resample_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resamples daily OHLCV data into weekly bars."""
    if df_daily.empty:
        return pd.DataFrame()
    df = df_daily.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    resampled_list = []
    for ticker, group in df.groupby('Ticker'):
        group_idx = group.set_index('Date')
        weekly = group_idx.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna(subset=['Close'])
        weekly = weekly.reset_index()
        weekly['Ticker'] = ticker
        resampled_list.append(weekly)
        
    if not resampled_list:
        return pd.DataFrame()
        
    resampled_df = pd.concat(resampled_list, ignore_index=True)
    # Match schema column order: Ticker first
    resampled_df = resampled_df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return resampled_df

# For backward compatibility test support
def download_symbol_data(symbol: str, interval: str, period: str = "max") -> pd.DataFrame:
    """Downloads daily data for a single ticker (kept for compatibility in tests)."""
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data for: {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    rename_map = {'datetime': 'Date', 'index': 'Date'}
    for col in df.columns:
        if str(col).lower() == 'date':
            rename_map[col] = 'Date'
        elif str(col).lower() in ['open', 'high', 'low', 'close', 'volume']:
            rename_map[col] = str(col).capitalize()
    df = df.rename(columns=rename_map)
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    df = df[expected_cols]
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

def save_to_parquet(df: pd.DataFrame, symbol: str, interval: str, output_dir: str) -> str:
    """Saves a dataframe to individual parquet (kept for backward compatibility tests)."""
    os.makedirs(output_dir, exist_ok=True)
    safe_symbol = symbol.replace("^", "").replace(".", "_").lower()
    filepath = os.path.join(output_dir, f"{safe_symbol}_{interval}.parquet")
    df_to_save = df.copy()
    df_to_save['Date'] = pd.to_datetime(df_to_save['Date'])
    df_to_save.to_parquet(filepath, index=False, engine='pyarrow')
    return filepath


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "tickers.json")
    output_dir = os.path.join(base_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        tickers, intervals = load_config(config_path)
        # We only ingest '1d' daily data because weekly and metadata are compiled from it
        print(f"Downloading {len(tickers)} tickers in batch...")
        
        try:
            bulk_df = yf.download(tickers, interval="1d", period="max", group_by='ticker', progress=False)
        except Exception as e:
            print(f"Bulk download failed: {e}")
            bulk_df = None
            
        if bulk_df is not None:
            stock_data_list = []
            sector_data_list = []
            stock_tickers_processed = []
            
            for ticker in tickers:
                try:
                    if isinstance(bulk_df.columns, pd.MultiIndex):
                        if ticker in bulk_df.columns.levels[0]:
                            df = bulk_df[ticker].dropna(how='all').reset_index()
                        else:
                            continue
                    else:
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
                    
                    if df_cleaned.empty:
                        continue
                        
                    # Append Ticker column
                    df_cleaned['Ticker'] = ticker
                    # Reorder columns
                    df_cleaned = df_cleaned[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    
                    if ticker in SECTOR_TICKERS:
                        sector_data_list.append(df_cleaned)
                    else:
                        stock_data_list.append(df_cleaned)
                        stock_tickers_processed.append(ticker)
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
            
            # 1. Compile SectorDailyBars.parquet
            if sector_data_list:
                df_sector = pd.concat(sector_data_list, ignore_index=True)
                df_sector['Date'] = pd.to_datetime(df_sector['Date'])
                df_sector = df_sector.sort_values(['Ticker', 'Date']).reset_index(drop=True)
                sector_path = os.path.join(output_dir, "SectorDailyBars.parquet")
                df_sector.to_parquet(sector_path, index=False, engine='pyarrow')
                print(f"Saved consolidated SectorDailyBars.parquet with {len(df_sector)} rows.")
            
            # 2. Compile DailyBars.parquet
            if stock_data_list:
                df_stock_daily = pd.concat(stock_data_list, ignore_index=True)
                df_stock_daily['Date'] = pd.to_datetime(df_stock_daily['Date'])
                df_stock_daily = df_stock_daily.sort_values(['Ticker', 'Date']).reset_index(drop=True)
                daily_path = os.path.join(output_dir, "DailyBars.parquet")
                df_stock_daily.to_parquet(daily_path, index=False, engine='pyarrow')
                print(f"Saved consolidated DailyBars.parquet with {len(df_stock_daily)} rows.")
                
                # 3. Compile WeeklyBars.parquet
                df_stock_weekly = resample_weekly(df_stock_daily)
                if not df_stock_weekly.empty:
                    df_stock_weekly = df_stock_weekly.sort_values(['Ticker', 'Date']).reset_index(drop=True)
                    weekly_path = os.path.join(output_dir, "WeeklyBars.parquet")
                    df_stock_weekly.to_parquet(weekly_path, index=False, engine='pyarrow')
                    print(f"Saved consolidated WeeklyBars.parquet with {len(df_stock_weekly)} rows.")
            
            # 4. Compile StockMetadatas.parquet
            if stock_tickers_processed:
                metadata_rows = []
                for st_ticker in sorted(stock_tickers_processed):
                    name = st_ticker.replace(".NS", "")
                    metadata_rows.append({"Ticker": st_ticker, "Name": name, "Sector": "NSE"})
                df_meta = pd.DataFrame(metadata_rows)
                meta_path = os.path.join(output_dir, "StockMetadatas.parquet")
                df_meta.to_parquet(meta_path, index=False, engine='pyarrow')
                print(f"Saved consolidated StockMetadatas.parquet with {len(df_meta)} rows.")
                
    except Exception as e:
        print(f"Failed to run data ingestion: {e}")
