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

def merge_and_deduplicate(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Combines old and new bar DataFrames, deduplicating on ['Ticker', 'Date'] keeping the latest."""
    if df_old.empty:
        return df_new
    if df_new.empty:
        return df_old
        
    combined = pd.concat([df_old, df_new], ignore_index=True)
    # Ensure Date is datetime64 for matching and sorting
    combined['Date'] = pd.to_datetime(combined['Date'])
    
    # Deduplicate: keep the last occurrence (which is df_new)
    combined = combined.drop_duplicates(subset=['Ticker', 'Date'], keep='last')
    combined = combined.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    return combined

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
        
        # Define consolidated output paths
        sector_path = os.path.join(output_dir, "SectorDailyBars.parquet")
        daily_path = os.path.join(output_dir, "DailyBars.parquet")
        weekly_path = os.path.join(output_dir, "WeeklyBars.parquet")
        meta_path = os.path.join(output_dir, "StockMetadatas.parquet")
        
        # Load existing data lake cache
        df_sector_old = pd.DataFrame()
        df_daily_old = pd.DataFrame()
        
        if os.path.exists(sector_path):
            try:
                df_sector_old = pd.read_parquet(sector_path)
                df_sector_old['Date'] = pd.to_datetime(df_sector_old['Date'])
            except Exception as e:
                print(f"Failed to read existing SectorDailyBars.parquet: {e}")
                
        if os.path.exists(daily_path):
            try:
                df_daily_old = pd.read_parquet(daily_path)
                df_daily_old['Date'] = pd.to_datetime(df_daily_old['Date'])
            except Exception as e:
                print(f"Failed to read existing DailyBars.parquet: {e}")
                
        # Identify incremental vs bootstrap tickers
        cached_sector_tickers = set(df_sector_old['Ticker'].unique()) if not df_sector_old.empty else set()
        cached_stock_tickers = set(df_daily_old['Ticker'].unique()) if not df_daily_old.empty else set()
        
        tickers_sector_inc = [t for t in tickers if t in SECTOR_TICKERS and t in cached_sector_tickers]
        tickers_sector_boot = [t for t in tickers if t in SECTOR_TICKERS and t not in cached_sector_tickers]
        
        tickers_stock_inc = [t for t in tickers if t not in SECTOR_TICKERS and t in cached_stock_tickers]
        tickers_stock_boot = [t for t in tickers if t not in SECTOR_TICKERS and t not in cached_stock_tickers]
        
        # Batch download helper
        def download_batch(ticker_list: list, period: str) -> pd.DataFrame:
            if not ticker_list:
                return pd.DataFrame()
            print(f"Downloading batch of {len(ticker_list)} symbols with period='{period}'...")
            try:
                # yfinance download in batch
                return yf.download(ticker_list, interval="1d", period=period, group_by='ticker', progress=False)
            except Exception as e:
                print(f"Batch download failed for period '{period}': {e}")
                return pd.DataFrame()
                
        # Batch processing helper
        def process_batch(bulk_df: pd.DataFrame, ticker_list: list) -> list[pd.DataFrame]:
            rows_list = []
            if bulk_df is None or bulk_df.empty:
                return rows_list
                
            for ticker in ticker_list:
                try:
                    df = pd.DataFrame()
                    if isinstance(bulk_df.columns, pd.MultiIndex):
                        if ticker in bulk_df.columns.levels[0]:
                            df = bulk_df[ticker].dropna(how='all').reset_index()
                        else:
                            # Retry individually with period='5y' if missing from batch
                            print(f"Ticker {ticker} missing from batch. Retrying with period='5y'...")
                            try:
                                df_retry = yf.download(ticker, interval="1d", period="5y", progress=False)
                                if not df_retry.empty:
                                    if isinstance(df_retry.columns, pd.MultiIndex):
                                        df_retry.columns = df_retry.columns.get_level_values(0)
                                    df = df_retry.dropna(how='all').reset_index()
                            except Exception as re:
                                print(f"Retry failed for {ticker}: {re}")
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
                        
                    df_cleaned['Ticker'] = ticker
                    df_cleaned = df_cleaned[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    rows_list.append(df_cleaned)
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
            return rows_list

        # Execute Ingestion Batches
        sec_inc_rows = process_batch(download_batch(tickers_sector_inc, "7d"), tickers_sector_inc)
        sec_boot_rows = process_batch(download_batch(tickers_sector_boot, "max"), tickers_sector_boot)
        
        stk_inc_rows = process_batch(download_batch(tickers_stock_inc, "7d"), tickers_stock_inc)
        stk_boot_rows = process_batch(download_batch(tickers_stock_boot, "max"), tickers_stock_boot)
        
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=3)
        
        # Combine and compile Sector data
        new_sector_df = pd.DataFrame()
        all_sec_rows = sec_inc_rows + sec_boot_rows
        if all_sec_rows:
            new_sector_df = pd.concat(all_sec_rows, ignore_index=True)
            
        df_sector_final = merge_and_deduplicate(df_sector_old, new_sector_df)
        if not df_sector_final.empty:
            df_sector_final['Date_dt'] = pd.to_datetime(df_sector_final['Date'])
            df_sector_final = df_sector_final[df_sector_final['Date_dt'] >= cutoff_date].copy()
            df_sector_final = df_sector_final.drop(columns=['Date_dt'])
            df_sector_final.to_parquet(sector_path, index=False, engine='pyarrow', compression='zstd')
            print(f"Saved consolidated SectorDailyBars.parquet with {len(df_sector_final)} rows.")
            
        # Combine and compile Stock data
        new_stock_df = pd.DataFrame()
        all_stk_rows = stk_inc_rows + stk_boot_rows
        if all_stk_rows:
            new_stock_df = pd.concat(all_stk_rows, ignore_index=True)
            
        df_stock_final = merge_and_deduplicate(df_daily_old, new_stock_df)
        if not df_stock_final.empty:
            df_stock_final['Date_dt'] = pd.to_datetime(df_stock_final['Date'])
            df_stock_final = df_stock_final[df_stock_final['Date_dt'] >= cutoff_date].copy()
            df_stock_final = df_stock_final.drop(columns=['Date_dt'])
            
            df_stock_final.to_parquet(daily_path, index=False, engine='pyarrow', compression='zstd')
            print(f"Saved consolidated DailyBars.parquet with {len(df_stock_final)} rows.")
            
            # Resample weekly
            df_weekly = resample_weekly(df_stock_final)
            if not df_weekly.empty:
                df_weekly = df_weekly.sort_values(['Ticker', 'Date']).reset_index(drop=True)
                df_weekly.to_parquet(weekly_path, index=False, engine='pyarrow', compression='zstd')
                print(f"Saved consolidated WeeklyBars.parquet with {len(df_weekly)} rows.")
                
        # Regenerate stock metadata
        active_tickers = sorted(list(df_stock_final['Ticker'].unique())) if not df_stock_final.empty else []
        if active_tickers:
            metadata_rows = []
            for st_ticker in active_tickers:
                name = st_ticker.replace(".NS", "")
                metadata_rows.append({"Ticker": st_ticker, "Name": name, "Sector": "NSE"})
            df_meta = pd.DataFrame(metadata_rows)
            df_meta.to_parquet(meta_path, index=False, engine='pyarrow', compression='zstd')
            print(f"Saved consolidated StockMetadatas.parquet with {len(df_meta)} rows.")
            
    except Exception as e:
        print(f"Failed to run data ingestion: {e}")
