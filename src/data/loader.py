import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")


def download_ticker_data(
    ticker: str, start: str = "2015-01-01", end: str = None
) -> pd.DataFrame:

    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DATA_DIR / f"{ticker}.csv"

    df.to_csv(output_path)

    return df
