from src.data.loader import download_ticker_data
from src.data.preprocessing import combine_returns, save_processed_data
from src.data.validation import generate_quality_report

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "GLD"]


def main():

    data = {}

    for ticker in TICKERS:

        print(f"Downloading {ticker}")

        df = download_ticker_data(ticker)

        data[ticker] = df

    returns_df = combine_returns(data)

    save_processed_data(returns_df, "portfolio_returns.csv")

    report = generate_quality_report(returns_df)

    with open("data/data_quality_report.txt", "w") as f:
        f.write(report)

    print("Pipeline complete")


if __name__ == "__main__":
    main()
