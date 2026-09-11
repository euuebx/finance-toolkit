# Algorithmic Trading Backtester
# 20/50 day moving average trading strategy

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Get the stock data
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end,
                       auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError("No stock data found. Check the ticker and dates.")

    # yfinance can sometimes return multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Close"]].copy()
    data.columns = ["Price"]
    return data.dropna()


# Create the moving averages and buy/sell signals
def make_signals(data):
    data = data.copy()

    data["MA20"] = data["Price"].rolling(20).mean()
    data["MA50"] = data["Price"].rolling(50).mean()

    # 1 means we are holding the stock, 0 means we are in cash
    data["Position"] = np.where(data["MA20"] > data["MA50"], 1, 0)

    # A change from 0 -> 1 is a buy, 1 -> 0 is a sell
    data["Signal"] = data["Position"].diff()

    return data.dropna()


# Run the trading strategy
def backtest(data, starting_money):
    data = data.copy()

    cash = starting_money
    shares = 0
    values = []
    trades = []

    for date, row in data.iterrows():
        price = float(row["Price"])
        signal = row["Signal"]

        if signal == 1 and cash > 0:
            shares = cash / price
            cash = 0
            trades.append([date, "BUY", price])

        elif signal == -1 and shares > 0:
            cash = shares * price
            shares = 0
            trades.append([date, "SELL", price])

        # Value of cash + stock currently held
        values.append(cash + shares * price)

    data["Portfolio"] = values

    # Simple buy-and-hold comparison
    data["BuyHold"] = starting_money * (
        data["Price"] / data["Price"].iloc[0]
    )

    trades = pd.DataFrame(trades, columns=["Date", "Action", "Price"])
    return data, trades


# Calculate a few useful performance measures
def get_metrics(data, starting_money, trades):
    portfolio = data["Portfolio"]

    strategy_return = (portfolio.iloc[-1] / starting_money - 1) * 100
    buyhold_return = (data["BuyHold"].iloc[-1] / starting_money - 1) * 100

    daily_returns = portfolio.pct_change().dropna()

    if daily_returns.std() != 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    peak = portfolio.cummax()
    drawdown = (portfolio - peak) / peak * 100
    max_drawdown = drawdown.min()

    # Work out how many completed buy -> sell trades made money
    buys = trades[trades["Action"] == "BUY"]["Price"].tolist()
    sells = trades[trades["Action"] == "SELL"]["Price"].tolist()

    pairs = min(len(buys), len(sells))

    if pairs > 0:
        wins = sum(sells[i] > buys[i] for i in range(pairs))
        win_rate = wins / pairs * 100
    else:
        win_rate = None

    return {
        "Starting Capital": starting_money,
        "Final Strategy Value": round(portfolio.iloc[-1], 2),
        "Final Buy & Hold Value": round(data["BuyHold"].iloc[-1], 2),
        "Strategy Return (%)": round(strategy_return, 2),
        "Buy & Hold Return (%)": round(buyhold_return, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Number of Trades": len(trades),
        "Win Rate (%)": round(win_rate, 1) if win_rate is not None else "N/A"
    }


# Make a simple chart
def show_chart(data, trades, ticker):
    plt.figure(figsize=(12, 6))

    plt.plot(data.index, data["Price"], label="Price")
    plt.plot(data.index, data["MA20"], label="20 Day MA")
    plt.plot(data.index, data["MA50"], label="50 Day MA")

    buys = trades[trades["Action"] == "BUY"]
    sells = trades[trades["Action"] == "SELL"]

    if not buys.empty:
        plt.scatter(buys["Date"], buys["Price"],
                    marker="^", s=80, label="Buy")

    if not sells.empty:
        plt.scatter(sells["Date"], sells["Price"],
                    marker="v", s=80, label="Sell")

    plt.title(ticker + " Moving Average Strategy")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("strategy_chart.png", dpi=150)
    plt.show()


def main():
    print("Algorithmic Trading Backtester")
    print("-" * 35)

    ticker = input("Stock ticker (e.g. AAPL): ").upper().strip()
    start = input("Start date (YYYY-MM-DD): ").strip()
    end = input("End date (YYYY-MM-DD): ").strip()

    try:
        starting_money = float(input("Starting capital (€): "))
    except ValueError:
        print("Please enter a valid amount of money.")
        return

    print("\nDownloading data...")
    data = get_data(ticker, start, end)

    print("Creating signals...")
    data = make_signals(data)

    print("Running backtest...")
    data, trades = backtest(data, starting_money)

    # Display trades
    print("\nTrades:")
    if trades.empty:
        print("No trades were made.")
    else:
        for _, trade in trades.iterrows():
            print(
                f"{trade['Action']} | "
                f"{str(trade['Date'])[:10]} | "
                f"${trade['Price']:.2f}"
            )

    # Display results
    results = get_metrics(data, starting_money, trades)

    print("\nPerformance")
    print("-" * 35)

    for name, value in results.items():
        if isinstance(value, float):
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value}")

    # Save the data
    data.to_csv(f"{ticker}_backtest_results.csv")
    trades.to_csv(f"{ticker}_trades.csv", index=False)

    print("\nFiles saved:")
    print(f"- {ticker}_backtest_results.csv")
    print(f"- {ticker}_trades.csv")

    show_chart(data, trades, ticker)


if __name__ == "__main__":
    main()
