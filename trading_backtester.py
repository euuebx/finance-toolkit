"""
Algorithmic Trading Backtester
================================
Downloads real historical stock price data, implements a Moving Average
Crossover trading strategy, and backtests it — measuring how much money
the strategy would have made or lost historically.

This is what quantitative analysts (quants) do professionally:
test trading ideas against historical data before risking real money.

THE STRATEGY — Moving Average Crossover:
    - Calculate a FAST moving average (20 days) and SLOW moving average (50 days)
    - When FAST crosses ABOVE SLOW → BUY  (upward momentum signal)
    - When FAST crosses BELOW SLOW → SELL (downward momentum signal)
    - Hold position between signals

PERFORMANCE METRICS:
    - Total Return (%):   how much the strategy made overall
    - Sharpe Ratio:       return per unit of risk (higher = better)
    - Max Drawdown (%):   biggest peak-to-trough loss (lower = better)
    - Win Rate (%):       % of trades that were profitable

Libraries:
    - yfinance:    downloads real stock data from Yahoo Finance
    - pandas:      data manipulation and signal logic
    - numpy:       performance metric calculations
    - matplotlib:  visualising price, signals, and performance
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")


# ── 1. DOWNLOAD STOCK DATA ────────────────────────────────────────────────────

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical daily stock price data from Yahoo Finance.

    Args:
        ticker: Stock symbol e.g. "AAPL" (Apple), "MSFT" (Microsoft)
        start:  Start date as "YYYY-MM-DD"
        end:    End date as "YYYY-MM-DD"

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"  Downloading {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].copy()
    df.columns = ["Price"]
    df.dropna(inplace=True)

    print(f"  ✔ Downloaded {len(df)} trading days of data")
    return df


# ── 2. GENERATE TRADING SIGNALS ───────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """
    Calculate moving averages and generate BUY/SELL signals.

    Signal logic:
        +1 = BUY  (fast MA just crossed above slow MA)
        -1 = SELL (fast MA just crossed below slow MA)
         0 = HOLD (no crossover, stay in current position)

    Args:
        df:   DataFrame with Price column
        fast: Fast moving average window (days)
        slow: Slow moving average window (days)
    """
    df = df.copy()

    # Calculate moving averages
    df["MA_Fast"] = df["Price"].rolling(window=fast).mean()
    df["MA_Slow"] = df["Price"].rolling(window=slow).mean()

    # Position: 1 = holding (fast > slow), 0 = out of market (fast < slow)
    df["Position"] = np.where(df["MA_Fast"] > df["MA_Slow"], 1, 0)

    # Signal: difference in position day to day = crossover points
    # +1 = just entered (BUY), -1 = just exited (SELL)
    df["Signal"] = df["Position"].diff()

    return df.dropna()


# ── 3. RUN THE BACKTEST ───────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Simulate trading the strategy with real capital.

    How it works:
        - Start with initial_capital in cash
        - On a BUY signal:  spend all cash to buy shares at today's price
        - On a SELL signal: sell all shares, convert back to cash
        - Track portfolio value every day

    Args:
        df:              DataFrame with Price and Signal columns
        initial_capital: Starting cash amount (€)

    Returns:
        DataFrame with daily portfolio value and trade log
    """
    df = df.copy()

    cash        = initial_capital
    shares_held = 0.0
    portfolio_values = []
    trades      = []

    for i, (date, row) in enumerate(df.iterrows()):
        price = float(row["Price"])

        # BUY signal — spend all cash on shares
        if row["Signal"] == 1.0 and cash > 0:
            shares_held = cash / price
            cash        = 0
            trades.append({"Date": date, "Action": "BUY", "Price": price})

        # SELL signal — sell all shares
        elif row["Signal"] == -1.0 and shares_held > 0:
            cash        = shares_held * price
            shares_held = 0
            trades.append({"Date": date, "Action": "SELL", "Price": price})

        # Calculate current portfolio value
        portfolio_value = cash + (shares_held * price)
        portfolio_values.append(portfolio_value)

    df["Portfolio"] = portfolio_values

    # Buy and hold benchmark: just hold from day 1 to end
    df["BuyHold"] = initial_capital * (df["Price"] / df["Price"].iloc[0])

    trades_df = pd.DataFrame(trades)
    return df, trades_df


# ── 4. PERFORMANCE METRICS ────────────────────────────────────────────────────

def calculate_metrics(df: pd.DataFrame, initial_capital: float, trades_df: pd.DataFrame) -> dict:
    """
    Calculate professional trading performance metrics.

    SHARPE RATIO:
        Measures return per unit of risk.
        Formula: (mean daily return - risk free rate) / std of daily returns × √252
        > 1.0 is considered good, > 2.0 is excellent
        Named after Nobel laureate William Sharpe.

    MAX DRAWDOWN:
        The biggest percentage drop from a peak before a new peak is reached.
        If your portfolio hit €15,000 then fell to €9,000, drawdown = -40%.
        Key risk metric — shows worst case loss scenario.

    WIN RATE:
        % of completed trades (buy→sell pairs) that were profitable.
    """
    portfolio = df["Portfolio"]
    buyhold   = df["BuyHold"]

    # Total returns
    strategy_return = ((portfolio.iloc[-1] - initial_capital) / initial_capital) * 100
    buyhold_return  = ((buyhold.iloc[-1]   - initial_capital) / initial_capital) * 100

    # Daily returns for Sharpe calculation
    daily_returns = portfolio.pct_change().dropna()

    # Sharpe Ratio (annualised, assuming 0% risk-free rate for simplicity)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    # Maximum Drawdown
    rolling_max  = portfolio.cummax()
    drawdowns    = (portfolio - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()

    # Win rate from trades
    win_rate = None
    if len(trades_df) >= 2:
        buys  = trades_df[trades_df["Action"] == "BUY"]["Price"].values
        sells = trades_df[trades_df["Action"] == "SELL"]["Price"].values
        pairs = min(len(buys), len(sells))
        if pairs > 0:
            wins     = sum(sells[:pairs] > buys[:pairs])
            win_rate = round((wins / pairs) * 100, 1)

    return {
        "Initial Capital":         initial_capital,
        "Final Value (Strategy)":  round(portfolio.iloc[-1], 2),
        "Final Value (Buy & Hold)": round(buyhold.iloc[-1], 2),
        "Strategy Return (%)":     round(strategy_return, 2),
        "Buy & Hold Return (%)":   round(buyhold_return, 2),
        "Sharpe Ratio":            round(sharpe, 3),
        "Max Drawdown (%)":        round(max_drawdown, 2),
        "Total Trades":            len(trades_df),
        "Win Rate (%)":            win_rate if win_rate is not None else "N/A",
    }


# ── 5. VISUALISATIONS ─────────────────────────────────────────────────────────

def plot_strategy(df: pd.DataFrame, trades_df: pd.DataFrame, ticker: str):
    """
    Main strategy chart: price, moving averages, and buy/sell signals.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # ── Top panel: Price + MAs + signals ──
    ax1.plot(df.index, df["Price"],   color="#94A3B8", linewidth=1,   label="Price", alpha=0.9)
    ax1.plot(df.index, df["MA_Fast"], color="#2563EB", linewidth=1.8, label="20-Day MA (Fast)")
    ax1.plot(df.index, df["MA_Slow"], color="#DC2626", linewidth=1.8, label="50-Day MA (Slow)")

    # Plot BUY signals (green triangles pointing up)
    buys = trades_df[trades_df["Action"] == "BUY"]
    ax1.scatter(buys["Date"], buys["Price"], marker="^", color="#16A34A",
                s=120, zorder=5, label="BUY Signal")

    # Plot SELL signals (red triangles pointing down)
    sells = trades_df[trades_df["Action"] == "SELL"]
    ax1.scatter(sells["Date"], sells["Price"], marker="v", color="#DC2626",
                s=120, zorder=5, label="SELL Signal")

    ax1.set_title(f"{ticker} — Moving Average Crossover Strategy (20/50 Day)",
                  fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # ── Bottom panel: Portfolio value vs Buy & Hold ──
    ax2.plot(df.index, df["Portfolio"], color="#16A34A", linewidth=2, label="Strategy")
    ax2.plot(df.index, df["BuyHold"],   color="#2563EB", linewidth=2,
             linestyle="--", label="Buy & Hold")
    ax2.set_ylabel("Portfolio (€)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("strategy_chart.png", dpi=150)
    plt.show()
    print("  ✔ Saved: strategy_chart.png")


def plot_drawdown(df: pd.DataFrame, ticker: str):
    """
    Drawdown chart — shows how far the portfolio fell from its peak at each point.
    """
    rolling_max = df["Portfolio"].cummax()
    drawdown    = (df["Portfolio"] - rolling_max) / rolling_max * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(df.index, drawdown, 0, color="#DC2626", alpha=0.5, label="Drawdown")
    ax.plot(df.index, drawdown, color="#DC2626", linewidth=1)

    ax.set_title(f"{ticker} Strategy — Drawdown Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("drawdown_chart.png", dpi=150)
    plt.show()
    print("  ✔ Saved: drawdown_chart.png")


def plot_monthly_returns(df: pd.DataFrame, ticker: str):
    """
    Bar chart of monthly returns — shows which months were profitable.
    """
    monthly = df["Portfolio"].resample("ME").last().pct_change().dropna() * 100

    colours = ["#16A34A" if r >= 0 else "#DC2626" for r in monthly]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly.index, monthly.values, color=colours, alpha=0.85, width=20)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title(f"{ticker} Strategy — Monthly Returns (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Return (%)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("monthly_returns.png", dpi=150)
    plt.show()
    print("  ✔ Saved: monthly_returns.png")


# ── 6. MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("   📈  Algorithmic Trading Backtester")
    print("=" * 58)
    print("\nImplements a Moving Average Crossover strategy and")
    print("tests it against real historical stock price data.\n")

    # ── Get inputs ──
    ticker  = input("Stock ticker [e.g. AAPL, MSFT, TSLA]: ").upper().strip()
    start   = input("Start date [YYYY-MM-DD, e.g. 2020-01-01]: ").strip()
    end     = input("End date   [YYYY-MM-DD, e.g. 2024-01-01]: ").strip()
    capital = float(input("Starting capital (€) [e.g. 10000]: "))

    # ── Download data ──
    df = download_data(ticker, start, end)

    # ── Generate signals ──
    print("\n  Calculating moving averages and signals...")
    df = generate_signals(df, fast=20, slow=50)

    # ── Run backtest ──
    print("  Running backtest...")
    df, trades_df = run_backtest(df, capital)

    # ── Print trade log ──
    if not trades_df.empty:
        print(f"\n  TRADE LOG ({len(trades_df)} trades)")
        print("─" * 45)
        for _, trade in trades_df.iterrows():
            action = trade["Action"]
            emoji  = "🟢" if action == "BUY" else "🔴"
            print(f"  {emoji} {action}  |  {str(trade['Date'])[:10]}  |  ${trade['Price']:.2f}")

    # ── Print performance metrics ──
    metrics = calculate_metrics(df, capital, trades_df)
    print("\n" + "─" * 58)
    print("  PERFORMANCE METRICS")
    print("─" * 58)
    for key, value in metrics.items():
        if isinstance(value, float):
            if "%" in key:
                print(f"  {key:<35} {value:>8.2f}%")
            else:
                print(f"  {key:<35} €{value:>10,.2f}")
        else:
            print(f"  {key:<35} {value}")

    # ── Save results ──
    df.to_csv(f"{ticker}_backtest_results.csv")
    trades_df.to_csv(f"{ticker}_trades.csv", index=False)
    print(f"\n  ✔ Results saved: {ticker}_backtest_results.csv")
    print(f"  ✔ Trades saved:  {ticker}_trades.csv")

    # ── Generate charts ──
    print("\n  Generating charts...")
    plot_strategy(df, trades_df, ticker)
    plot_drawdown(df, ticker)
    plot_monthly_returns(df, ticker)

    print("\n✅ Backtest complete!\n")


if __name__ == "__main__":
    main()
