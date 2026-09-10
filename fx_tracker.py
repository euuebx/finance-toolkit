"""
Currency Exchange Rate Tracker
Fetches live and historical EUR/USD exchange rates from a free public API,
tracks trends over time, calculates key statistics, and lets you convert
between currencies using the latest available rate.

Libraries:
- requests:    fetching live data from the internet
- pandas:      organising and analysing the rate data
- matplotlib:  plotting exchange rate charts
- datetime:    handling dates for historical data
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from datetime import datetime, timedelta


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# We use the free frankfurter.app API — no API key needed
# Documentation: https://www.frankfurter.app/docs
BASE_URL = "https://api.frankfurter.app"


# ── 1. FETCHING DATA FROM THE API ─────────────────────────────────────────────

def get_latest_rate(base: str = "EUR", target: str = "USD") -> dict:
    """
    Fetch the latest exchange rate between two currencies.

    How it works:
        We send a GET request to the API URL.
        The API responds with JSON data containing the current rate.
        We parse that JSON and return the rate.

    Args:
        base:   The currency you're converting FROM (e.g. "EUR")
        target: The currency you're converting TO   (e.g. "USD")

    Returns:
        dict with rate info
    """
    url = f"{BASE_URL}/latest?from={base}&to={target}"

    print(f"  Fetching latest {base}/{target} rate...")
    response = requests.get(url, timeout=10)

    # Check the request succeeded (HTTP 200 = OK)
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}")

    data = response.json()

    return {
        "base":   base,
        "target": target,
        "rate":   data["rates"][target],
        "date":   data["date"],
    }


def get_historical_rates(
    base: str = "EUR",
    target: str = "USD",
    days: int = 90,
) -> pd.DataFrame:
    """
    Fetch historical exchange rates for the past N days.

    The frankfurter API lets us request a date range like:
        /YYYY-MM-DD..YYYY-MM-DD?from=EUR&to=USD

    Args:
        base:   Currency converting from
        target: Currency converting to
        days:   How many days of history to fetch

    Returns:
        DataFrame with Date and Rate columns
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    url = f"{BASE_URL}/{start_str}..{end_str}?from={base}&to={target}"

    print(f"  Fetching {days} days of {base}/{target} history...")
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}")

    data = response.json()

    # Convert the nested JSON into a clean DataFrame
    records = [
        {"Date": datetime.strptime(date, "%Y-%m-%d"), "Rate": rates[target]}
        for date, rates in data["rates"].items()
    ]

    df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    return df


# ── 2. STATISTICS ─────────────────────────────────────────────────────────────

def rate_statistics(df: pd.DataFrame, base: str, target: str) -> dict:
    """
    Calculate key descriptive statistics for the exchange rate history.
    These are the same stats you'd calculate in any statistics module.
    """
    rates = df["Rate"]
    return {
        "Currency Pair":   f"{base}/{target}",
        "Period":          f"{df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}",
        "Data Points":     len(df),
        "Current Rate":    round(rates.iloc[-1], 4),
        "Mean Rate":       round(rates.mean(), 4),
        "Median Rate":     round(rates.median(), 4),
        "Std Deviation":   round(rates.std(), 4),
        "Min Rate":        round(rates.min(), 4),
        "Max Rate":        round(rates.max(), 4),
        "Range":           round(rates.max() - rates.min(), 4),
        "% Change":        round(((rates.iloc[-1] - rates.iloc[0]) / rates.iloc[0]) * 100, 2),
    }


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7-day and 30-day moving averages to the DataFrame.

    A moving average smooths out short-term noise to reveal the trend.
    At each point, it averages the last N days of rates.
    Widely used in financial analysis and algorithmic trading.
    """
    df = df.copy()
    df["MA7"]  = df["Rate"].rolling(window=7).mean()   # 7-day moving average
    df["MA30"] = df["Rate"].rolling(window=30).mean()  # 30-day moving average
    return df


# ── 3. CURRENCY CONVERTER ─────────────────────────────────────────────────────

def convert_currency(amount: float, rate: float, base: str, target: str) -> None:
    """Convert an amount using the live rate and print the result."""
    converted = amount * rate
    print(f"\n  💱 {amount:,.2f} {base} = {converted:,.2f} {target}")
    print(f"     (Rate used: 1 {base} = {rate:.4f} {target})")


# ── 4. VISUALISATIONS ─────────────────────────────────────────────────────────

def plot_rate_history(df: pd.DataFrame, base: str, target: str):
    """
    Line chart of exchange rate over time with moving averages.
    Moving averages help identify whether the trend is up or down.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["Date"], df["Rate"], color="#94A3B8", linewidth=1,
            alpha=0.8, label=f"{base}/{target} Daily Rate")

    if "MA7" in df.columns:
        ax.plot(df["Date"], df["MA7"],  color="#2563EB", linewidth=2,
                label="7-Day Moving Average")
    if "MA30" in df.columns:
        ax.plot(df["Date"], df["MA30"], color="#DC2626", linewidth=2,
                linestyle="--", label="30-Day Moving Average")

    ax.set_title(f"{base}/{target} Exchange Rate — Last {len(df)} Trading Days",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(f"Rate (1 {base} = X {target})", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("rate_history.png", dpi=150)
    plt.show()
    print("  ✔ Saved: rate_history.png")


def plot_rate_distribution(df: pd.DataFrame, base: str, target: str):
    """
    Histogram showing the distribution of daily exchange rates.
    Shows whether rates cluster tightly or vary widely.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    current = df["Rate"].iloc[-1]
    mean    = df["Rate"].mean()

    ax.hist(df["Rate"], bins=30, color="#2563EB", alpha=0.75, edgecolor="white")
    ax.axvline(current, color="#16A34A", linewidth=2,
               linestyle="--", label=f"Current: {current:.4f}")
    ax.axvline(mean,    color="#DC2626", linewidth=2,
               linestyle="--", label=f"Mean: {mean:.4f}")

    ax.set_title(f"{base}/{target} Rate Distribution",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Rate (1 {base} = X {target})", fontsize=12)
    ax.set_ylabel("Frequency (days)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("rate_distribution.png", dpi=150)
    plt.show()
    print("  ✔ Saved: rate_distribution.png")


def plot_daily_changes(df: pd.DataFrame, base: str, target: str):
    """
    Bar chart of daily rate changes (positive = rate went up, negative = down).
    Shows volatility and direction of movement day to day.
    """
    df = df.copy()
    df["Change"] = df["Rate"].diff()  # difference from previous day

    fig, ax = plt.subplots(figsize=(12, 4))

    colours = ["#16A34A" if c >= 0 else "#DC2626" for c in df["Change"]]
    ax.bar(df["Date"], df["Change"], color=colours, alpha=0.8, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title(f"{base}/{target} Daily Rate Changes",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Change in Rate", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("daily_changes.png", dpi=150)
    plt.show()
    print("  ✔ Saved: daily_changes.png")


# ── 5. MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   💱  Currency Exchange Rate Tracker")
    print("=" * 55)

    base   = "EUR"
    target = "USD"

    # ── Fetch latest rate ──
    latest = get_latest_rate(base, target)
    print(f"\n  Latest Rate ({latest['date']}): "
          f"1 {base} = {latest['rate']:.4f} {target}")

    # ── Currency converter ──
    print("\n" + "─" * 55)
    amount = float(input(f"\nEnter amount to convert ({base} → {target}): "))
    convert_currency(amount, latest["rate"], base, target)

    # ── Fetch historical data ──
    days = int(input("\nHow many days of history to analyse? [e.g. 90]: "))
    df   = get_historical_rates(base, target, days)
    df   = add_moving_averages(df)

    # ── Print statistics ──
    stats = rate_statistics(df, base, target)
    print("\n" + "─" * 55)
    print("  RATE STATISTICS")
    print("─" * 55)
    for key, value in stats.items():
        print(f"  {key:<20} {value}")

    # ── Save data ──
    df.to_csv("eur_usd_history.csv", index=False)
    print("\n  ✔ Historical data saved: eur_usd_history.csv")

    # ── Generate charts ──
    print("\n  Generating charts...")
    plot_rate_history(df, base, target)
    plot_rate_distribution(df, base, target)
    plot_daily_changes(df, base, target)

    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()
