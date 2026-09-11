# Currency Exchange Rate Tracker
# Tracks EUR/USD rates and does some basic analysis

import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

BASE_URL = "https://api.frankfurter.app"


# Get the latest exchange rate
def get_latest_rate(base="EUR", target="USD"):
    url = f"{BASE_URL}/latest?from={base}&to={target}"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise Exception("Could not get exchange rate data")

    data = response.json()

    return {
        "base": base,
        "target": target,
        "rate": data["rates"][target],
        "date": data["date"]
    }


# Get exchange rates for a number of previous days
def get_history(base="EUR", target="USD", days=90):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")

    url = f"{BASE_URL}/{start}..{end}?from={base}&to={target}"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise Exception("Could not get historical data")

    data = response.json()

    rows = []

    for date, rates in data["rates"].items():
        rows.append({
            "Date": datetime.strptime(date, "%Y-%m-%d"),
            "Rate": rates[target]
        })

    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


# Calculate some basic statistics
def get_statistics(df, base, target):
    rates = df["Rate"]

    return {
        "Currency Pair": f"{base}/{target}",
        "Data Points": len(df),
        "Current Rate": round(rates.iloc[-1], 4),
        "Mean Rate": round(rates.mean(), 4),
        "Median Rate": round(rates.median(), 4),
        "Standard Deviation": round(rates.std(), 4),
        "Minimum": round(rates.min(), 4),
        "Maximum": round(rates.max(), 4),
        "Range": round(rates.max() - rates.min(), 4),
        "Percentage Change": round(
            (rates.iloc[-1] - rates.iloc[0]) / rates.iloc[0] * 100, 2
        )
    }


# Add moving averages to see the general trend
def add_moving_averages(df):
    df = df.copy()
    df["MA7"] = df["Rate"].rolling(7).mean()
    df["MA30"] = df["Rate"].rolling(30).mean()
    return df


# Convert money using the latest rate
def convert(amount, rate, base, target):
    result = amount * rate
    print(f"\n{amount:,.2f} {base} = {result:,.2f} {target}")
    print(f"Rate used: 1 {base} = {rate:.4f} {target}")


# Plot the exchange rate and moving averages
def plot_history(df, base, target):
    plt.figure(figsize=(12, 5))

    plt.plot(df["Date"], df["Rate"], label=f"{base}/{target}")
    plt.plot(df["Date"], df["MA7"], label="7 Day MA")
    plt.plot(df["Date"], df["MA30"], label="30 Day MA")

    plt.title(f"{base}/{target} Exchange Rate")
    plt.xlabel("Date")
    plt.ylabel(f"1 {base} in {target}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("rate_history.png", dpi=150)
    plt.show()


# Show how the rates are distributed
def plot_distribution(df):
    plt.figure(figsize=(9, 5))

    plt.hist(df["Rate"], bins=20, edgecolor="black")

    plt.axvline(df["Rate"].iloc[-1], linestyle="--",
                label="Current Rate")
    plt.axvline(df["Rate"].mean(), linestyle="--",
                label="Mean")

    plt.title("EUR/USD Rate Distribution")
    plt.xlabel("Exchange Rate")
    plt.ylabel("Number of Days")
    plt.legend()
    plt.tight_layout()

    plt.savefig("rate_distribution.png", dpi=150)
    plt.show()


# Show the change in the exchange rate each day
def plot_daily_changes(df):
    changes = df["Rate"].diff()

    plt.figure(figsize=(12, 4))
    plt.bar(df["Date"], changes)

    plt.axhline(0, linewidth=0.8)
    plt.title("Daily EUR/USD Rate Changes")
    plt.xlabel("Date")
    plt.ylabel("Change")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("daily_changes.png", dpi=150)
    plt.show()


def main():
    print("Currency Exchange Rate Tracker")
    print("-" * 35)

    base = "EUR"
    target = "USD"

    latest = get_latest_rate(base, target)

    print(
        f"\nLatest rate ({latest['date']}): "
        f"1 {base} = {latest['rate']:.4f} {target}"
    )

    try:
        amount = float(input(f"\nEnter amount to convert ({base} to {target}): "))
        days = int(input("How many days of history should be analysed? "))

    except ValueError:
        print("Please enter valid numbers.")
        return

    convert(amount, latest["rate"], base, target)

    print("\nGetting historical data...")
    df = get_history(base, target, days)

    df = add_moving_averages(df)

    stats = get_statistics(df, base, target)

    print("\nStatistics")
    print("-" * 35)

    for key, value in stats.items():
        print(f"{key}: {value}")

    # Save the data for later use
    df.to_csv("eur_usd_history.csv", index=False)
    print("\nHistorical data saved to eur_usd_history.csv")

    print("Creating charts...")
    plot_history(df, base, target)
    plot_distribution(df)
    plot_daily_changes(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
