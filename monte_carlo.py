"""
Monte Carlo Portfolio Simulation
==================================
Simulates thousands of possible portfolio outcomes using random sampling.
Instead of predicting ONE future, we model THOUSANDS of possible futures
based on historical market behaviour (average return + volatility).

The result: a probability distribution of outcomes — showing best case,
worst case, and the most likely range for a given investment.

This technique is used by banks, hedge funds, and risk analysts daily.

Libraries:
- numpy:      fast random number generation and maths
- matplotlib: plotting all simulation paths and distributions
- pandas:     organising results into a summary table
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


# ── 1. THE SIMULATION ────────────────────────────────────────────────────────

def run_simulation(
    initial_investment: float,
    annual_return: float,
    annual_volatility: float,
    years: int,
    num_simulations: int = 1000,
) -> np.ndarray:
    """
    Run Monte Carlo simulation of portfolio growth.

    How it works:
    - Each simulation is one possible future for your portfolio
    - Every year, we pick a random return from a normal distribution
      centred on the expected annual return, with spread = volatility
    - We do this thousands of times to see the full range of outcomes

    Args:
        initial_investment: Starting portfolio value (€)
        annual_return:      Expected yearly return, e.g. 7 for 7% (%)
        annual_volatility:  How much returns vary year to year, e.g. 15 for 15% (%)
        years:              How many years to simulate
        num_simulations:    How many different futures to model (default 1000)

    Returns:
        2D numpy array of shape (years, num_simulations)
        Each column is one simulated portfolio path
    """
    # Convert percentages to decimals
    mu    = annual_return / 100        # expected annual return
    sigma = annual_volatility / 100    # annual volatility (standard deviation)

    # Array to store all simulation results
    # Rows = years, Columns = each simulation
    portfolio = np.zeros((years, num_simulations))

    for sim in range(num_simulations):
        balance = initial_investment

        for year in range(years):
            # Draw a random annual return from a normal distribution
            # np.random.normal(mean, std_dev) — this is the core of Monte Carlo
            random_return = np.random.normal(mu, sigma)

            # Grow (or shrink) the portfolio by that return
            balance = balance * (1 + random_return)
            balance = max(balance, 0)  # portfolio can't go below zero

            portfolio[year, sim] = balance

    return portfolio


# ── 2. SUMMARY STATISTICS ────────────────────────────────────────────────────

def simulation_summary(portfolio: np.ndarray, initial_investment: float) -> pd.DataFrame:
    """
    Extract key statistics from all simulations at each year.

    Percentiles tell us:
    - 10th percentile: 90% of simulations did BETTER than this (bad scenario)
    - 50th percentile: the median outcome (middle scenario)
    - 90th percentile: only 10% of simulations did better (good scenario)
    """
    years = portfolio.shape[0]
    records = []

    for year in range(years):
        year_values = portfolio[year, :]
        records.append({
            "Year":          year + 1,
            "10th %ile (€)": round(np.percentile(year_values, 10), 2),
            "25th %ile (€)": round(np.percentile(year_values, 25), 2),
            "Median (€)":    round(np.percentile(year_values, 50), 2),
            "75th %ile (€)": round(np.percentile(year_values, 75), 2),
            "90th %ile (€)": round(np.percentile(year_values, 90), 2),
            "Mean (€)":      round(np.mean(year_values), 2),
        })

    return pd.DataFrame(records)


def final_value_stats(portfolio: np.ndarray, initial_investment: float) -> dict:
    """Key stats for the final year only."""
    final_values = portfolio[-1, :]
    return {
        "Initial Investment":  initial_investment,
        "Simulations Run":     portfolio.shape[1],
        "Worst Case (5th %)":  round(np.percentile(final_values, 5), 2),
        "Bad Case (10th %)":   round(np.percentile(final_values, 10), 2),
        "Median Outcome":      round(np.percentile(final_values, 50), 2),
        "Good Case (90th %)":  round(np.percentile(final_values, 90), 2),
        "Best Case (95th %)":  round(np.percentile(final_values, 95), 2),
        "Mean Outcome":        round(np.mean(final_values), 2),
        "Prob. of Profit (%)": round((final_values > initial_investment).mean() * 100, 1),
        "Prob. of Loss (%)":   round((final_values < initial_investment).mean() * 100, 1),
    }


# ── 3. VISUALISATIONS ────────────────────────────────────────────────────────

def plot_simulation_paths(
    portfolio: np.ndarray,
    initial_investment: float,
    annual_return: float,
    annual_volatility: float,
    years: int,
    num_simulations: int,
):
    """
    Spaghetti chart: plot all simulation paths.
    Shows the full range of possible futures visually.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    year_axis = np.arange(1, years + 1)

    # Plot all simulation paths in light grey (too many to show individually)
    for sim in range(min(num_simulations, 300)):  # cap at 300 for performance
        ax.plot(year_axis, portfolio[:, sim], color="steelblue", alpha=0.05, linewidth=0.8)

    # Highlight key percentiles
    ax.plot(year_axis, np.percentile(portfolio, 10, axis=1), color="#DC2626",
            linewidth=2, linestyle="--", label="10th percentile (bad)")
    ax.plot(year_axis, np.percentile(portfolio, 50, axis=1), color="#16A34A",
            linewidth=2.5, label="Median outcome")
    ax.plot(year_axis, np.percentile(portfolio, 90, axis=1), color="#2563EB",
            linewidth=2, linestyle="--", label="90th percentile (good)")

    # Reference line: initial investment
    ax.axhline(y=initial_investment, color="orange", linewidth=1.5,
               linestyle=":", label="Initial investment")

    ax.set_title(f"Monte Carlo Simulation — {num_simulations:,} Portfolio Paths\n"
                 f"€{initial_investment:,.0f} | {annual_return}% expected return | "
                 f"{annual_volatility}% volatility | {years} years",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Portfolio Value (€)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("simulation_paths.png", dpi=150)
    plt.show()
    print("  ✔ Saved: simulation_paths.png")


def plot_final_distribution(portfolio: np.ndarray, initial_investment: float, years: int):
    """
    Histogram of final portfolio values across all simulations.
    Shows the probability distribution of outcomes.
    """
    final_values = portfolio[-1, :]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Split into profit (green) and loss (red) bars
    profits = final_values[final_values >= initial_investment]
    losses  = final_values[final_values <  initial_investment]

    ax.hist(profits, bins=60, color="#16A34A", alpha=0.7, label="Profit")
    ax.hist(losses,  bins=60, color="#DC2626", alpha=0.7, label="Loss")

    # Mark key percentiles
    for pct, label, colour in [
        (10, "10th %ile", "#DC2626"),
        (50, "Median",    "#16A34A"),
        (90, "90th %ile", "#2563EB"),
    ]:
        val = np.percentile(final_values, pct)
        ax.axvline(val, color=colour, linewidth=2, linestyle="--",
                   label=f"{label}: €{val:,.0f}")

    ax.axvline(initial_investment, color="orange", linewidth=2,
               linestyle=":", label=f"Initial: €{initial_investment:,.0f}")

    ax.set_title(f"Distribution of Final Portfolio Values after {years} Years",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Final Portfolio Value (€)", fontsize=12)
    ax.set_ylabel("Number of Simulations", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("final_distribution.png", dpi=150)
    plt.show()
    print("  ✔ Saved: final_distribution.png")


def plot_percentile_bands(portfolio: np.ndarray, initial_investment: float, years: int):
    """
    Shaded band chart showing confidence intervals over time.
    The darker the band, the more likely the outcome falls there.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    year_axis = np.arange(1, years + 1)

    p10 = np.percentile(portfolio, 10, axis=1)
    p25 = np.percentile(portfolio, 25, axis=1)
    p50 = np.percentile(portfolio, 50, axis=1)
    p75 = np.percentile(portfolio, 75, axis=1)
    p90 = np.percentile(portfolio, 90, axis=1)

    ax.fill_between(year_axis, p10, p90, alpha=0.15, color="#2563EB", label="10th–90th %ile")
    ax.fill_between(year_axis, p25, p75, alpha=0.30, color="#2563EB", label="25th–75th %ile")
    ax.plot(year_axis, p50, color="#2563EB", linewidth=2.5, label="Median")
    ax.axhline(y=initial_investment, color="orange", linewidth=1.5,
               linestyle=":", label="Initial investment")

    ax.set_title(f"Portfolio Confidence Intervals Over {years} Years\n"
                 f"Starting investment: €{initial_investment:,.0f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Portfolio Value (€)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("percentile_bands.png", dpi=150)
    plt.show()
    print("  ✔ Saved: percentile_bands.png")


# ── 4. MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   🎲  Monte Carlo Portfolio Simulation")
    print("=" * 55)
    print("\nThis tool simulates thousands of possible futures for")
    print("your portfolio based on expected return and volatility.\n")

    # ── Get inputs ──
    initial_investment = float(input("Initial investment (€): "))
    annual_return      = float(input("Expected annual return (%) [e.g. 7]: "))
    annual_volatility  = float(input("Annual volatility / risk (%) [e.g. 15]: "))
    years              = int(input("Investment horizon (years): "))
    num_simulations    = int(input("Number of simulations [e.g. 1000]: "))

    print(f"\n  Running {num_simulations:,} simulations over {years} years...")

    # ── Run simulation ──
    np.random.seed(42)  # makes results reproducible
    portfolio = run_simulation(
        initial_investment, annual_return, annual_volatility,
        years, num_simulations
    )

    # ── Print results ──
    stats = final_value_stats(portfolio, initial_investment)

    print("\n" + "─" * 55)
    print("  FINAL YEAR RESULTS")
    print("─" * 55)
    for key, value in stats.items():
        if "%" in key and "ile" not in key:
            print(f"  {key:<30} {value:>10.1f}%")
        elif isinstance(value, float):
            print(f"  {key:<30} €{value:>12,.2f}")
        else:
            print(f"  {key:<30} {value:>13,}")

    # ── Generate charts ──
    print("\n  Generating charts...")
    plot_simulation_paths(portfolio, initial_investment, annual_return,
                          annual_volatility, years, num_simulations)
    plot_final_distribution(portfolio, initial_investment, years)
    plot_percentile_bands(portfolio, initial_investment, years)

    print("\n✅ Done! Check your project folder for the charts.\n")


if __name__ == "__main__":
    main()
