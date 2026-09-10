"""
Compares multiple investment scenarios side by side showing how
starting amount, regular contributions, interest rate, and time
interact to grow wealth over time.

The core idea: compound interest means you earn interest on your
interest, which causes exponential rather than linear growth.

Libraries used:
- matplotlib: for all charts
- pandas: for organising scenario data into tables
- numpy: for efficient numerical calculations
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np


# ── 1. CORE FORMULA ──────────────────────────────────────────────────────────

def compound_growth(
    principal: float,
    annual_rate: float,
    years: int,
    monthly_contribution: float = 0.0,
    compounds_per_year: int = 12,
) -> pd.DataFrame:
    """
    Calculate portfolio value year by year.

    Formula (lump sum only):
        A = P × (1 + r/n)^(n×t)

    With monthly contributions, each contribution also compounds
    from the point it's added — so we build this month by month.

    Args:
        principal:             Starting investment (€)
        annual_rate:           Annual interest rate (%)
        years:                 Investment horizon
        monthly_contribution:  Amount added every month (€)
        compounds_per_year:    How often interest is applied (default: monthly = 12)

    Returns:
        DataFrame with year-by-year breakdown
    """
    r = annual_rate / 100 / compounds_per_year  # rate per compounding period
    records = []

    balance = principal
    total_contributed = principal

    for year in range(1, years + 1):
        # Run through each month of this year
        for _ in range(12):
            balance = balance * (1 + r) + monthly_contribution
            total_contributed += monthly_contribution

        interest_earned = balance - total_contributed

        records.append({
            "Year":                year,
            "Balance (€)":         round(balance, 2),
            "Total Contributed (€)": round(total_contributed, 2),
            "Interest Earned (€)": round(interest_earned, 2),
        })

    return pd.DataFrame(records)


# ── 2. SCENARIO COMPARISON ───────────────────────────────────────────────────

def compare_scenarios(scenarios: list[dict], years: int) -> dict[str, pd.DataFrame]:
    """
    Run compound_growth for multiple scenarios and return all results.

    Each scenario dict should have keys:
        label, principal, annual_rate, monthly_contribution
    """
    results = {}
    for s in scenarios:
        df = compound_growth(
            principal=s["principal"],
            annual_rate=s["annual_rate"],
            years=years,
            monthly_contribution=s.get("monthly_contribution", 0),
        )
        results[s["label"]] = df
    return results


# ── 3. VISUALISATIONS ────────────────────────────────────────────────────────

COLOURS = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]


def plot_growth_comparison(results: dict, years: int):
    """Line chart comparing final balance across all scenarios."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (label, df) in enumerate(results.items()):
        colour = COLOURS[i % len(COLOURS)]
        ax.plot(df["Year"], df["Balance (€)"], label=label,
                color=colour, linewidth=2.2)
        # Annotate the final value
        final = df["Balance (€)"].iloc[-1]
        ax.annotate(f"€{final:,.0f}", xy=(years, final),
                    xytext=(6, 0), textcoords="offset points",
                    color=colour, fontsize=9, va="center")

    ax.set_title("Investment Growth Over Time — Scenario Comparison",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Portfolio Value (€)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("growth_comparison.png", dpi=150)
    plt.show()
    print("  ✔ Saved: growth_comparison.png")


def plot_stacked_breakdown(label: str, df: pd.DataFrame):
    """
    Stacked area chart for one scenario showing:
    - Money you actually put in (contributions)
    - Interest earned on top of that
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.stackplot(
        df["Year"],
        df["Total Contributed (€)"],
        df["Interest Earned (€)"],
        labels=["Your Contributions", "Interest Earned"],
        colors=["#2563EB", "#16A34A"],
        alpha=0.8,
    )

    ax.set_title(f"Contributions vs Interest Earned — {label}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Value (€)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    filename = f"breakdown_{label.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"  ✔ Saved: {filename}")


def plot_rate_sensitivity(principal: float, monthly_contribution: float, years: int):
    """
    Bar chart: final value at different interest rates.
    Illustrates how sensitive outcomes are to rate changes.
    """
    rates  = [2, 4, 6, 8, 10, 12]
    finals = []

    for rate in rates:
        df = compound_growth(principal, rate, years, monthly_contribution)
        finals.append(df["Balance (€)"].iloc[-1])

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([f"{r}%" for r in rates], finals,
                  color=COLOURS[:len(rates)], alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, finals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(finals) * 0.01,
                f"€{val:,.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"Final Value by Interest Rate after {years} Years\n"
                 f"Starting €{principal:,.0f} + €{monthly_contribution:,.0f}/month",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Annual Interest Rate", fontsize=12)
    ax.set_ylabel("Final Portfolio Value (€)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("rate_sensitivity.png", dpi=150)
    plt.show()
    print("  ✔ Saved: rate_sensitivity.png")


# ── 4. MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   📈  Compound Interest Visualizer")
    print("=" * 55)

    # ── Get inputs from the user ──
    principal            = float(input("\nStarting investment (€): "))
    annual_rate          = float(input("Annual interest rate (%): "))
    monthly_contribution = float(input("Monthly contribution (€): "))
    years                = int(input("Investment term (years): "))

    # ── Build 4 scenarios based on their input ──
    # Each scenario tweaks one variable so the user can see the difference
    scenarios = [
        {
            "label":                f"€{principal:,.0f} lump sum, {annual_rate}%",
            "principal":            principal,
            "annual_rate":          annual_rate,
            "monthly_contribution": 0,
        },
        {
            "label":                f"€{principal:,.0f} + €{monthly_contribution/2:,.0f}/month, {annual_rate}%",
            "principal":            principal,
            "annual_rate":          annual_rate,
            "monthly_contribution": monthly_contribution / 2,
        },
        {
            "label":                f"€{principal:,.0f} + €{monthly_contribution:,.0f}/month, {annual_rate}%",
            "principal":            principal,
            "annual_rate":          annual_rate,
            "monthly_contribution": monthly_contribution,
        },
        {
            "label":                f"€{principal:,.0f} + €{monthly_contribution:,.0f}/month, {annual_rate + 2}%",
            "principal":            principal,
            "annual_rate":          annual_rate + 2,
            "monthly_contribution": monthly_contribution,
        },
    ]

    print(f"\nRunning {len(scenarios)} scenarios over {years} years...\n")

    # ── Run all scenarios ──
    results = compare_scenarios(scenarios, years)

    # ── Print summary table ──
    print("─" * 55)
    print(f"  {'Scenario':<35} {'Final Value':>15}")
    print("─" * 55)
    for label, df in results.items():
        final = df["Balance (€)"].iloc[-1]
        print(f"  {label:<35} €{final:>13,.2f}")
    print("─" * 55)

    # ── Generate charts ──
    print("\n  Generating charts...")
    plot_growth_comparison(results, years)

    # Detailed breakdown for the most interesting scenario
    best_label = list(results.keys())[-1]
    plot_stacked_breakdown(best_label, results[best_label])

    # Rate sensitivity for base case
    plot_rate_sensitivity(
        principal=5_000,
        monthly_contribution=200,
        years=years,
    )

    print("\n✅ Done! All charts saved to your project folder.\n")


if __name__ == "__main__":
    main()
