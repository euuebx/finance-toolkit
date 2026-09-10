"""
Loan & Mortgage Calculator
Calculates monthly repayments, total interest paid, and produces
an amortization schedule showing how each payment is split between
principal and interest over the life of the loan.

Libraries used:
- matplotlib: for plotting graphs
- pandas: for displaying the amortization table neatly
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


# ── 1. CORE FORMULA ──────────────────────────────────────────────────────────

def monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    """
    Calculate the fixed monthly repayment using the standard amortization formula.

    Formula:
        M = P * [r(1+r)^n] / [(1+r)^n - 1]

    Where:
        P = principal (loan amount)
        r = monthly interest rate (annual rate / 12)
        n = total number of monthly payments (years * 12)
    """
    r = annual_rate / 100 / 12   # convert annual % to monthly decimal
    n = years * 12               # total number of payments

    if r == 0:                   # edge case: 0% interest loan
        return principal / n

    M = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return M


# ── 2. AMORTIZATION SCHEDULE ─────────────────────────────────────────────────

def amortization_schedule(principal: float, annual_rate: float, years: int) -> pd.DataFrame:
    """
    Build a month-by-month table showing:
    - How much of each payment goes to interest
    - How much goes to reducing the principal (balance owed)
    - The remaining balance after each payment
    """
    r = annual_rate / 100 / 12
    n = years * 12
    M = monthly_payment(principal, annual_rate, years)

    balance = principal
    records = []

    for month in range(1, n + 1):
        interest_payment = balance * r            # interest owed this month
        principal_payment = M - interest_payment  # remainder chips away at balance
        balance -= principal_payment
        balance = max(balance, 0)                 # avoid floating-point negatives

        records.append({
            "Month":             month,
            "Payment (€)":       round(M, 2),
            "Principal (€)":     round(principal_payment, 2),
            "Interest (€)":      round(interest_payment, 2),
            "Remaining Balance (€)": round(balance, 2),
        })

    return pd.DataFrame(records)


# ── 3. SUMMARY STATS ─────────────────────────────────────────────────────────

def loan_summary(principal: float, annual_rate: float, years: int) -> dict:
    """Return key summary figures for the loan."""
    M   = monthly_payment(principal, annual_rate, years)
    n   = years * 12
    total_paid    = M * n
    total_interest = total_paid - principal

    return {
        "Principal":        principal,
        "Annual Rate (%)":  annual_rate,
        "Term (years)":     years,
        "Monthly Payment":  round(M, 2),
        "Total Paid":       round(total_paid, 2),
        "Total Interest":   round(total_interest, 2),
    }


# ── 4. VISUALISATIONS ────────────────────────────────────────────────────────

def plot_balance_over_time(df: pd.DataFrame, principal: float, annual_rate: float, years: int):
    """Line chart: how the remaining balance falls each month."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["Month"], df["Remaining Balance (€)"], color="#2563EB", linewidth=2)
    ax.fill_between(df["Month"], df["Remaining Balance (€)"], alpha=0.1, color="#2563EB")

    ax.set_title(f"Remaining Loan Balance Over Time\n"
                 f"€{principal:,.0f} at {annual_rate}% over {years} years",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Balance (€)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("balance_over_time.png", dpi=150)
    plt.show()
    print("  ✔ Saved: balance_over_time.png")


def plot_payment_breakdown(df: pd.DataFrame, principal: float, annual_rate: float, years: int):
    """Stacked area chart: each payment split into principal vs interest."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.stackplot(
        df["Month"],
        df["Principal (€)"],
        df["Interest (€)"],
        labels=["Principal", "Interest"],
        colors=["#16A34A", "#DC2626"],
        alpha=0.8,
    )

    ax.set_title(f"Monthly Payment Breakdown: Principal vs Interest\n"
                 f"€{principal:,.0f} at {annual_rate}% over {years} years",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Amount (€)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("payment_breakdown.png", dpi=150)
    plt.show()
    print("  ✔ Saved: payment_breakdown.png")


def plot_pie_summary(summary: dict):
    """Pie chart: what proportion of total payment is interest vs principal."""
    labels  = ["Principal", "Total Interest"]
    sizes   = [summary["Principal"], summary["Total Interest"]]
    colors  = ["#16A34A", "#DC2626"]
    explode = (0.03, 0.03)

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12}
    )
    ax.set_title("Total Cost Breakdown", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("cost_breakdown.png", dpi=150)
    plt.show()
    print("  ✔ Saved: cost_breakdown.png")


# ── 5. MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("   🏦  Loan & Mortgage Calculator")
    print("=" * 50)

    # ── Get inputs from the user ──
    principal   = float(input("\nLoan amount (€): "))
    annual_rate = float(input("Annual interest rate (%): "))
    years       = int(input("Loan term (years): "))

    # ── Calculate everything ──
    summary = loan_summary(principal, annual_rate, years)
    df      = amortization_schedule(principal, annual_rate, years)

    # ── Print summary ──
    print("\n" + "─" * 50)
    print("  LOAN SUMMARY")
    print("─" * 50)
    for key, value in summary.items():
        if "Rate" in key or "years" in key.lower():
            print(f"  {key:<25} {value}")
        else:
            print(f"  {key:<25} €{value:>12,.2f}")

    # ── Show first and last 5 rows of the schedule ──
    print("\n  AMORTIZATION SCHEDULE (first 5 months)")
    print(df.head().to_string(index=False))
    print("\n  ... (middle rows hidden) ...\n")
    print("  AMORTIZATION SCHEDULE (last 5 months)")
    print(df.tail().to_string(index=False))

    # ── Save full schedule to CSV ──
    df.to_csv("amortization_schedule.csv", index=False)
    print("\n  ✔ Full schedule saved: amortization_schedule.csv")

    # ── Generate plots ──
    print("\n  Generating charts...")
    plot_balance_over_time(df, principal, annual_rate, years)
    plot_payment_breakdown(df, principal, annual_rate, years)
    plot_pie_summary(summary)

    print("\n✅ Done! Check the charts and CSV in your project folder.\n")


if __name__ == "__main__":
    main()
