"""
eda.py — Exploratory Data Analysis for the IBM synthetic credit card fraud dataset.

Produces 14 publication-ready figures saved to ../outcomes/eda_ibm/
Covers:
  1.  Class distribution (fraud vs legitimate)
  2.  Fraud rate by hour of day
  3.  Fraud rate by day of week
  4.  Fraud rate by month
  5.  Transaction amount distribution (fraud vs legitimate)
  6.  Amount box-plots by class (log scale)
  7.  Top-20 merchant cities by fraud rate
  8.  Fraud rate by Use Chip method
  9.  Fraud rate by Error type
  10. Top-20 MCC codes by fraud rate
  11. Per-user transaction count distribution
  12. Per-card transaction count distribution
  13. Per-user fraud rate distribution
  14. Amount-to-user-average ratio by class

Run:
    cd ibm
    python eda.py

Dataset required:
    IBM synthetic credit card transactions (CSV or Parquet)
    https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Update DATA_PATH below to your local file before running.
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH = "/path/to/ibm_transactions.parquet"   # ← update this
OUT_DIR   = os.path.join("..", "outcomes", "eda_ibm")
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE    = {"Legitimate": "#4C72B0", "Fraud": "#DD8452"}
FRAUD_COLOR = "#DD8452"
LEG_COLOR   = "#4C72B0"
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "savefig.bbox": "tight", "font.family": "sans-serif"})


# ── Load & engineer ───────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print(f"Loading IBM dataset from {DATA_PATH}...")
    path_lower = DATA_PATH.lower()
    df = (pd.read_parquet(DATA_PATH)
          if path_lower.endswith((".parquet", ".pq"))
          else pd.read_csv(DATA_PATH))
    df.columns = [c.strip() for c in df.columns]
    print(f"  Loaded: {len(df):,} rows")

    # Normalise label
    y = df["Is Fraud?"].astype(str).str.strip().str.lower()
    df["is_fraud"] = y.map({"yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0).astype(int)
    df["class_label"] = df["is_fraud"].map({0: "Legitimate", 1: "Fraud"})

    # Amount
    df["Amount"] = pd.to_numeric(
        df["Amount"].astype(str).str.replace(r"[£$€,]", "", regex=True),
        errors="coerce").fillna(0)

    # Time
    time_parsed = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
    df["hour"]   = time_parsed.dt.hour.fillna(-1).astype(int)
    df["minute"] = time_parsed.dt.minute.fillna(-1).astype(int)

    # Date
    date_str = (df["Year"].astype(str).str.zfill(4) + "-" +
                df["Month"].astype(str).str.zfill(2) + "-" +
                df["Day"].astype(str).str.zfill(2))
    df["transaction_dt"] = pd.to_datetime(date_str, errors="coerce")
    df["day_of_week"]    = df["transaction_dt"].dt.dayofweek.fillna(-1).astype(int)
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)

    # User-average amount
    user_avg = df.groupby("User")["Amount"].transform("mean")
    df["amt_to_user_avg"] = df["Amount"] / (user_avg + 1e-10)

    gc.collect()
    return df


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 1. Class distribution ─────────────────────────────────────────────────────
def plot_class_distribution(df):
    counts    = df["is_fraud"].value_counts().sort_index()
    labels    = ["Legitimate", "Fraud"]
    colors    = [LEG_COLOR, FRAUD_COLOR]
    fraud_pct = 100 * counts.get(1, 0) / len(df)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Figure 1 — Class Distribution", fontsize=13, fontweight="bold")

    axes[0].bar(labels, [counts.get(0,0), counts.get(1,0)],
                color=colors, edgecolor="white")
    for i, v in enumerate([counts.get(0,0), counts.get(1,0)]):
        axes[0].text(i, v + len(df)*0.003, f"{v:,}", ha="center", fontsize=10)
    axes[0].set_ylabel("Transaction count"); axes[0].set_title("Transaction counts by class")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    axes[1].pie([counts.get(0,0), counts.get(1,0)], labels=labels, colors=colors,
                autopct="%1.2f%%", startangle=140, wedgeprops={"edgecolor": "white"})
    axes[1].set_title(f"Class balance  (fraud rate = {fraud_pct:.2f}%)")

    plt.tight_layout(); save(fig, "01_class_distribution.png")
    print(f"    Fraud: {counts.get(1,0):,} ({fraud_pct:.2f}%)  |  "
          f"Legitimate: {counts.get(0,0):,}")


# ── 2. Fraud rate by hour ─────────────────────────────────────────────────────
def plot_fraud_by_hour(df):
    valid = df[df["hour"] >= 0]
    hourly = valid.groupby("hour")["is_fraud"].agg(["sum","count"])
    hourly["rate"] = hourly["sum"] / hourly["count"] * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(hourly.index, hourly["rate"], color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    ax.axhline(hourly["rate"].mean(), linestyle="--", color="grey", linewidth=1,
               label=f"Mean {hourly['rate'].mean():.2f}%")
    ax.set_xlabel("Hour of day (0–23)"); ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Figure 2 — Fraud Rate by Hour of Day", fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, 24)); ax.legend()
    plt.tight_layout(); save(fig, "02_fraud_rate_by_hour.png")


# ── 3. Fraud rate by day of week ──────────────────────────────────────────────
def plot_fraud_by_weekday(df):
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    valid = df[df["day_of_week"] >= 0]
    daily = valid.groupby("day_of_week")["is_fraud"].agg(["sum","count"])
    daily["rate"] = daily["sum"] / daily["count"] * 100
    daily = daily.reindex(range(7), fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Figure 3 — Volume & Fraud Rate by Day of Week",
                 fontsize=13, fontweight="bold")
    axes[0].bar(day_names, daily["count"], color=LEG_COLOR, alpha=0.85, edgecolor="white")
    axes[0].set_ylabel("Transaction count"); axes[0].set_title("Volume by weekday")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    axes[1].bar(day_names, daily["rate"], color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    axes[1].axhline(daily["rate"].mean(), linestyle="--", color="grey", linewidth=1)
    axes[1].set_ylabel("Fraud rate (%)"); axes[1].set_title("Fraud rate by weekday")
    plt.tight_layout(); save(fig, "03_fraud_by_weekday.png")


# ── 4. Fraud rate by month ────────────────────────────────────────────────────
def plot_fraud_by_month(df):
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = df.groupby("Month")["is_fraud"].agg(["sum","count"])
    monthly["rate"] = monthly["sum"] / monthly["count"] * 100
    monthly = monthly.reindex(range(1,13), fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly.index, monthly["rate"], marker="o", color=FRAUD_COLOR, linewidth=2)
    ax.fill_between(monthly.index, monthly["rate"], alpha=0.15, color=FRAUD_COLOR)
    ax.set_xticks(range(1,13)); ax.set_xticklabels(month_names)
    ax.set_ylabel("Fraud rate (%)"); ax.set_xlabel("Month")
    ax.set_title("Figure 4 — Fraud Rate by Month", fontsize=13, fontweight="bold")
    plt.tight_layout(); save(fig, "04_fraud_by_month.png")


# ── 5. Amount distribution ────────────────────────────────────────────────────
def plot_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Figure 5 — Transaction Amount Distribution by Class",
                 fontsize=13, fontweight="bold")
    for label, color in PALETTE.items():
        sub = df[df["class_label"] == label]["Amount"]
        axes[0].hist(sub.clip(upper=sub.quantile(0.99)), bins=80,
                     alpha=0.6, color=color, label=label, density=True)
    axes[0].set_xlabel("Amount ($)"); axes[0].set_ylabel("Density")
    axes[0].set_title("Amount distribution (clipped 99th pct)"); axes[0].legend()

    for label, color in PALETTE.items():
        sub = np.log1p(df[df["class_label"] == label]["Amount"].clip(lower=0))
        axes[1].hist(sub, bins=80, alpha=0.6, color=color, label=label, density=True)
    axes[1].set_xlabel("log(1 + Amount)"); axes[1].set_ylabel("Density")
    axes[1].set_title("Log-amount distribution"); axes[1].legend()
    plt.tight_layout(); save(fig, "05_amount_distribution.png")


# ── 6. Amount box-plots ───────────────────────────────────────────────────────
def plot_amount_boxplot(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [np.log1p(df[df["class_label"]==l]["Amount"].clip(lower=0))
            for l in ["Legitimate","Fraud"]]
    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops={"color":"white","linewidth":2})
    for patch, color in zip(bp["boxes"], [LEG_COLOR, FRAUD_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_xticklabels(["Legitimate","Fraud"])
    ax.set_ylabel("log(1 + Amount)")
    ax.set_title("Figure 6 — Amount Box-Plots by Class (log scale)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); save(fig, "06_amount_boxplot.png")
    for label in ["Legitimate","Fraud"]:
        sub = df[df["class_label"]==label]["Amount"]
        print(f"  {label:12s}  median=${sub.median():.2f}  mean=${sub.mean():.2f}")


# ── 7. Top merchant cities by fraud rate ──────────────────────────────────────
def plot_city_fraud(df):
    if "Merchant City" not in df.columns: return
    city = df.groupby("Merchant City")["is_fraud"].agg(["sum","count"])
    city = city[city["count"] >= 50]   # minimum support
    city["rate"] = city["sum"] / city["count"] * 100
    city = city.sort_values("rate", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(city.index[::-1], city["rate"][::-1], color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Fraud rate (%)"); ax.set_title(
        "Figure 7 — Top-20 Merchant Cities by Fraud Rate (min 50 txns)",
        fontsize=13, fontweight="bold")
    plt.tight_layout(); save(fig, "07_fraud_by_city.png")


# ── 8. Fraud rate by Use Chip ─────────────────────────────────────────────────
def plot_chip_fraud(df):
    if "Use Chip" not in df.columns: return
    chip = df.groupby("Use Chip")["is_fraud"].agg(["sum","count"])
    chip["rate"] = chip["sum"] / chip["count"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Figure 8 — Transaction Volume & Fraud Rate by Payment Method",
                 fontsize=13, fontweight="bold")
    axes[0].bar(chip.index, chip["count"], color=LEG_COLOR, alpha=0.85, edgecolor="white")
    axes[0].set_ylabel("Transaction count"); axes[0].set_title("Volume by payment method")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(chip.index, chip["rate"], color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    axes[1].set_ylabel("Fraud rate (%)"); axes[1].set_title("Fraud rate by payment method")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout(); save(fig, "08_fraud_by_chip.png")


# ── 9. Fraud rate by error type ───────────────────────────────────────────────
def plot_error_fraud(df):
    if "Errors?" not in df.columns: return
    err = df.groupby("Errors?")["is_fraud"].agg(["sum","count"])
    err["rate"] = err["sum"] / err["count"] * 100
    err = err.sort_values("rate", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 9 — Transaction Volume & Fraud Rate by Error Type",
                 fontsize=13, fontweight="bold")
    axes[0].barh(err.index[::-1], err["count"][::-1], color=LEG_COLOR, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Transaction count"); axes[0].set_title("Volume by error type")

    axes[1].barh(err.index[::-1], err["rate"][::-1], color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Fraud rate (%)"); axes[1].set_title("Fraud rate by error type")
    plt.tight_layout(); save(fig, "09_fraud_by_error.png")


# ── 10. Top MCC codes ─────────────────────────────────────────────────────────
def plot_mcc_fraud(df):
    if "MCC" not in df.columns: return
    mcc = df.groupby("MCC")["is_fraud"].agg(["sum","count"])
    mcc = mcc[mcc["count"] >= 100]
    mcc["rate"] = mcc["sum"] / mcc["count"] * 100
    mcc = mcc.sort_values("rate", ascending=False).head(20)
    mcc.index = mcc.index.astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(mcc.index[::-1], mcc["rate"][::-1], color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Fraud rate (%)")
    ax.set_title("Figure 10 — Top-20 MCC Codes by Fraud Rate (min 100 txns)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); save(fig, "10_fraud_by_mcc.png")


# ── 11. Per-user transaction count ───────────────────────────────────────────
def plot_user_txn_count(df):
    counts = df.groupby("User").size()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Figure 11 — Per-User Transaction Count",
                 fontsize=13, fontweight="bold")
    axes[0].hist(counts, bins=60, color=LEG_COLOR, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Transactions per user"); axes[0].set_ylabel("Number of users")
    axes[0].set_title("Full distribution")

    axes[1].hist(np.log1p(counts), bins=60, color=LEG_COLOR, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("log(1 + transactions per user)"); axes[1].set_title("Log-scale")
    plt.tight_layout(); save(fig, "11_user_txn_count.png")
    print(f"  Users: {len(counts):,}  |  median txns/user: {counts.median():.0f}  "
          f"|  max: {counts.max()}")


# ── 12. Per-card transaction count ───────────────────────────────────────────
def plot_card_txn_count(df):
    if "Card" not in df.columns: return
    counts = df.groupby("Card").size()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Figure 12 — Per-Card Transaction Count",
                 fontsize=13, fontweight="bold")
    axes[0].hist(counts, bins=60, color=LEG_COLOR, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Transactions per card"); axes[0].set_ylabel("Number of cards")
    axes[0].set_title("Full distribution")

    axes[1].hist(np.log1p(counts), bins=60, color=LEG_COLOR, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("log(1 + transactions per card)"); axes[1].set_title("Log-scale")
    plt.tight_layout(); save(fig, "12_card_txn_count.png")
    print(f"  Cards: {len(counts):,}  |  median txns/card: {counts.median():.0f}  "
          f"|  max: {counts.max()}")


# ── 13. Per-user fraud rate ───────────────────────────────────────────────────
def plot_user_fraud_rate(df):
    user_stats = df.groupby("User")["is_fraud"].agg(["mean","sum","count"])
    user_stats.columns = ["fraud_rate","fraud_count","total"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Figure 13 — Per-User Fraud Rate Distribution",
                 fontsize=13, fontweight="bold")
    axes[0].hist(user_stats["fraud_rate"]*100, bins=60,
                 color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Per-user fraud rate (%)"); axes[0].set_ylabel("Number of users")
    axes[0].set_title("Fraud rate distribution across users")

    has_fraud = user_stats[user_stats["fraud_count"] > 0]
    axes[1].hist(has_fraud["fraud_count"], bins=30,
                 color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Fraud transactions per user")
    axes[1].set_title(f"Fraud count per user\n({len(has_fraud):,} affected users)")

    plt.tight_layout(); save(fig, "13_user_fraud_rate.png")
    clean = user_stats[user_stats["fraud_count"] == 0]
    print(f"  Users with zero fraud: {len(clean):,} ({100*len(clean)/len(user_stats):.1f}%)")
    print(f"  Users with ≥1 fraud:   {len(has_fraud):,}")


# ── 14. Amount-to-user-average ratio ─────────────────────────────────────────
def plot_amt_to_user_avg(df):
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, color in PALETTE.items():
        sub = df[df["class_label"] == label]["amt_to_user_avg"]
        sub_clip = sub.clip(upper=sub.quantile(0.99), lower=0)
        ax.hist(sub_clip, bins=80, alpha=0.6, color=color, label=label, density=True)
    ax.axvline(1.0, linestyle="--", color="grey", linewidth=1, label="ratio = 1")
    ax.set_xlabel("Amount / user average (clipped 99th pct)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 14 — Amount-to-User-Average Ratio by Class",
                 fontsize=13, fontweight="bold")
    ax.legend(); plt.tight_layout(); save(fig, "14_amt_to_user_avg.png")
    for label in ["Legitimate","Fraud"]:
        sub = df[df["class_label"]==label]["amt_to_user_avg"]
        print(f"  {label:12s}  median ratio={sub.median():.2f}  "
              f"% above avg={100*(sub>1).mean():.1f}%")


# ── Summary stats ─────────────────────────────────────────────────────────────
def print_summary(df):
    print("\n"+"="*60)
    print("IBM DATASET — SUMMARY STATISTICS")
    print("="*60)
    print(f"  Total transactions : {len(df):,}")
    print(f"  Fraud transactions : {df['is_fraud'].sum():,} ({100*df['is_fraud'].mean():.3f}%)")
    print(f"  Unique users       : {df['User'].nunique():,}")
    if "Card" in df.columns:
        print(f"  Unique cards       : {df['Card'].nunique():,}")
    if "Merchant Name" in df.columns:
        print(f"  Unique merchants   : {df['Merchant Name'].nunique():,}")
    if "Merchant City" in df.columns:
        print(f"  Merchant cities    : {df['Merchant City'].nunique():,}")
    if "MCC" in df.columns:
        print(f"  Unique MCC codes   : {df['MCC'].nunique():,}")
    print(f"  Amount range       : ${df['Amount'].min():.2f} – ${df['Amount'].max():.2f}")
    print(f"  Median amount      : ${df['Amount'].median():.2f}")
    print(f"  Fraud median amt   : ${df[df['is_fraud']==1]['Amount'].median():.2f}")
    print(f"  Legit median amt   : ${df[df['is_fraud']==0]['Amount'].median():.2f}")
    if "Year" in df.columns:
        print(f"  Year range         : {df['Year'].min():.0f} – {df['Year'].max():.0f}")
    if "Use Chip" in df.columns:
        chip_dist = df["Use Chip"].value_counts()
        for k, v in chip_dist.items():
            rate = 100*df[df["Use Chip"]==k]["is_fraud"].mean()
            print(f"  Use Chip '{k:20s}': {v:,} txns, fraud rate {rate:.2f}%")
    print("="*60+"\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print_summary(df)

    print("\nGenerating figures...")
    plot_class_distribution(df)
    plot_fraud_by_hour(df)
    plot_fraud_by_weekday(df)
    plot_fraud_by_month(df)
    plot_amount_distribution(df)
    plot_amount_boxplot(df)
    plot_city_fraud(df)
    plot_chip_fraud(df)
    plot_error_fraud(df)
    plot_mcc_fraud(df)
    plot_user_txn_count(df)
    plot_card_txn_count(df)
    plot_user_fraud_rate(df)
    plot_amt_to_user_avg(df)

    print(f"\nAll figures saved to: {os.path.abspath(OUT_DIR)}")
