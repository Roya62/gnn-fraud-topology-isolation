"""
eda.py — Exploratory Data Analysis for the Sparkov credit card fraud dataset.

Produces 14 publication-ready figures saved to ../outcomes/eda_sparkov/
Covers:
  1.  Class distribution (fraud vs legitimate)
  2.  Fraud rate by hour of day
  3.  Fraud rate by day of week
  4.  Fraud rate by month
  5.  Transaction amount distribution (fraud vs legitimate)
  6.  Amount box-plots by class (log scale)
  7.  Top-20 merchant categories by fraud rate
  8.  Fraud rate by state (choropleth-style bar)
  9.  Customer age distribution by class
  10. Inter-transaction time difference (trans_time_diff) by class
  11. Distance (customer → merchant) by class
  12. Amount-to-average ratio (amt_to_avg) by class
  13. Per-cardholder transaction count distribution
  14. Per-cardholder fraud rate distribution

Run:
    cd sparkov
    python eda.py

Dataset required:
    fraudTrain.csv + fraudTest.csv  (Kaggle Sparkov)
    https://www.kaggle.com/datasets/kartik2112/fraud-detection
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
from matplotlib.gridspec import GridSpec
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_PATH   = "fraudTrain.csv"
TEST_PATH    = "fraudTest.csv"
OUT_DIR      = os.path.join("..", "outcomes", "eda_sparkov")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE      = {"Legitimate": "#4C72B0", "Fraud": "#DD8452"}
FRAUD_COLOR  = "#DD8452"
LEG_COLOR    = "#4C72B0"
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "savefig.bbox": "tight", "font.family": "sans-serif"})


# ── Load & engineer ───────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("Loading Sparkov dataset...")
    df = pd.concat([pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)],
                   ignore_index=True).dropna()
    print(f"  Combined: {len(df):,} rows")

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"]     = pd.to_datetime(df["dob"])
    df["age"]     = ((df["trans_date_trans_time"] - df["dob"]).dt.days // 365).astype(int)
    df["hour"]    = df["trans_date_trans_time"].dt.hour
    df["weekday"] = df["trans_date_trans_time"].dt.weekday
    df["month"]   = df["trans_date_trans_time"].dt.month
    df["is_fraud"] = df["is_fraud"].astype(int)
    df["class_label"] = df["is_fraud"].map({0: "Legitimate", 1: "Fraud"})

    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    df["trans_time_diff"] = (
        df.groupby("cc_num")["trans_date_trans_time"]
        .diff().dt.total_seconds().div(60).fillna(0))

    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        df["distance"] = np.sqrt(
            (df["lat"] - df["merch_lat"])**2 + (df["long"] - df["merch_long"])**2)

    cc_avg = df.groupby("cc_num")["amt"].transform("mean")
    df["amt_to_avg"] = df["amt"] / (cc_avg + 1e-10)

    gc.collect()
    return df


# ── Helper ────────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 1. Class distribution ─────────────────────────────────────────────────────
def plot_class_distribution(df):
    counts   = df["is_fraud"].value_counts().sort_index()
    labels   = ["Legitimate", "Fraud"]
    colors   = [LEG_COLOR, FRAUD_COLOR]
    fraud_pct = 100 * counts[1] / len(df)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Figure 1 — Class Distribution", fontsize=13, fontweight="bold")

    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + len(df)*0.005, f"{v:,}", ha="center", fontsize=10)
    axes[0].set_ylabel("Transaction count")
    axes[0].set_title("Transaction counts by class")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.2f%%",
                startangle=140, wedgeprops={"edgecolor": "white"})
    axes[1].set_title(f"Class balance  (fraud rate = {fraud_pct:.2f}%)")

    plt.tight_layout()
    save(fig, "01_class_distribution.png")
    print(f"    Fraud: {counts[1]:,} ({fraud_pct:.2f}%)  |  Legitimate: {counts[0]:,}")


# ── 2. Fraud rate by hour ─────────────────────────────────────────────────────
def plot_fraud_by_hour(df):
    hourly = df.groupby("hour")["is_fraud"].agg(["sum", "count"])
    hourly["rate"] = hourly["sum"] / hourly["count"] * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(hourly.index, hourly["rate"], color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Hour of day (0–23)"); ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Figure 2 — Fraud Rate by Hour of Day", fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, 24))
    ax.axhline(hourly["rate"].mean(), linestyle="--", color="grey", linewidth=1,
               label=f"Mean {hourly['rate'].mean():.2f}%")
    ax.legend(); plt.tight_layout()
    save(fig, "02_fraud_rate_by_hour.png")


# ── 3. Fraud rate by day of week ──────────────────────────────────────────────
def plot_fraud_by_weekday(df):
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = df.groupby("weekday")["is_fraud"].agg(["sum", "count"])
    daily["rate"] = daily["sum"] / daily["count"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Figure 3 — Transaction Volume & Fraud Rate by Day of Week",
                 fontsize=13, fontweight="bold")

    axes[0].bar(day_names, daily["count"], color=LEG_COLOR, alpha=0.85, edgecolor="white")
    axes[0].set_ylabel("Transaction count"); axes[0].set_title("Volume by weekday")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))

    axes[1].bar(day_names, daily["rate"], color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    axes[1].set_ylabel("Fraud rate (%)"); axes[1].set_title("Fraud rate by weekday")
    axes[1].axhline(daily["rate"].mean(), linestyle="--", color="grey", linewidth=1)

    plt.tight_layout(); save(fig, "03_fraud_by_weekday.png")


# ── 4. Fraud rate by month ────────────────────────────────────────────────────
def plot_fraud_by_month(df):
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = df.groupby("month")["is_fraud"].agg(["sum","count"])
    monthly["rate"] = monthly["sum"] / monthly["count"] * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly.index, monthly["rate"], marker="o", color=FRAUD_COLOR, linewidth=2)
    ax.fill_between(monthly.index, monthly["rate"], alpha=0.15, color=FRAUD_COLOR)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(month_names)
    ax.set_ylabel("Fraud rate (%)"); ax.set_xlabel("Month")
    ax.set_title("Figure 4 — Fraud Rate by Month", fontsize=13, fontweight="bold")
    plt.tight_layout(); save(fig, "04_fraud_by_month.png")


# ── 5. Amount distribution ────────────────────────────────────────────────────
def plot_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Figure 5 — Transaction Amount Distribution by Class",
                 fontsize=13, fontweight="bold")

    for label, color in PALETTE.items():
        sub = df[df["class_label"] == label]["amt"]
        axes[0].hist(sub.clip(upper=sub.quantile(0.99)), bins=80,
                     alpha=0.6, color=color, label=label, density=True)
    axes[0].set_xlabel("Amount ($)"); axes[0].set_ylabel("Density")
    axes[0].set_title("Amount distribution (clipped at 99th pct)"); axes[0].legend()

    for label, color in PALETTE.items():
        sub = np.log1p(df[df["class_label"] == label]["amt"])
        axes[1].hist(sub, bins=80, alpha=0.6, color=color, label=label, density=True)
    axes[1].set_xlabel("log(1 + Amount)"); axes[1].set_ylabel("Density")
    axes[1].set_title("Log-amount distribution"); axes[1].legend()

    plt.tight_layout(); save(fig, "05_amount_distribution.png")


# ── 6. Amount box-plots ───────────────────────────────────────────────────────
def plot_amount_boxplot(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [np.log1p(df[df["class_label"] == l]["amt"]) for l in ["Legitimate", "Fraud"]]
    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], [LEG_COLOR, FRAUD_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_ylabel("log(1 + Amount)")
    ax.set_title("Figure 6 — Amount Box-Plots by Class (log scale)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); save(fig, "06_amount_boxplot.png")

    for label in ["Legitimate", "Fraud"]:
        sub = df[df["class_label"] == label]["amt"]
        print(f"  {label:12s}  median=${sub.median():.2f}  mean=${sub.mean():.2f}  "
              f"max=${sub.max():.2f}")


# ── 7. Top-20 merchant categories ────────────────────────────────────────────
def plot_category_fraud(df):
    if "category" not in df.columns:
        return
    cat = df.groupby("category")["is_fraud"].agg(["sum","count"])
    cat["rate"] = cat["sum"] / cat["count"] * 100
    cat = cat.sort_values("rate", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(cat.index[::-1], cat["rate"][::-1], color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Fraud rate (%)"); ax.set_title(
        "Figure 7 — Top-20 Merchant Categories by Fraud Rate",
        fontsize=13, fontweight="bold")
    plt.tight_layout(); save(fig, "07_fraud_by_category.png")


# ── 8. Fraud rate by state ────────────────────────────────────────────────────
def plot_state_fraud(df):
    if "state" not in df.columns:
        return
    state = df.groupby("state")["is_fraud"].agg(["sum","count"])
    state["rate"] = state["sum"] / state["count"] * 100
    state = state.sort_values("rate", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(state.index, state["rate"], color=FRAUD_COLOR, alpha=0.8, edgecolor="white")
    ax.axhline(state["rate"].mean(), linestyle="--", color="grey", linewidth=1,
               label=f"Mean {state['rate'].mean():.2f}%")
    ax.set_xlabel("State"); ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Figure 8 — Fraud Rate by State", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=7); ax.legend()
    plt.tight_layout(); save(fig, "08_fraud_by_state.png")


# ── 9. Customer age distribution ──────────────────────────────────────────────
def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, color in PALETTE.items():
        sub = df[df["class_label"] == label]["age"]
        ax.hist(sub, bins=50, alpha=0.6, color=color, label=label, density=True)
    ax.set_xlabel("Customer age (years)"); ax.set_ylabel("Density")
    ax.set_title("Figure 9 — Customer Age Distribution by Class",
                 fontsize=13, fontweight="bold")
    ax.legend(); plt.tight_layout(); save(fig, "09_age_distribution.png")

    for label in ["Legitimate", "Fraud"]:
        sub = df[df["class_label"] == label]["age"]
        print(f"  {label:12s}  median age={sub.median():.0f}  mean={sub.mean():.1f}")


# ── 10. Inter-transaction time difference ────────────────────────────────────
def plot_time_diff(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Figure 10 — Inter-Transaction Time Difference (minutes) by Class",
                 fontsize=13, fontweight="bold")

    for label, color in PALETTE.items():
        sub = df[df["class_label"] == label]["trans_time_diff"]
        sub_clip = sub.clip(upper=sub.quantile(0.99))
        axes[0].hist(sub_clip, bins=80, alpha=0.6, color=color, label=label, density=True)

    axes[0].set_xlabel("Time diff (min, clipped 99th pct)")
    axes[0].set_ylabel("Density"); axes[0].legend()
    axes[0].set_title("Raw distribution")

    log_data = [np.log1p(df[df["class_label"] == l]["trans_time_diff"].clip(lower=0))
                for l in ["Legitimate", "Fraud"]]
    bp = axes[1].boxplot(log_data, patch_artist=True, notch=False,
                         medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], [LEG_COLOR, FRAUD_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    axes[1].set_xticklabels(["Legitimate", "Fraud"])
    axes[1].set_ylabel("log(1 + time diff)"); axes[1].set_title("Log-scale box-plot")

    plt.tight_layout(); save(fig, "10_time_diff_distribution.png")


# ── 11. Distance distribution ─────────────────────────────────────────────────
def plot_distance(df):
    if "distance" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Figure 11 — Customer→Merchant Distance by Class",
                 fontsize=13, fontweight="bold")

    for label, color in PALETTE.items():
        sub = df[df["class_label"] == label]["distance"]
        axes[0].hist(sub.clip(upper=sub.quantile(0.99)), bins=80,
                     alpha=0.6, color=color, label=label, density=True)
    axes[0].set_xlabel("Distance (degrees, clipped 99th pct)")
    axes[0].set_ylabel("Density"); axes[0].legend(); axes[0].set_title("Raw distribution")

    log_data = [np.log1p(df[df["class_label"] == l]["distance"]) for l in ["Legitimate","Fraud"]]
    bp = axes[1].boxplot(log_data, patch_artist=True, notch=False,
                         medianprops={"color":"white","linewidth":2})
    for patch, color in zip(bp["boxes"], [LEG_COLOR, FRAUD_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    axes[1].set_xticklabels(["Legitimate","Fraud"])
    axes[1].set_ylabel("log(1 + distance)"); axes[1].set_title("Log-scale box-plot")

    plt.tight_layout(); save(fig, "11_distance_distribution.png")


# ── 12. Amount-to-average ratio ───────────────────────────────────────────────
def plot_amt_to_avg(df):
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, color in PALETTE.items():
        sub = df[df["class_label"] == label]["amt_to_avg"]
        sub_clip = sub.clip(upper=sub.quantile(0.99))
        ax.hist(sub_clip, bins=80, alpha=0.6, color=color, label=label, density=True)
    ax.set_xlabel("Amount / cardholder average (clipped 99th pct)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 12 — Amount-to-Cardholder-Average Ratio by Class",
                 fontsize=13, fontweight="bold")
    ax.axvline(1.0, linestyle="--", color="grey", linewidth=1, label="ratio = 1")
    ax.legend(); plt.tight_layout(); save(fig, "12_amt_to_avg_distribution.png")

    for label in ["Legitimate","Fraud"]:
        sub = df[df["class_label"]==label]["amt_to_avg"]
        print(f"  {label:12s}  median ratio={sub.median():.2f}  "
              f"% above avg={100*(sub>1).mean():.1f}%")


# ── 13. Per-cardholder transaction count ─────────────────────────────────────
def plot_cardholder_txn_count(df):
    counts = df.groupby("cc_num").size()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Figure 13 — Per-Cardholder Transaction Count",
                 fontsize=13, fontweight="bold")

    axes[0].hist(counts, bins=60, color=LEG_COLOR, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Transactions per cardholder"); axes[0].set_ylabel("Number of cardholders")
    axes[0].set_title("Full distribution")

    axes[1].hist(np.log1p(counts), bins=60, color=LEG_COLOR, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("log(1 + transactions per cardholder)")
    axes[1].set_title("Log-scale distribution")

    plt.tight_layout(); save(fig, "13_cardholder_txn_count.png")
    print(f"  Cardholders: {len(counts):,}  |  median txns/card: {counts.median():.0f}  "
          f"|  max: {counts.max()}")


# ── 14. Per-cardholder fraud rate ─────────────────────────────────────────────
def plot_cardholder_fraud_rate(df):
    card_stats = df.groupby("cc_num")["is_fraud"].agg(["mean","sum","count"])
    card_stats.columns = ["fraud_rate","fraud_count","total"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Figure 14 — Per-Cardholder Fraud Rate Distribution",
                 fontsize=13, fontweight="bold")

    axes[0].hist(card_stats["fraud_rate"]*100, bins=60,
                 color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Per-cardholder fraud rate (%)")
    axes[0].set_ylabel("Number of cardholders")
    axes[0].set_title("Fraud rate distribution across cardholders")

    has_fraud = card_stats[card_stats["fraud_count"] > 0]
    axes[1].hist(has_fraud["fraud_count"], bins=30,
                 color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Number of fraud transactions per cardholder")
    axes[1].set_title(f"Fraud count per cardholder\n(among {len(has_fraud):,} affected cardholders)")

    plt.tight_layout(); save(fig, "14_cardholder_fraud_rate.png")
    clean = card_stats[card_stats["fraud_count"] == 0]
    print(f"  Cardholders with zero fraud: {len(clean):,} ({100*len(clean)/len(card_stats):.1f}%)")
    print(f"  Cardholders with ≥1 fraud:   {len(has_fraud):,}")


# ── Summary stats ─────────────────────────────────────────────────────────────
def print_summary(df):
    print("\n" + "="*60)
    print("SPARKOV DATASET — SUMMARY STATISTICS")
    print("="*60)
    print(f"  Total transactions : {len(df):,}")
    print(f"  Fraud transactions : {df['is_fraud'].sum():,} ({100*df['is_fraud'].mean():.3f}%)")
    print(f"  Unique cardholders : {df['cc_num'].nunique():,}")
    print(f"  Unique merchants   : {df['merchant'].nunique():,}")
    if "category" in df.columns:
        print(f"  Merchant categories: {df['category'].nunique()}")
    if "state" in df.columns:
        print(f"  States             : {df['state'].nunique()}")
    print(f"  Amount range       : ${df['amt'].min():.2f} – ${df['amt'].max():.2f}")
    print(f"  Median amount      : ${df['amt'].median():.2f}")
    print(f"  Fraud median amt   : ${df[df['is_fraud']==1]['amt'].median():.2f}")
    print(f"  Legit median amt   : ${df[df['is_fraud']==0]['amt'].median():.2f}")
    print(f"  Age range          : {df['age'].min()}–{df['age'].max()} years")
    date_col = "trans_date_trans_time"
    if date_col in df.columns:
        print(f"  Date range         : {df[date_col].min().date()} → {df[date_col].max().date()}")
    print("="*60 + "\n")


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
    plot_category_fraud(df)
    plot_state_fraud(df)
    plot_age_distribution(df)
    plot_time_diff(df)
    plot_distance(df)
    plot_amt_to_avg(df)
    plot_cardholder_txn_count(df)
    plot_cardholder_fraud_rate(df)

    print(f"\nAll figures saved to: {os.path.abspath(OUT_DIR)}")
