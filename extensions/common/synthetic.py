"""
synthetic.py — small synthetic datasets in the exact Sparkov / IBM-TabFormer
raw schemas, used ONLY to test code paths and demonstrate mechanisms.

Nothing produced from these data is a result about Sparkov or IBM. The
generator is deliberately simple: cardholders transact at random times;
a minority of cards suffer a short fraud burst (several fraudulent
transactions within ~48 h, higher amounts, new merchants). This burst
structure is the qualitative pattern both real datasets are known to show,
and it is what makes order-/neighbour-dependent artefacts visible.
"""
import numpy as np
import pandas as pd


def _bursty_card_history(rng, n_tx, t0, t1, fraud, burst_len):
    times = np.sort(rng.uniform(t0, t1, n_tx))
    is_fraud = np.zeros(n_tx, dtype=int)
    if fraud:
        start = rng.uniform(t0 + 0.3 * (t1 - t0), t1 - 3 * 86400)
        b_times = np.sort(start + rng.uniform(0, 2 * 86400, burst_len))
        times = np.concatenate([times, b_times])
        is_fraud = np.concatenate([is_fraud, np.ones(burst_len, dtype=int)])
        order = np.argsort(times, kind="mergesort")
        times, is_fraud = times[order], is_fraud[order]
    return times, is_fraud


def make_sparkov(n_cards=300, tx_per_card=(150, 450), fraud_card_share=0.12,
                 burst=(5, 14), n_merchants=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2019-01-01").value / 1e9
    t1 = pd.Timestamp("2020-12-31").value / 1e9
    cats = ["grocery_pos", "gas_transport", "shopping_net", "misc_net", "food_dining",
            "health_fitness", "entertainment", "travel", "home", "kids_pets",
            "personal_care", "shopping_pos", "misc_pos", "grocery_net"]
    states = ["NY", "CA", "TX", "PA", "OH", "FL", "IL", "MI"]
    jobs = [f"job_{i}" for i in range(60)]
    rows = []
    for c in range(n_cards):
        cc = 4_000_000_000_000_000 + c * 7919
        fraud = rng.random() < fraud_card_share
        n = int(rng.integers(*tx_per_card))
        times, y = _bursty_card_history(rng, n, t0, t1, fraud, int(rng.integers(*burst)))
        home_lat, home_long = rng.uniform(30, 45), rng.uniform(-120, -75)
        fav = rng.choice(n_merchants, size=12, replace=False)
        st, job = rng.choice(states), rng.choice(jobs)
        zipc, city = int(rng.integers(10000, 99999)), f"city_{c % 150}"
        dob = pd.Timestamp("1950-01-01") + pd.Timedelta(days=int(rng.integers(0, 18000)))
        for t, f in zip(times, y):
            m = int(rng.integers(n_merchants)) if f else int(rng.choice(fav))
            amt = float(rng.lognormal(6.0, 0.8) if f else rng.lognormal(3.5, 1.0))
            rows.append({
                "trans_date_trans_time": pd.Timestamp(t, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                "cc_num": cc, "merchant": f"fraud_m{m}", "category": cats[m % len(cats)],
                "amt": round(amt, 2), "first": "A", "last": "B", "gender": "F" if c % 2 else "M",
                "street": "x", "city": city, "state": st, "zip": zipc,
                "lat": home_lat, "long": home_long, "city_pop": int(1000 + 50 * c),
                "job": job, "dob": dob.strftime("%Y-%m-%d"),
                "trans_num": f"t{len(rows)}", "unix_time": int(t),
                "merch_lat": home_lat + rng.normal(0, 1 if not f else 3),
                "merch_long": home_long + rng.normal(0, 1 if not f else 3),
                "is_fraud": int(f)})
    df = pd.DataFrame(rows)
    df.insert(0, "Unnamed: 0", np.arange(len(df)))
    return df


def make_ibm(n_users=250, cards_per_user=(1, 4), tx_per_card=(60, 200),
             fraud_card_share=0.10, burst=(4, 12), n_merchants=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2015-01-01").value / 1e9
    t1 = pd.Timestamp("2019-12-31").value / 1e9
    cities = [f"City{i}" for i in range(80)]
    states_ = ["CA", "NY", "TX", "FL", "WA", "IL"]
    mccs = [5411, 5812, 5541, 5912, 4829, 5311, 5300, 7538, 5814, 4121]
    rows = []
    for u in range(n_users):
        for card in range(int(rng.integers(*cards_per_user))):
            fraud = rng.random() < fraud_card_share
            n = int(rng.integers(*tx_per_card))
            times, y = _bursty_card_history(rng, n, t0, t1, fraud, int(rng.integers(*burst)))
            fav = rng.choice(n_merchants, size=10, replace=False)
            home = int(rng.integers(len(cities)))
            for t, f in zip(times, y):
                ts = pd.Timestamp(t, unit="s")
                m = int(rng.integers(n_merchants)) if f else int(rng.choice(fav))
                amt = float(rng.lognormal(5.5, 0.9) if f else rng.lognormal(3.3, 1.0))
                rows.append({
                    "User": u, "Card": card, "Year": ts.year, "Month": ts.month, "Day": ts.day,
                    "Time": ts.strftime("%H:%M"), "Amount": f"${amt:.2f}",
                    "Use Chip": "Online Transaction" if (f and rng.random() < .6) else "Chip Transaction",
                    "Merchant Name": 1_000_000 + m * 7,
                    "Merchant City": cities[int(rng.integers(len(cities)))] if f else cities[home],
                    "Merchant State": states_[m % len(states_)],
                    "Zip": float(10000 + home), "MCC": mccs[m % len(mccs)],
                    "Errors?": "Insufficient Balance" if rng.random() < 0.01 else np.nan,
                    "Is Fraud?": "Yes" if f else "No"})
    return pd.DataFrame(rows)
