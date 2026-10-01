"""Synthetic card-transaction generator with realistic fraud patterns.

Why synthetic?  The real IEEE-CIS / Kaggle datasets cannot be bundled in a repo (licence + size)
and the Kaggle set has anonymised PCA features, so it has no card / device / IP identifiers
needed for graph features.  This generator creates those identifiers and three fraud typologies:

1. stolen card   - high amount, night, foreign, new device (some are 'stealthy' and look normal)
2. fraud ring    - groups of mule cards sharing devices / IPs, normal-looking amounts, bursts
3. card testing  - bursts of tiny transactions on one card

It also creates hard negatives (households sharing a device, travellers abroad, big purchases).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CATEGORIES, FRAUD_RATE, N_TXN, SEED

TS0 = 1_725_148_800            # 2024-09-01 00:00:00 UTC
DAYS = 30
RISKY_CATS = ["electronics", "online_retail", "travel"]
HOUR_P = np.array([1, 1, 1, 1, 1, 2, 3, 5, 6, 6, 6, 7, 8, 7, 6, 6, 6, 7, 8, 8, 7, 5, 3, 2], float)
HOUR_P /= HOUR_P.sum()


def _ts(rng, n, night_bias=False):
    p = HOUR_P.copy()
    if night_bias:
        p = np.where(np.arange(24) < 6, 0.12, 0.02)
        p = p / p.sum()
    day = rng.integers(0, DAYS, n)
    hour = rng.choice(24, n, p=p)
    return TS0 + day * 86400 + hour * 3600 + rng.integers(0, 3600, n)


def generate_synthetic(n_txn: int = N_TXN, fraud_rate: float = FRAUD_RATE, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_cards, n_merch, n_rings, ring_size = 1500, 300, 10, 6
    n_fraud = max(10, int(n_txn * fraud_rate))
    n_legit = n_txn - n_fraud

    # ---- card population --------------------------------------------------
    mu = rng.normal(3.6, 0.5, n_cards)                       # log-mean spend per card
    card_age = rng.integers(30, 3000, n_cards)
    home_dev = np.array([f"DH{i}" for i in range(n_cards)], dtype=object)
    home_ip = np.array([f"IH{i}" for i in range(n_cards)], dtype=object)
    perm = rng.permutation(n_cards - n_rings * ring_size)    # households (hard negatives)
    for i in range(0, 90, 2):
        home_dev[perm[i + 1]] = home_dev[perm[i]]
        home_ip[perm[i + 1]] = home_ip[perm[i]]
    ring_cards = np.arange(n_cards - n_rings * ring_size, n_cards)   # last 60 cards = mules
    card_age[ring_cards] = rng.integers(20, 150, len(ring_cards))
    merch_cat = rng.choice(CATEGORIES, n_merch)
    risky_m = np.where(np.isin(merch_cat, RISKY_CATS))[0]

    # ---- legitimate traffic ----------------------------------------------
    w = rng.gamma(2.0, 1.0, n_cards)
    card = rng.choice(n_cards, n_legit, p=w / w.sum())
    merch = rng.integers(0, n_merch, n_legit)
    amount = np.clip(np.exp(rng.normal(mu[card], 0.8)), 1, 5000)
    foreign = (rng.random(n_legit) < 0.03).astype(int)
    dist = np.where(rng.random(n_legit) < 0.05, rng.exponential(300, n_legit), rng.exponential(6, n_legit))
    new_dev = rng.random(n_legit) < 0.06
    dev = np.where(new_dev, [f"DR{x}" for x in rng.integers(0, 200_000, n_legit)], home_dev[card])
    new_ip = rng.random(n_legit) < 0.10
    ip = np.where(new_ip, [f"IR{x}" for x in rng.integers(0, 200_000, n_legit)], home_ip[card])
    legit = pd.DataFrame({
        "timestamp": _ts(rng, n_legit), "card": card, "merch": merch, "amount": amount,
        "device_id": dev, "ip_id": ip, "is_foreign": foreign, "dist": dist, "is_fraud": 0})

    # ---- fraud ------------------------------------------------------------
    n_stolen, n_ring = int(0.45 * n_fraud), int(0.30 * n_fraud)
    n_test = n_fraud - n_stolen - n_ring
    rows = []
    attacker_devs = [f"DA{i}" for i in range(15)]

    # 1) stolen cards
    ts_s = _ts(rng, n_stolen, night_bias=True)
    for k in range(n_stolen):
        c = int(rng.integers(0, n_cards))
        stealth = rng.random() < 0.25
        if stealth:
            amt, fr, ds = float(np.exp(rng.normal(mu[c], 0.8))), int(rng.random() < 0.05), float(rng.exponential(10))
            ts = int(_ts(rng, 1)[0])
        else:
            amt, fr, ds = float(np.exp(mu[c]) * rng.uniform(3, 10)), int(rng.random() < 0.5), float(rng.exponential(250))
            ts = int(ts_s[k])
        d = str(rng.choice(attacker_devs)) if rng.random() < 0.5 else f"DR{rng.integers(0, 200_000)}"
        m = int(rng.choice(risky_m)) if rng.random() < 0.7 else int(rng.integers(0, n_merch))
        rows.append((ts, c, m, min(amt, 5000), d, f"IR{rng.integers(0, 200_000)}", fr, ds))

    # 2) fraud rings (mule cards share 2 devices + 2 IPs, bursts, ordinary amounts)
    rings = [dict(cards=ring_cards[i * ring_size:(i + 1) * ring_size],
                  devs=[f"DX{i}a", f"DX{i}b"], ips=[f"IX{i}a", f"IX{i}b"],
                  merchants=rng.choice(risky_m, 3, replace=False)) for i in range(n_rings)]
    made = 0
    while made < n_ring:
        r = rings[int(rng.integers(0, n_rings))]
        t = int(_ts(rng, 1)[0])
        for _ in range(int(rng.integers(3, 6))):
            if made >= n_ring:
                break
            t += int(rng.integers(60, 1200))
            c = int(rng.choice(r["cards"]))
            rows.append((t, c, int(rng.choice(r["merchants"])), float(np.clip(np.exp(rng.normal(4.2, 0.5)), 5, 800)),
                         str(rng.choice(r["devs"])), str(rng.choice(r["ips"])),
                         int(rng.random() < 0.1), float(rng.exponential(20))))
            made += 1

    # 3) card testing (bursts of tiny amounts on one card)
    made = 0
    while made < n_test:
        c = int(rng.integers(0, n_cards))
        t = int(_ts(rng, 1)[0])
        d, i_ = f"DR{rng.integers(0, 200_000)}", f"IR{rng.integers(0, 200_000)}"
        for _ in range(int(rng.integers(5, 9))):
            if made >= n_test:
                break
            t += int(rng.integers(5, 120))
            rows.append((t, c, int(rng.integers(0, n_merch)), float(rng.uniform(0.5, 5)), d, i_,
                         int(rng.random() < 0.3), float(rng.exponential(30))))
            made += 1

    fraud = pd.DataFrame(rows, columns=["timestamp", "card", "merch", "amount", "device_id", "ip_id", "is_foreign", "dist"])
    fraud["is_fraud"] = 1

    df = pd.concat([legit, fraud], ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)
    out = pd.DataFrame({
        "txn_id": [f"T{i:06d}" for i in range(len(df))],
        "timestamp": df["timestamp"].astype("int64"),
        "card_id": ["C%04d" % c for c in df["card"]],
        "merchant_id": ["M%03d" % m for m in df["merch"]],
        "merchant_category": merch_cat[df["merch"].to_numpy()],
        "amount": df["amount"].round(2).astype(float),
        "device_id": df["device_id"].astype(str),
        "ip_id": df["ip_id"].astype(str),
        "is_foreign": df["is_foreign"].astype(int),
        "dist_from_home_km": df["dist"].round(1).astype(float),
        "card_age_days": card_age[df["card"].to_numpy()].astype(int),
        "is_fraud": df["is_fraud"].astype(int),
    })
    return out


if __name__ == "__main__":
    d = generate_synthetic()
    print(d.shape, "fraud rate: %.3f%%" % (100 * d.is_fraud.mean()))
    print(d.head())
