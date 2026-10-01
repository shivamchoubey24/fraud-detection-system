"""Leakage-free streaming feature engineering.

`FeatureStore` is processed in strict time order.  For every transaction the features only use
information from transactions that happened BEFORE (plus the current one's identifiers), never labels.
The exact same code runs offline (training) and online (API), so there is no train/serve skew.

Graph features (transaction network, NetworkX-style but implemented with a union-find so that it is
incremental and O(1) per transaction - NetworkX is used offline for the fraud-ring analysis/plot):
  nodes  = cards, devices, IPs      edges = "card used device / IP in a transaction"
  device_n_cards     distinct cards seen on this device        (fraud rings share devices)
  ip_n_cards         distinct cards seen on this IP
  card_n_devices     distinct devices used by this card
  component_log_size log(1 + size of the connected component the txn touches)
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import pandas as pd

from config import CATEGORIES

STATELESS = ["amount", "log_amount", "hour", "day_of_week", "is_night", "is_foreign",
             "dist_from_home_km", "card_age_days"] + [f"cat_{c}" for c in CATEGORIES]
VELOCITY = ["is_new_device", "card_txn_1h", "card_txn_24h", "secs_since_last", "amount_ratio"]
GRAPH = ["device_n_cards", "ip_n_cards", "card_n_devices", "component_log_size"]
ALL_FEATURES = STATELESS + VELOCITY + GRAPH
NO_GRAPH_FEATURES = STATELESS + VELOCITY
MAX_GAP = 7 * 86400


class FeatureStore:
    def __init__(self):
        self.parent: dict = {}
        self.size: dict = {}
        self.device_cards: dict = {}
        self.ip_cards: dict = {}
        self.card_devices: dict = {}
        self.card_hist: dict = {}          # card -> deque[(ts, amount)] (last 50)
        self.n_seen = 0

    # ---- union-find ------------------------------------------------------
    def _find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:       # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    # ---- features ----------------------------------------------------------
    def features(self, t: dict, commit: bool) -> dict:
        """Compute state-dependent features for txn `t`; if commit, then add txn to the state."""
        cid, did, iid, ts, amt = t["card_id"], t["device_id"], t["ip_id"], int(t["timestamp"]), float(t["amount"])
        nodes = [("c", cid), ("d", did), ("i", iid)]

        n_dev_cards = len(self.device_cards.get(did, set()) | {cid})
        n_ip_cards = len(self.ip_cards.get(iid, set()) | {cid})
        cd = self.card_devices.get(cid, set())
        is_new_dev = int(did not in cd)
        n_card_devs = len(cd | {did})

        hist = self.card_hist.get(cid)
        if hist:
            gap = min(max(ts - hist[-1][0], 0), MAX_GAP)
            n1h = 1 + sum(1 for h, _ in hist if 0 <= ts - h <= 3600)
            n24h = 1 + sum(1 for h, _ in hist if 0 <= ts - h <= 86400)
            ratio = amt / max(float(np.mean([a for _, a in hist])), 1.0)
        else:
            gap, n1h, n24h, ratio = MAX_GAP, 1, 1, 1.0

        roots = {self._find(n) for n in nodes if n in self.parent}
        comp = sum(self.size[r] for r in roots) + sum(1 for n in nodes if n not in self.parent)

        out = {"is_new_device": is_new_dev, "card_txn_1h": n1h, "card_txn_24h": n24h,
               "secs_since_last": gap, "amount_ratio": ratio,
               "device_n_cards": n_dev_cards, "ip_n_cards": n_ip_cards,
               "card_n_devices": n_card_devs, "component_log_size": math.log1p(comp)}

        if commit:
            for n in nodes:
                if n not in self.parent:
                    self.parent[n], self.size[n] = n, 1
            self._union(nodes[0], nodes[1])
            self._union(nodes[0], nodes[2])
            self.device_cards.setdefault(did, set()).add(cid)
            self.ip_cards.setdefault(iid, set()).add(cid)
            self.card_devices.setdefault(cid, set()).add(did)
            self.card_hist.setdefault(cid, deque(maxlen=50)).append((ts, amt))
            self.n_seen += 1
        return out


def stateless_features(t: dict) -> dict:
    ts = int(t["timestamp"])
    hour = (ts // 3600) % 24
    f = {"amount": float(t["amount"]), "log_amount": math.log1p(float(t["amount"])),
         "hour": hour, "day_of_week": ((ts // 86400) + 3) % 7,      # 1970-01-01 was a Thursday
         "is_night": int(hour < 6), "is_foreign": int(t["is_foreign"]),
         "dist_from_home_km": float(t["dist_from_home_km"]), "card_age_days": int(t["card_age_days"])}
    for c in CATEGORIES:
        f[f"cat_{c}"] = int(t["merchant_category"] == c)
    return f


def featurize(t: dict, store: FeatureStore, commit: bool) -> dict:
    f = stateless_features(t)
    f.update(store.features(t, commit))
    return f


def build_feature_frame(df: pd.DataFrame, store: FeatureStore | None = None):
    """Process a time-sorted dataframe sequentially -> (X, store)."""
    assert df["timestamp"].is_monotonic_increasing, "dataframe must be sorted by time"
    store = store or FeatureStore()
    rows = [featurize(t, store, commit=True) for t in df.to_dict("records")]
    X = pd.DataFrame(rows)[ALL_FEATURES].astype("float32")
    return X, store


def now_ts() -> int:
    return int(time.time())
