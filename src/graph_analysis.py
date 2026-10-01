"""Offline fraud-ring analysis with NetworkX.

* Builds the card - device - IP transaction network.
* Cross-checks the streaming union-find component sizes against NetworkX connected components.
* Finds suspicious connected components (many cards sharing devices / IPs) and draws the biggest ones.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    for c, d, i, f in zip(df["card_id"], df["device_id"], df["ip_id"], df["is_fraud"]):
        g.add_edge(("c", c), ("d", d))
        g.add_edge(("c", c), ("i", i))
        if f:
            g.nodes[("c", c)]["fraud"] = g.nodes[("c", c)].get("fraud", 0) + 1
    return g


def component_table(g: nx.Graph, min_cards: int = 3) -> pd.DataFrame:
    rows = []
    for k, comp in enumerate(nx.connected_components(g)):
        cards = [n for n in comp if n[0] == "c"]
        if len(cards) >= min_cards:
            rows.append({"component": k, "n_cards": len(cards), "n_nodes": len(comp),
                         "fraud_txns": sum(g.nodes[n].get("fraud", 0) for n in cards),
                         "fraud_cards": sum(1 for n in cards if g.nodes[n].get("fraud", 0) > 0)})
    out = pd.DataFrame(rows)
    if len(out):
        out["fraud_card_share"] = out["fraud_cards"] / out["n_cards"]
        out = out.sort_values(["fraud_card_share", "n_cards"], ascending=False).reset_index(drop=True)
    return out


def verify_against_store(g: nx.Graph, store) -> bool:
    """Union-find component sizes in the FeatureStore must equal NetworkX component sizes."""
    nx_sizes = {}
    for comp in nx.connected_components(g):
        for n in comp:
            nx_sizes[n] = len(comp)
    return all(store.size[store._find(n)] == s for n, s in nx_sizes.items())


def plot_rings(g: nx.Graph, path, top: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = sorted((c for c in nx.connected_components(g)
                    if sum(1 for n in c if n[0] == "c") >= 3), key=lambda c: -sum(
        1 for n in c if n[0] == "c" and g.nodes[n].get("fraud", 0)))[:top]
    if not comps:
        return
    fig, axes = plt.subplots(1, len(comps), figsize=(4.2 * len(comps), 4.2))
    axes = [axes] if len(comps) == 1 else axes
    for ax, comp in zip(axes, comps):
        sub = g.subgraph(comp)
        pos = nx.spring_layout(sub, seed=7)
        colors = ["#d62728" if n[0] == "c" and g.nodes[n].get("fraud", 0) else
                  "#1f77b4" if n[0] == "c" else "#7f7f7f" for n in sub]
        sizes = [70 if n[0] == "c" else 30 for n in sub]
        nx.draw(sub, pos, ax=ax, node_color=colors, node_size=sizes, edge_color="#bbbbbb", width=0.8)
        ax.set_title(f"{sum(1 for n in sub if n[0]=='c')} cards", fontsize=9)
    fig.suptitle("Suspicious card-device-IP components (red = card with fraud, blue = clean card, grey = device/IP)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
