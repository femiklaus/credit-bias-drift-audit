"""
Render the Model Governance & Drift dashboard charts from the audit outputs.
Reads the CSVs in outputs/ and writes PNGs to charts/.

Usage:  python make_charts.py   (run credit_bias_drift_audit.py first)
Requires: pandas, matplotlib
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, CH = "outputs", "charts"
os.makedirs(CH, exist_ok=True)

GREEN, AMBER, RED, INK = "#2e7d57", "#d9a441", "#c0392b", "#1a1a1a"


def colour_for_di(di: float) -> str:
    return RED if di < 0.80 else AMBER if di < 0.90 else GREEN


# --- 1. Fairness: disparate impact by protected attribute --------------------
def fairness_chart() -> None:
    df = pd.read_csv(f"{OUT}/fairness_metrics.csv")
    s = df[df["group"] == "[SUMMARY]"][["protected_attribute", "disparate_impact_ratio"]]
    labels = s["protected_attribute"].str.replace("_", " ").tolist()
    vals = s["disparate_impact_ratio"].tolist()

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals, color=[colour_for_di(v) for v in vals], width=0.5, zorder=3)
    ax.axhline(0.80, color=RED, ls="--", lw=1, zorder=2)
    ax.axhline(0.90, color=AMBER, ls="--", lw=1, zorder=2)
    ax.text(ax.get_xlim()[1], 0.80, "  0.80 legal threshold (four-fifths rule)",
            va="center", ha="left", color=RED, fontsize=8)
    ax.text(ax.get_xlim()[1], 0.90, "  0.90 watch", va="center", ha="left",
            color=AMBER, fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", color=INK, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Disparate Impact ratio\n(unprivileged / privileged approval rate)")
    ax.set_title("Fairness — Disparate Impact of the credit-approval model",
                 fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    fig.tight_layout()
    fig.savefig(f"{CH}/fairness_disparate_impact.png", dpi=150)
    plt.close(fig)


# --- 2. Drift: PSI per feature, live vs simulated next batch ------------------
def drift_chart() -> None:
    df = pd.read_csv(f"{OUT}/drift_metrics.csv")
    feats = df["feature"].str.replace("_", " ").tolist()
    live = df["psi_reference_vs_live"].tolist()
    nxt = df["psi_reference_vs_next_batch"].tolist()
    x = range(len(feats))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar([i - w / 2 for i in x], live, w, label="Live batch", color="#4a6fa5", zorder=3)
    ax.bar([i + w / 2 for i in x], nxt, w, label="Simulated next quarter",
           color="#c0392b", zorder=3)
    ax.axhline(0.20, color=RED, ls="--", lw=1, zorder=2)
    ax.axhline(0.10, color=AMBER, ls="--", lw=1, zorder=2)
    ax.text(len(feats) - 0.5, 0.20, "  0.20 significant", color=RED, fontsize=8, va="center")
    ax.text(len(feats) - 0.5, 0.10, "  0.10 moderate", color=AMBER, fontsize=8, va="center")
    ax.set_xticks(list(x))
    ax.set_xticklabels(feats, rotation=20, ha="right")
    ax.set_ylabel("Population Stability Index (PSI)")
    ax.set_title("Data Drift — PSI vs training reference", fontweight="bold", color=INK)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    fig.tight_layout()
    fig.savefig(f"{CH}/drift_psi.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fairness_chart()
    drift_chart()
    print(f"Charts written to ./{CH}/  (fairness_disparate_impact.png, drift_psi.png)")
