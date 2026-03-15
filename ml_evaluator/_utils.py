"""
mleval._utils
=============
Internal shared helpers — not part of the public API.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
)

# ── Default palette ────────────────────────────────────────────────────────────
DEFAULT_COLORS = [
    "#2E75B6", "#2ECC71", "#E67E22", "#E74C3C",
    "#9B59B6", "#1ABC9C", "#F39C12", "#3498DB",
]

# ── Diagnosis thresholds ───────────────────────────────────────────────────────
OVERFIT_THRESHOLD   = 0.10   # train−val gap
HIGH_BIAS_THRESHOLD = 0.15   # 1 − val_accuracy


def get_colors(n: int):
    """Return n colors, cycling the palette if needed."""
    return [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n)]


def compute_metrics(model, X_test, y_test) -> dict:
    """
    Compute all standard classification metrics for one model.
    Returns a plain dict — no side effects.
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )
    rep = classification_report(y_test, y_pred, output_dict=True)

    positive_label = str(sorted(rep.keys())[-2])   # '1' or last non-avg key

    metrics = {
        "y_pred":     y_pred,
        "y_prob":     y_prob,
        "accuracy":   round(accuracy_score(y_test, y_pred), 4),
        "f1":         round(f1_score(y_test, y_pred, average="binary"), 4),
        "precision":  round(precision_score(y_test, y_pred, average="binary"), 4),
        "recall":     round(recall_score(y_test, y_pred, average="binary"), 4),
        "roc_auc":    round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        "report":     rep,
    }
    return metrics


def diagnose_bv(gap: float, bias_proxy: float,
                overfit_threshold=OVERFIT_THRESHOLD,
                high_bias_threshold=HIGH_BIAS_THRESHOLD) -> tuple[str, str, str]:
    """
    Returns (label, color, explanation) for a bias-variance diagnosis.
    """
    if gap > overfit_threshold:
        return (
            "Overfit",
            "#E74C3C",
            f"Train–val gap is {gap:.3f} (>{overfit_threshold}). "
            "The model memorises training data but fails to generalise.\n"
            "→ Try: regularisation, reduce model complexity, more training data, dropout."
        )
    if bias_proxy > high_bias_threshold:
        return (
            "Underfit",
            "#E67E22",
            f"Val error rate is {bias_proxy:.3f} (>{high_bias_threshold}). "
            "The model is too simple to capture the pattern in the data.\n"
            "→ Try: more complex model, add features, reduce regularisation, train longer."
        )
    return (
        "Good Fit",
        "#2ECC71",
        f"Train–val gap is {gap:.3f} and val error is {bias_proxy:.3f}. "
        "The model generalises well — no strong signs of overfit or underfit.\n"
        "→ Next step: evaluate on the held-out test set."
    )


def styled_box(ax, text: str, color: str, fontsize: int = 9):
    """Add an interpretation text box at the bottom of an axes."""
    ax.text(
        0.01, -0.32, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=color + "22",   # 13% opacity hex
            edgecolor=color,
            linewidth=1.4,
        ),
        wrap=True,
    )


def add_panel_label(ax, label: str, color="#2E75B6"):
    """Top-left panel tag  e.g. 'A · Confusion Matrix'."""
    ax.text(
        0.0, 1.04, label,
        transform=ax.transAxes,
        fontsize=9, fontweight="bold",
        color=color, va="bottom",
    )
