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


# ── Silent result container ───────────────────────────────────────────────────
class _Result:
    """
    A dict-like container that prints nothing in Jupyter notebooks.

    Functions like ev.metrics() and ev.bv_stats() return this so their
    output isn't auto-displayed as a raw dict. You can still access all
    data normally: r["accuracy"], r["diagnosis"], r.get("f1"), etc.

    Usage
    -----
    ev.metrics(rf, X_test, y_test)                  # prints cleanly, no dict shown
    m = ev.metrics(rf, X_test, y_test)              # store result
    print(m["accuracy"])                             # access any key
    dict(m)                                          # convert to plain dict
    """

    def __init__(self, data: dict):
        self._data = data

    # ── dict-like access ──────────────────────────────────────────
    def __getitem__(self, key):        return self._data[key]
    def __setitem__(self, key, value): self._data[key] = value
    def __contains__(self, key):       return key in self._data
    def __iter__(self):                return iter(self._data)
    def __len__(self):                 return len(self._data)
    def keys(self):                    return self._data.keys()
    def values(self):                  return self._data.values()
    def items(self):                   return self._data.items()
    def get(self, key, default=None):  return self._data.get(key, default)

    # ── suppress Jupyter auto-display ────────────────────────────
    def __repr__(self):  return ""
    def __str__(self):   return ""

    # ── convert to plain dict when needed ────────────────────────
    def to_dict(self):   return dict(self._data)


class _ResultDF:
    """
    A pandas DataFrame wrapper that prints nothing in Jupyter notebooks.

    Functions like ev.compare_metrics() return this so the DataFrame
    isn't auto-displayed. You can still use all normal DataFrame operations:
    df["F1"], df.idxmax(), df.values, for row in df.iterrows(), etc.

    Usage
    -----
    ev.compare_metrics(models, X_test, y_test)          # prints table, no display
    df = ev.compare_metrics(models, X_test, y_test)     # store result
    print(df["F1"].idxmax())                             # use like normal DataFrame
    df.to_dataframe()                                    # get plain pandas DataFrame
    """

    def __init__(self, df):
        self._df = df

    # ── delegate all DataFrame operations ────────────────────────
    def __getitem__(self, key):       return self._df[key]
    def __setitem__(self, key, val):  self._df[key] = val
    def __len__(self):                return len(self._df)
    def __iter__(self):               return iter(self._df)
    def __contains__(self, key):      return key in self._df

    # delegate common DataFrame attributes and methods
    @property
    def columns(self):   return self._df.columns
    @property
    def index(self):     return self._df.index
    @property
    def values(self):    return self._df.values
    @property
    def shape(self):     return self._df.shape

    def idxmax(self, *a, **kw):      return self._df.idxmax(*a, **kw)
    def idxmin(self, *a, **kw):      return self._df.idxmin(*a, **kw)
    def iterrows(self):              return self._df.iterrows()
    def loc(self, *a, **kw):         return self._df.loc[*a]
    def get(self, key, default=None):return self._df.get(key, default)
    def items(self):                 return self._df.items()

    # ── suppress Jupyter auto-display ────────────────────────────
    def __repr__(self):  return ""
    def __str__(self):   return ""

    # ── convert to plain DataFrame ───────────────────────────────
    def to_dataframe(self): return self._df

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
