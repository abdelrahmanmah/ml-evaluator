"""
ml_evaluator.bias_variance
====================
Bias–Variance analysis — every piece is its own function.

Stats & Interpretation
-----------------------
    ml_evaluator.bv_stats(model, X_train, y_train)

Individual plot
---------------
    ml_evaluator.plot_learning_curve(model, X_train, y_train)

All-in-one shortcut
-------------------
    ml_evaluator.bias_variance(model, X_train, y_train)

Multi-model
-----------
    ml_evaluator.compare_bias_variance(models, X_train, y_train)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedKFold, learning_curve

from ._utils import (
    DEFAULT_COLORS, get_colors,
    OVERFIT_THRESHOLD, HIGH_BIAS_THRESHOLD,
    diagnose_bv, styled_box, add_panel_label, _Result,
)

warnings.filterwarnings("ignore")

__all__ = [
    "bv_stats",
    "plot_learning_curve",
    "bias_variance",
    "compare_bias_variance",
]


# ══════════════════════════════════════════════════════════════════════════════
# Core computation (internal)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_bv(model, X_train, y_train,
                n_splits, random_state, train_sizes, scoring,
                overfit_threshold, high_bias_threshold) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    sizes, lc_train, lc_val = learning_curve(
        model, X_train, y_train,
        cv=skf, scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1,
    )
    mean_train = lc_train[-1].mean()
    mean_val   = lc_val[-1].mean()
    val_std    = lc_val[-1].std()
    gap        = mean_train - mean_val
    bias_proxy = 1.0 - mean_val

    label, color, explanation = diagnose_bv(
        gap, bias_proxy, overfit_threshold, high_bias_threshold
    )
    return {
        "lc_sizes":    sizes,
        "lc_train":    lc_train,
        "lc_val":      lc_val,
        "mean_train":  mean_train,
        "mean_val":    mean_val,
        "val_std":     val_std,
        "gap":         gap,
        "bias_proxy":  bias_proxy,
        "diagnosis":   label,
        "diag_color":  color,
        "explanation": explanation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1.  bv_stats()  — numbers + diagnosis only, no plot
# ══════════════════════════════════════════════════════════════════════════════

def bv_stats(
    model,
    X_train,
    y_train,
    *,
    model_name:          str   = "Model",
    n_splits:            int   = 5,
    random_state:        int   = 42,
    train_sizes                = None,
    scoring:             str   = "accuracy",
    overfit_threshold:   float = OVERFIT_THRESHOLD,
    high_bias_threshold: float = HIGH_BIAS_THRESHOLD,
    verbose:             bool  = True,
):
    """
    Compute and print Bias–Variance stats for one model.
    No plot — numbers and diagnosis only.

    Parameters
    ----------
    model               : fitted sklearn estimator
    X_train             : training features
    y_train             : training labels
    model_name          : label shown in the printed header
    n_splits            : number of CV folds (default 5)
    random_state        : reproducibility seed
    train_sizes         : array of fractions — default linspace(0.1, 1.0, 8)
    scoring             : sklearn scoring string (default 'accuracy')
    overfit_threshold   : train−val gap above which → Overfit
    high_bias_threshold : val error rate above which → Underfit
    verbose             : if False, suppresses terminal output
    return_data         : if True, returns the results dict (default False)

    Returns
    -------
    None by default. dict if return_data=True.

    Example
    -------
    >>> ev.bv_stats(rf, X_train, y_train)                        # prints only
    >>> r = ev.bv_stats(rf, X_train, y_train, return_data=True)  # prints + returns
    >>> print(r["diagnosis"])
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    r = _compute_bv(model, X_train, y_train,
                    n_splits, random_state, train_sizes, scoring,
                    overfit_threshold, high_bias_threshold)

    if verbose:
        _print_bv(model_name, r)

    return _Result(r)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  plot_learning_curve()  — plot only
# ══════════════════════════════════════════════════════════════════════════════

def plot_learning_curve(
    model,
    X_train,
    y_train,
    *,
    model_name:          str           = "Model",
    n_splits:            int           = 5,
    random_state:        int           = 42,
    train_sizes                        = None,
    scoring:             str           = "accuracy",
    overfit_threshold:   float         = OVERFIT_THRESHOLD,
    high_bias_threshold: float         = HIGH_BIAS_THRESHOLD,
    color:               str           = DEFAULT_COLORS[0],
    figsize:             tuple         = (8, 5),
    save_path:           Optional[str] = None,
) -> None:
    """
    Plot the learning curve for one model.
    No terminal output — plot only.

    Shows train vs validation accuracy as training size increases,
    with ±1 std shading and a gap annotation at the full-data point.

    Parameters
    ----------
    model               : fitted sklearn estimator
    X_train             : training features
    y_train             : training labels
    model_name          : title label
    n_splits            : CV folds
    random_state        : seed
    train_sizes         : fractions — default linspace(0.1, 1.0, 8)
    scoring             : sklearn scoring string
    overfit_threshold   : threshold for Overfit diagnosis
    high_bias_threshold : threshold for Underfit diagnosis
    color               : train curve colour
    figsize             : figure size
    save_path           : optional path to save

    Returns
    -------
    dict  — same as bv_stats()

    Example
    -------
    >>> ml_evaluator.plot_learning_curve(rf, X_train, y_train, color="#2ECC71")
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    r = _compute_bv(model, X_train, y_train,
                    n_splits, random_state, train_sizes, scoring,
                    overfit_threshold, high_bias_threshold)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.30)

    _draw_lc_ax(ax, r, color)
    ax.set_title(
        f"Learning Curve — {model_name}\nDiagnosis: {r['diagnosis']}",
        fontweight="bold", fontsize=11, color=r["diag_color"],
    )
    add_panel_label(ax, "Learning Curve")
    styled_box(ax, r["explanation"], r["diag_color"], fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  bias_variance()  — stats + plot together (shortcut)
# ══════════════════════════════════════════════════════════════════════════════

def bias_variance(
    model,
    X_train,
    y_train,
    *,
    model_name:          str           = "Model",
    n_splits:            int           = 5,
    random_state:        int           = 42,
    train_sizes                        = None,
    scoring:             str           = "accuracy",
    overfit_threshold:   float         = OVERFIT_THRESHOLD,
    high_bias_threshold: float         = HIGH_BIAS_THRESHOLD,
    color:               str           = DEFAULT_COLORS[0],
    figsize:             tuple         = (8, 5),
    save_path:           Optional[str] = None,
):
    """
    Full Bias–Variance analysis for one model: terminal stats + learning curve.

    Parameters
    ----------
    (same as bv_stats + plot_learning_curve)
    return_data : if True, returns the results dict (default False)

    Returns
    -------
    None by default. dict if return_data=True.

    Example
    -------
    >>> ev.bias_variance(rf, X_train, y_train)
    >>> r = ev.bias_variance(rf, X_train, y_train, return_data=True)
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    r = _compute_bv(model, X_train, y_train,
                    n_splits, random_state, train_sizes, scoring,
                    overfit_threshold, high_bias_threshold)

    _print_bv(model_name, r)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.30)

    _draw_lc_ax(ax, r, color)
    ax.set_title(
        f"Learning Curve — {model_name}\nDiagnosis: {r['diagnosis']}",
        fontweight="bold", fontsize=11, color=r["diag_color"],
    )
    add_panel_label(ax, "Learning Curve")
    styled_box(ax, r["explanation"], r["diag_color"], fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()

    return _Result(r)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  compare_bias_variance()  — multi-model dashboard
# ══════════════════════════════════════════════════════════════════════════════

def compare_bias_variance(
    models: Dict,
    X_train,
    y_train,
    *,
    n_splits:            int            = 5,
    random_state:        int            = 42,
    train_sizes                         = None,
    scoring:             str            = "accuracy",
    overfit_threshold:   float          = OVERFIT_THRESHOLD,
    high_bias_threshold: float          = HIGH_BIAS_THRESHOLD,
    colors:              Optional[List] = None,
    figsize:             Optional[tuple]= None,
    save_path:           Optional[str]  = None,
):
    """
    Bias–Variance dashboard for multiple models.

    Row 1 — Learning curve per model with diagnosis label.
    Row 2 — Bar charts: Bias Proxy · Variance · Overfitting Gap.

    Parameters
    ----------
    models      : dict  {model_name: fitted sklearn estimator}
    X_train     : training features
    y_train     : training labels
    return_data : if True, returns {model_name: result_dict} (default False)

    Returns
    -------
    None by default. dict if return_data=True.

    Example
    -------
    >>> ev.compare_bias_variance(models, X_train, y_train)
    >>> bv = ev.compare_bias_variance(models, X_train, y_train, return_data=True)
    >>> print(bv["Random Forest"]["diagnosis"])
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    names  = list(models.keys())
    n      = len(names)
    colors = colors or get_colors(n)

    # compute
    results = {}
    print("=" * 60)
    print("  Bias–Variance Analysis")
    print("=" * 60)
    for name, model in models.items():
        r = _compute_bv(model, X_train, y_train,
                        n_splits, random_state, train_sizes, scoring,
                        overfit_threshold, high_bias_threshold)
        results[name] = r
        _print_bv(name, r)

    # figure
    if figsize is None:
        figsize = (max(5 * n, 14), 12)

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    gs_top = gridspec.GridSpec(1, n, figure=fig,
                               left=0.05, right=0.97,
                               top=0.93,  bottom=0.52)
    gs_bot = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.05, right=0.97,
                               top=0.42,  bottom=0.07)

    for col, (name, color) in enumerate(zip(names, colors)):
        ax = fig.add_subplot(gs_top[0, col])
        r  = results[name]
        _draw_lc_ax(ax, r, color)
        ax.set_title(f"{name}\n{r['diagnosis']}",
                     fontweight="bold", fontsize=9, color=r["diag_color"])
        short = _short_explanation(r)
        ax.text(0.5, -0.28, short,
                transform=ax.transAxes,
                fontsize=7.5, ha="center", va="top", color=r["diag_color"],
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=r["diag_color"] + "18",
                          edgecolor=r["diag_color"], linewidth=1))

    fig.text(0.01, 0.95, "Row 1 — Learning Curves",
             fontsize=10, fontweight="bold", color="#333")

    bias_vals = [results[n]["bias_proxy"] for n in names]
    var_vals  = [results[n]["val_std"]    for n in names]
    gap_vals  = [results[n]["gap"]        for n in names]

    _draw_summary_bars(fig, gs_bot, names, colors,
                       bias_vals, var_vals, gap_vals,
                       overfit_threshold, high_bias_threshold)

    fig.text(0.01, 0.44, "Row 2 — Bias · Variance · Gap",
             fontsize=10, fontweight="bold", color="#333")
    fig.suptitle("Bias–Variance Dashboard",
                 fontsize=14, fontweight="bold", y=0.99)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Saved → {save_path}")
    plt.show()

    return _Result({k: _Result(v) for k, v in results.items()})


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers (internal)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_lc_ax(ax, r: dict, color: str):
    sizes      = r["lc_sizes"]
    train_mean = r["lc_train"].mean(axis=1)
    train_std  = r["lc_train"].std(axis=1)
    val_mean   = r["lc_val"].mean(axis=1)
    val_std    = r["lc_val"].std(axis=1)

    ax.plot(sizes, train_mean, "o-",  color=color,    lw=2, label="Train")
    ax.plot(sizes, val_mean,   "s--", color="#7F8C8D", lw=2, label="Validation")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=color)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.10, color="#7F8C8D")
    ax.annotate("",
                xy=(sizes[-1], val_mean[-1]),
                xytext=(sizes[-1], train_mean[-1]),
                arrowprops=dict(arrowstyle="<->", color="#E74C3C", lw=1.5))
    ax.text(sizes[-1] * 1.01, (val_mean[-1] + train_mean[-1]) / 2,
            f"gap\n{r['gap']:.3f}", fontsize=7, color="#E74C3C", va="center")
    ax.set_xlabel("Training Set Size", fontsize=8)
    ax.set_ylabel("Score", fontsize=8)
    ax.set_ylim(max(0.4, min(train_mean.min(), val_mean.min()) - 0.05), 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)


def _draw_summary_bars(fig, gs, names, colors,
                       bias_vals, var_vals, gap_vals,
                       overfit_threshold, high_bias_threshold):
    datasets = [
        ("Bias Proxy\n(1 − Val Accuracy)",  bias_vals,
         "#E67E22", high_bias_threshold, "High Bias threshold"),
        ("Variance\n(Std of Val Scores)",    var_vals,
         "#9B59B6", 0.02,                "High Variance threshold"),
        ("Overfit Gap\n(Train − Val)",       gap_vals,
         "#E74C3C", overfit_threshold,    "Overfit threshold"),
    ]
    for col, (title, vals, th_color, threshold, th_label) in enumerate(datasets):
        ax = fig.add_subplot(gs[0, col])
        bar_colors = ["#E74C3C" if v > threshold else c
                      for v, c in zip(vals, colors)]
        bars = ax.bar(names, vals, color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.8)
        ax.axhline(threshold, color=th_color, linestyle="--",
                   lw=1.5, label=th_label, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, max(max(vals) * 1.3, threshold * 1.5))
        best_i = int(np.argmin(vals))
        ax.patches[best_i].set_edgecolor("#2ECC71")
        ax.patches[best_i].set_linewidth(2.5)


# ══════════════════════════════════════════════════════════════════════════════
# Terminal helpers (internal)
# ══════════════════════════════════════════════════════════════════════════════

def _print_bv(name: str, r: dict):
    icon = {"Good Fit": "✅", "Underfit": "⚠️ ", "Overfit": "🔴"}.get(r["diagnosis"], "  ")
    print(f"\n  {icon} {name}")
    print(f"     Train Acc : {r['mean_train']:.4f}")
    print(f"     Val   Acc : {r['mean_val']:.4f}")
    print(f"     Gap       : {r['gap']:.4f}   (threshold: {OVERFIT_THRESHOLD})")
    print(f"     Val Std   : {r['val_std']:.4f}   (variance proxy)")
    print(f"     Diagnosis : {r['diagnosis']}")
    print(f"     {r['explanation']}")


def _short_explanation(r: dict) -> str:
    d = r["diagnosis"]
    if d == "Overfit":
        return f"Gap={r['gap']:.3f} — memorises training data. Try regularisation."
    if d == "Underfit":
        return f"Error={r['bias_proxy']:.3f} — model too simple. Try more complexity."
    return f"Gap={r['gap']:.3f} — generalises well ✓"