"""
ml_evaluator.comparison
=================
Multi-model comparison — every piece is its own function.

Metrics & Interpretation
------------------------
    ml_evaluator.compare_metrics(models, X_test, y_test)
    ml_evaluator.compare_interpret(models, X_test, y_test)

Individual plots
----------------
    ml_evaluator.plot_confusion_matrices(models, X_test, y_test)
    ml_evaluator.compare_roc_curves(models, X_test, y_test)
    ml_evaluator.plot_metrics_comparison(models, X_test, y_test)

All-in-one shortcut
-------------------
    ml_evaluator.comparison_dashboard(models, X_test, y_test)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)

from ._utils import (
    get_colors, compute_metrics,
    styled_box, add_panel_label, _Result, _ResultDF,
)

warnings.filterwarnings("ignore")

__all__ = [
    "compare_metrics",
    "compare_interpret",
    "plot_confusion_matrices",
    "compare_roc_curves",
    "plot_metrics_comparison",
    "comparison_dashboard",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  compare_metrics()  — numbers + table only, no plot
# ══════════════════════════════════════════════════════════════════════════════

def compare_metrics(
    models: Dict,
    X_test,
    y_test,
    *,
    verbose: bool = True,
    return_data: bool = False,
):
    """
    Compute and print metrics for all models.
    No plot — numbers and winner summary only.

    Parameters
    ----------
    models  : dict  {model_name: fitted sklearn estimator}
    X_test  : test features
    y_test  : true test labels
    verbose : if False, suppresses terminal output

    Returns
    -------
    pandas.DataFrame  — rows = models, cols = [Accuracy, F1, Precision, Recall, ROC-AUC]

    Example
    -------
    >>> df = ml_evaluator.compare_metrics(models, X_test, y_test)
    >>> print(df["F1"].idxmax())
    """
    names   = list(models.keys())
    results = {n: compute_metrics(m, X_test, y_test) for n, m in models.items()}
    df      = _build_df(results, names)

    if verbose:
        _print_comparison(df)

    return _ResultDF(df)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  compare_interpret()  — plain-English interpretation per model, no plot
# ══════════════════════════════════════════════════════════════════════════════

def compare_interpret(
    models: Dict,
    X_test,
    y_test,
    *,
    verbose: bool = True,
    return_data: bool = False,
):
    """
    Print a plain-English interpretation for each model.
    No plot — text only.

    Parameters
    ----------
    models  : dict  {model_name: fitted sklearn estimator}
    X_test  : test features
    y_test  : true test labels
    verbose : if False, suppresses terminal output

    Returns
    -------
    dict  {model_name: interpretation_string}

    Example
    -------
    >>> texts = ml_evaluator.compare_interpret(models, X_test, y_test)
    """
    from .evaluation import _build_interpretation

    interpretations = {}
    for name, model in models.items():
        m    = compute_metrics(model, X_test, y_test)
        text = _build_interpretation(m)
        interpretations[name] = text

        if verbose:
            print(f"\n  ── {name} ──")
            for line in text.splitlines():
                print(f"     {line}")

    return _Result(interpretations)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  plot_confusion_matrices()  — confusion matrices only
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(
    models: Dict,
    X_test,
    y_test,
    *,
    class_labels: Optional[List[str]] = None,
    colors:       Optional[List[str]] = None,
    figsize:      Optional[tuple]     = None,
    save_path:    Optional[str]       = None,
) -> None:
    """
    Plot confusion matrices for all models — one subplot per model.

    Parameters
    ----------
    models       : dict  {model_name: fitted sklearn estimator}
    X_test       : test features
    y_test       : true test labels
    class_labels : axis labels — auto-inferred if None
    colors       : title colour per model
    figsize      : auto-sized if None
    save_path    : optional path to save

    Returns
    -------
    pandas.DataFrame  — metrics for all models

    Example
    -------
    >>> ml_evaluator.plot_confusion_matrices(models, X_test, y_test,
    ...     class_labels=["No", "Yes"])
    """
    names  = list(models.keys())
    n      = len(names)
    colors = colors or get_colors(n)

    if class_labels is None:
        class_labels = [str(c) for c in sorted(np.unique(y_test))]

    if figsize is None:
        figsize = (5 * n, 4.5)

    results = {name: compute_metrics(m, X_test, y_test)
               for name, m in models.items()}
    df = _build_df(results, names)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, name, color in zip(axes, names, colors):
        m    = results[name]
        cm   = confusion_matrix(y_test, m["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(
            f"{name}\nAcc={m['accuracy']:.3f}  F1={m['f1']:.3f}  AUC={m['roc_auc'] or 0:.3f}",
            fontsize=9, fontweight="bold", color=color,
        )
        add_panel_label(ax, "Confusion Matrix")

    plt.suptitle("Confusion Matrices", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  compare_roc_curves()  — overlaid ROC curves only
# ══════════════════════════════════════════════════════════════════════════════

def compare_roc_curves(
    models: Dict,
    X_test,
    y_test,
    *,
    colors:    Optional[List[str]] = None,
    figsize:   tuple               = (7, 6),
    save_path: Optional[str]       = None,
) -> None:
    """
    Plot overlaid ROC curves for all models.

    Parameters
    ----------
    models    : dict  {model_name: fitted sklearn estimator}
    X_test    : test features
    y_test    : true test labels
    colors    : one colour per model
    figsize   : figure size
    save_path : optional path to save

    Returns
    -------
    dict  {model_name: {"fpr", "tpr", "roc_auc"}}

    Example
    -------
    >>> roc = ml_evaluator.compare_roc_curves(models, X_test, y_test)
    >>> print(roc["Random Forest"]["roc_auc"])
    """
    names  = list(models.keys())
    colors = colors or get_colors(len(names))

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.28)

    roc_results = {}
    print("\n  ROC-AUC Scores")
    print("  " + "─" * 38)

    for name, color in zip(names, colors):
        m = compute_metrics(models[name], X_test, y_test)
        if m["y_prob"] is None:
            print(f"  {name}: predict_proba not available — skipped.")
            continue
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        roc_val     = auc(fpr, tpr)
        roc_results[name] = {"fpr": fpr, "tpr": tpr, "roc_auc": roc_val}

        ax.plot(fpr, tpr, color=color, lw=2.2,
                label=f"{name}  (AUC = {roc_val:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.05, color=color)
        print(f"  {name:<25}  AUC = {roc_val:.4f}  {_auc_grade(roc_val)}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate",  fontsize=10)
    ax.set_title("ROC Curves — Model Comparison",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    add_panel_label(ax, "ROC Curve Comparison")

    if roc_results:
        best  = max(roc_results, key=lambda k: roc_results[k]["roc_auc"])
        worst = min(roc_results, key=lambda k: roc_results[k]["roc_auc"])
        interp = (
            f"Best AUC: {best} ({roc_results[best]['roc_auc']:.3f})  |  "
            f"Lowest: {worst} ({roc_results[worst]['roc_auc']:.3f})\n"
            "The closer the curve hugs the top-left corner, the better."
        )
        styled_box(ax, interp, "#2E75B6", fontsize=8.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  plot_metrics_comparison()  — grouped bar chart only
# ══════════════════════════════════════════════════════════════════════════════

def plot_metrics_comparison(
    models: Dict,
    X_test,
    y_test,
    *,
    colors:    Optional[List[str]] = None,
    figsize:   tuple               = (11, 5),
    save_path: Optional[str]       = None,
) -> None:
    """
    Grouped bar chart comparing all metrics across models.
    No terminal output — plot only.

    Parameters
    ----------
    models    : dict  {model_name: fitted sklearn estimator}
    X_test    : test features
    y_test    : true test labels
    colors    : one colour per model
    figsize   : figure size
    save_path : optional path to save

    Returns
    -------
    pandas.DataFrame  — metrics for all models

    Example
    -------
    >>> ml_evaluator.plot_metrics_comparison(models, X_test, y_test)
    """
    names  = list(models.keys())
    colors = colors or get_colors(len(names))

    results = {n: compute_metrics(m, X_test, y_test) for n, m in models.items()}
    df      = _build_df(results, names)

    metric_cols = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]
    x     = np.arange(len(metric_cols))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = df.loc[name, metric_cols].values.astype(float)
        bars = ax.bar(x + i * width, vals, width,
                      label=name, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(metric_cols, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics",
                 fontweight="bold", fontsize=11)
    ax.axhline(0.5, color="#E74C3C", linestyle="--",
               lw=1, alpha=0.4, label="Random baseline")
    ax.legend(fontsize=8, loc="lower right",
              ncol=min(len(names) + 1, 5))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    add_panel_label(ax, "Metrics Comparison")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  comparison_dashboard()  — all-in-one shortcut
# ══════════════════════════════════════════════════════════════════════════════

def comparison_dashboard(
    models: Dict,
    X_test,
    y_test,
    *,
    class_labels: Optional[List[str]] = None,
    colors:       Optional[List[str]] = None,
    figsize:      Optional[tuple]     = None,
    save_path:    Optional[str]       = None,
) -> None:
    """
    Full comparison dashboard for multiple models.

    Row 1 — Confusion matrix per model
    Row 2 — ROC curves (overlaid)  +  Metrics bar chart
    Row 3 — Colour-coded summary table

    Also prints metrics + winner summary to the terminal.

    Parameters
    ----------
    models       : dict  {model_name: fitted sklearn estimator}
    X_test       : test features
    y_test       : true test labels
    class_labels : confusion matrix axis labels — auto-inferred if None
    colors       : one colour per model
    figsize      : auto-sized if None
    save_path    : optional path to save

    Returns
    -------
    pandas.DataFrame  — metrics for all models

    Example
    -------
    >>> df = ml_evaluator.comparison_dashboard(models, X_test, y_test)
    """
    names  = list(models.keys())
    n      = len(names)
    colors = colors or get_colors(n)

    if class_labels is None:
        class_labels = [str(c) for c in sorted(np.unique(y_test))]

    results = {name: compute_metrics(m, X_test, y_test)
               for name, m in models.items()}
    df = _build_df(results, names)

    _print_comparison(df)

    if figsize is None:
        figsize = (max(5 * n, 14), 14)

    fig = plt.figure(figsize=figsize)
    fig.suptitle("Model Comparison Dashboard",
                 fontsize=15, fontweight="bold", y=1.00)

    gs_r1 = gridspec.GridSpec(1, n, figure=fig,
                               left=0.04, right=0.98,
                               top=0.92,  bottom=0.62, wspace=0.30)
    gs_r2 = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.04, right=0.98,
                               top=0.55,  bottom=0.25, wspace=0.30)
    gs_r3 = gridspec.GridSpec(1, 1, figure=fig,
                               left=0.04, right=0.98,
                               top=0.18,  bottom=0.01)

    # Row 1 — Confusion matrices
    for col, (name, color) in enumerate(zip(names, colors)):
        ax  = fig.add_subplot(gs_r1[0, col])
        m   = results[name]
        cm  = confusion_matrix(y_test, m["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(
            f"{name}\nAcc={m['accuracy']:.3f}  F1={m['f1']:.3f}  AUC={m['roc_auc'] or 0:.3f}",
            fontsize=8, fontweight="bold",
        )
        add_panel_label(ax, "Confusion Matrix")

    fig.text(0.01, 0.94, "Row 1 — Confusion Matrices",
             fontsize=9, fontweight="bold", color="#444")

    # Row 2a — ROC curves
    ax_roc = fig.add_subplot(gs_r2[0, 0])
    for name, color in zip(names, colors):
        m = results[name]
        if m["y_prob"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        roc_val     = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC={roc_val:.3f})")
        ax_roc.fill_between(fpr, tpr, alpha=0.05, color=color)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax_roc.set_xlabel("FPR", fontsize=8)
    ax_roc.set_ylabel("TPR", fontsize=8)
    ax_roc.set_title("ROC Curves", fontweight="bold", fontsize=9)
    ax_roc.legend(loc="lower right", fontsize=7)
    ax_roc.grid(True, linestyle="--", alpha=0.4)
    add_panel_label(ax_roc, "B · ROC Curves")

    # Row 2b — Metrics bar
    ax_bar = fig.add_subplot(gs_r2[0, 1])
    metric_cols = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]
    x     = np.arange(len(metric_cols))
    width = 0.8 / n
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = df.loc[name, metric_cols].values.astype(float)
        bars = ax_bar.bar(x + i * width, vals, width,
                          label=name, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.2f}",
                        ha="center", va="bottom", fontsize=6.5)
    ax_bar.set_xticks(x + width * (n - 1) / 2)
    ax_bar.set_xticklabels(metric_cols, rotation=15, ha="right", fontsize=8)
    ax_bar.set_ylim(0, 1.2)
    ax_bar.set_title("Metrics Comparison", fontweight="bold", fontsize=9)
    ax_bar.legend(fontsize=7, loc="lower right")
    ax_bar.axhline(0.5, color="#E74C3C", linestyle="--", lw=1, alpha=0.4)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.35)
    add_panel_label(ax_bar, "C · Metrics Comparison")

    fig.text(0.01, 0.57, "Row 2 — ROC Curves & Metrics",
             fontsize=9, fontweight="bold", color="#444")

    # Row 3 — Summary table
    ax_tbl = fig.add_subplot(gs_r3[0, 0])
    ax_tbl.axis("off")
    _draw_summary_table(ax_tbl, df, colors)
    fig.text(0.01, 0.20, "Row 3 — Summary Table",
             fontsize=9, fontweight="bold", color="#444")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_df(results: dict, names: list) -> pd.DataFrame:
    rows = []
    for name in names:
        m = results[name]
        rows.append({
            "Model":     name,
            "Accuracy":  m["accuracy"],
            "F1":        m["f1"],
            "Precision": m["precision"],
            "Recall":    m["recall"],
            "ROC-AUC":   m["roc_auc"] or float("nan"),
        })
    return pd.DataFrame(rows).set_index("Model")


def _print_comparison(df: pd.DataFrame):
    metric_cols = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]
    print("\n" + "=" * 65)
    print("  Model Comparison")
    print("=" * 65)
    header = f"  {'Model':<22}" + "".join(f"{c:>10}" for c in metric_cols)
    print(header)
    print("  " + "─" * 62)
    for name, row in df.iterrows():
        vals = "".join(f"{row[c]:>10.4f}" for c in metric_cols)
        print(f"  {name:<22}{vals}")
    print()
    print("  Winners by metric:")
    for col in metric_cols:
        winner = df[col].idxmax()
        val    = df[col].max()
        print(f"    {col:<12} → {winner}  ({val:.4f})")

    # weighted overall recommendation
    df_norm = (df[metric_cols] - df[metric_cols].min()) / \
              (df[metric_cols].max() - df[metric_cols].min() + 1e-9)
    weights = {"Accuracy": 0.15, "F1": 0.30, "Precision": 0.15,
               "Recall": 0.20, "ROC-AUC": 0.20}
    scores  = sum(df_norm[c] * w for c, w in weights.items())
    best    = scores.idxmax()
    print(f"\n  ✅  Overall recommendation: {best}")
    print(f"      (weighted score — F1 & Recall weighted highest)\n")


def _auc_grade(val: float) -> str:
    if val >= 0.95: return "Excellent"
    if val >= 0.85: return "Good"
    if val >= 0.70: return "Fair"
    return "Poor"


def _draw_summary_table(ax, df: pd.DataFrame, colors: list):
    metric_cols = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]
    col_labels  = ["Model"] + metric_cols
    cell_text, cell_colors = [], []

    for name, row in df.iterrows():
        row_text   = [name]
        row_colors = ["#FFFFFF"]
        for col in metric_cols:
            val = row[col]
            row_text.append(f"{val:.4f}" if not np.isnan(val) else "—")
            rank = df[col].rank(pct=True)[name]
            row_colors.append(
                "#D6EFE0" if rank >= 0.75 else
                "#FAD7D7" if rank <= 0.25 else "#FFFFFF"
            )
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2E75B6")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    for i, color in enumerate(colors):
        tbl[(i + 1, 0)].set_facecolor(color + "33")
        tbl[(i + 1, 0)].set_text_props(fontweight="bold")
