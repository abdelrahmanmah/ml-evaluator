"""
ml_evaluator.evaluation
========================
Single-model evaluation — every piece is its own function.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)

from ._utils import DEFAULT_COLORS, compute_metrics, styled_box, add_panel_label, _Result

warnings.filterwarnings("ignore")

__all__ = [
    "metrics", "interpret",
    "plot_confusion_matrix", "plot_roc_curve",
    "plot_metrics_bar", "classification_report",
    "model_summary",
]


# ── 1. metrics() ──────────────────────────────────────────────────────────────

def metrics(model, X_test, y_test, *, model_name="Model", verbose=True):
    """Compute and print metrics. No plot.
    Returns a Result object — access data with m["accuracy"], m["f1"], etc."""
    m = compute_metrics(model, X_test, y_test)
    if verbose:
        _print_metrics(model_name, m)
    return _Result(m)


# ── 2. interpret() ────────────────────────────────────────────────────────────

def interpret(model, X_test, y_test, *, model_name="Model", verbose=True):
    """Print plain-English interpretation. No plot.
    Returns a Result object — access text with r.text or str(r)."""
    m    = compute_metrics(model, X_test, y_test)
    text = _build_interpretation(m)
    if verbose:
        print(f"\n  Interpretation — {model_name}")
        print("  " + "─" * 53)
        for line in text.splitlines():
            print(f"  {line}")
        print()
    return _Result({"text": text})


# ── 3. plot_confusion_matrix() ────────────────────────────────────────────────

def plot_confusion_matrix(
    model, X_test, y_test, *,
    model_name="Model", class_labels=None,
    color=DEFAULT_COLORS[0], figsize=(5, 4.5), save_path=None,
) -> None:
    """Plot confusion matrix. Returns None."""
    m = compute_metrics(model, X_test, y_test)

    if class_labels is None:
        class_labels = [str(c) for c in sorted(np.unique(y_test))]

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(top=0.85)

    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, m["y_pred"]),
                                  display_labels=class_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    # Two-line title: name on top, metrics below — no overlap
    ax.set_title(
        f"{model_name}\nAcc={m['accuracy']:.3f}  F1={m['f1']:.3f}  "
        f"AUC={m['roc_auc'] or 0:.3f}",
        fontsize=10, fontweight="bold", color=color, pad=12,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ── 4. plot_roc_curve() ───────────────────────────────────────────────────────

def plot_roc_curve(
    model, X_test, y_test, *,
    model_name="Model", color=DEFAULT_COLORS[0],
    figsize=(6, 5), save_path=None,
) -> None:
    """Plot ROC curve with AUC interpretation box. Returns None."""
    m = compute_metrics(model, X_test, y_test)

    if m["y_prob"] is None:
        print("  ⚠️  predict_proba not available — ROC curve cannot be plotted.")
        return

    fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
    roc_val     = auc(fpr, tpr)
    interp      = _interpret_auc(roc_val)

    print(f"\n  ROC-AUC ({model_name}): {roc_val:.4f}")
    print(f"  {interp}\n")

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.28)

    ax.plot(fpr, tpr, color=color, lw=2.5,
            label=f"{model_name}  (AUC = {roc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=color)

    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curve — {model_name}", fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    styled_box(ax, interp, color, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ── 5. plot_metrics_bar() ─────────────────────────────────────────────────────

def plot_metrics_bar(
    model, X_test, y_test, *,
    model_name="Model", color=DEFAULT_COLORS[0],
    figsize=(7, 4), save_path=None,
) -> None:
    """Bar chart of all metrics. Returns None."""
    m = compute_metrics(model, X_test, y_test)

    labels = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]
    values = [m["accuracy"], m["f1"], m["precision"],
              m["recall"],   m["roc_auc"] or 0]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(labels, values, color=color, alpha=0.82,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.18)
    ax.set_title(f"Performance Metrics — {model_name}",
                 fontweight="bold", fontsize=11)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(0.5, color="#E74C3C", linestyle="--",
               lw=1.2, alpha=0.5, label="Random baseline (0.5)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ── 6. classification_report() ───────────────────────────────────────────────

def classification_report(
    model, X_test, y_test, *,
    model_name="Model", class_labels=None,
) -> None:
    """
    Print classification report as a clean table in the terminal.
    No plot produced.
    """
    m = compute_metrics(model, X_test, y_test)

    if class_labels is None:
        class_labels = [str(c) for c in sorted(np.unique(y_test))]

    rep  = m["report"]
    keys = [k for k in rep if k not in ("accuracy", "macro avg", "weighted avg")]

    print(f"\n  Classification Report — {model_name}")
    print("  " + "─" * 56)
    print(f"  {'Class':<16} {'Precision':>9} {'Recall':>8} {'F1-Score':>9} {'Support':>8}")
    print("  " + "─" * 56)

    for k in keys:
        label = (class_labels[int(k)]
                 if k.isdigit() and int(k) < len(class_labels) else k)
        row = rep[k]
        print(f"  {label:<16} {row['precision']:>9.3f} {row['recall']:>8.3f} "
              f"{row['f1-score']:>9.3f} {int(row['support']):>8}")

    total = int(sum(rep[k]["support"] for k in keys))
    print("  " + "─" * 56)
    print(f"  {'Accuracy':<16} {'':>9} {'':>8} {m['accuracy']:>9.3f} {total:>8}")
    for avg in ("macro avg", "weighted avg"):
        row = rep[avg]
        print(f"  {avg:<16} {row['precision']:>9.3f} {row['recall']:>8.3f} "
              f"{row['f1-score']:>9.3f}")
    print()


# ── 7. model_summary() ───────────────────────────────────────────────────────

def model_summary(
    model, X_test, y_test, *,
    model_name="Model", class_labels=None,
    color=DEFAULT_COLORS[0], figsize=(14, 10), save_path=None,
) -> None:
    """
    Full 2×2 dashboard: Confusion Matrix + ROC + Metrics Bar + Classification Report.
    Also prints metrics and interpretation to the terminal. Returns None.
    """
    m = compute_metrics(model, X_test, y_test)

    if class_labels is None:
        class_labels = [str(c) for c in sorted(np.unique(y_test))]

    # Terminal
    _print_metrics(model_name, m)
    text = _build_interpretation(m)
    print(f"  Interpretation — {model_name}")
    print("  " + "─" * 53)
    for line in text.splitlines():
        print(f"  {line}")
    print()

    # Figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"Model Summary — {model_name}",
                 fontsize=14, fontweight="bold", y=0.99)

    gs     = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.38)
    ax_cm  = fig.add_subplot(gs[0, 0])
    ax_roc = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_rep = fig.add_subplot(gs[1, 1])

    # A — Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, m["y_pred"]),
                                  display_labels=class_labels)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title(f"Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}",
                    fontsize=9, color="#555555", pad=8)
    add_panel_label(ax_cm, "A · Confusion Matrix")

    # B — ROC Curve
    if m["y_prob"] is not None:
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        roc_val     = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"AUC = {roc_val:.3f}")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax_roc.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax_roc.legend(fontsize=8, loc="lower right")
    else:
        ax_roc.text(0.5, 0.5, "predict_proba\nnot available",
                    ha="center", va="center", transform=ax_roc.transAxes)
    ax_roc.set_xlabel("FPR", fontsize=8)
    ax_roc.set_ylabel("TPR", fontsize=8)
    ax_roc.set_title("ROC Curve", fontsize=9, pad=8)
    ax_roc.grid(True, linestyle="--", alpha=0.4)
    add_panel_label(ax_roc, "B · ROC Curve")

    # C — Metrics bar
    bar_labels = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]
    bar_values = [m["accuracy"], m["f1"], m["precision"],
                  m["recall"],   m["roc_auc"] or 0]
    bars = ax_bar.bar(bar_labels, bar_values,
                      color=color, alpha=0.82, edgecolor="white")
    for bar, val in zip(bars, bar_values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_xticklabels(bar_labels, rotation=15, ha="right", fontsize=8)
    ax_bar.axhline(0.5, color="#E74C3C", linestyle="--", lw=1, alpha=0.4)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    ax_bar.set_title("Metrics", fontsize=9, pad=8)
    add_panel_label(ax_bar, "C · Metrics")

    # D — Classification report text panel
    rep  = m["report"]
    keys = [k for k in rep if k not in ("accuracy", "macro avg", "weighted avg")]
    lines = [f"{'Class':<14} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Sup':>6}",
             "─" * 43]
    for k in keys:
        lbl = (class_labels[int(k)]
               if k.isdigit() and int(k) < len(class_labels) else k)
        r = rep[k]
        lines.append(f"{lbl:<14} {r['precision']:>6.3f} {r['recall']:>7.3f} "
                     f"{r['f1-score']:>6.3f} {int(r['support']):>6}")
    lines += ["─" * 43,
              f"{'Accuracy':<14} {'':>6} {'':>7} {m['accuracy']:>6.3f}"]
    for avg in ("macro avg", "weighted avg"):
        r = rep[avg]
        lines.append(f"{avg:<14} {r['precision']:>6.3f} {r['recall']:>7.3f} "
                     f"{r['f1-score']:>6.3f}")

    ax_rep.text(0.04, 0.92, "\n".join(lines),
                transform=ax_rep.transAxes,
                fontsize=8.5, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="#F8F9FA", edgecolor="#CCCCCC"))
    ax_rep.axis("off")
    ax_rep.set_title("Classification Report", fontsize=9, pad=8)
    add_panel_label(ax_rep, "D · Classification Report")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _print_metrics(model_name, m):
    print("\n" + "=" * 55)
    print(f"  Model Summary — {model_name}")
    print("=" * 55)
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    if m["roc_auc"]:
        print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
    print()


def _build_interpretation(m):
    lines = []
    acc = m["accuracy"]
    lines.append(
        f"✅  Accuracy {acc:.3f} — high overall correctness." if acc >= 0.90 else
        f"⚠️   Accuracy {acc:.3f} — acceptable but worth improving." if acc >= 0.75 else
        f"🔴  Accuracy {acc:.3f} — low. Check for class imbalance or weak features."
    )
    f1 = m["f1"]
    lines.append(
        f"✅  F1 {f1:.3f} — strong balance between precision and recall." if f1 >= 0.85 else
        f"⚠️   F1 {f1:.3f} — moderate. Consider threshold tuning." if f1 >= 0.65 else
        f"🔴  F1 {f1:.3f} — low. The model struggles with the positive class."
    )
    p, r = m["precision"], m["recall"]
    if abs(p - r) > 0.15:
        lines.append(
            f"⚠️   Precision ({p:.3f}) >> Recall ({r:.3f}) — "
            "model is conservative: misses positives to avoid false alarms."
            if p > r else
            f"⚠️   Recall ({r:.3f}) >> Precision ({p:.3f}) — "
            "model catches most positives but raises many false alarms."
        )
    else:
        lines.append(f"✅  Precision ({p:.3f}) ≈ Recall ({r:.3f}) — well balanced.")
    if m["roc_auc"] is not None:
        lines.append(f"     {_interpret_auc(m['roc_auc'])}")
    return "\n".join(lines)


def _interpret_auc(val):
    if val >= 0.95:
        return f"AUC = {val:.3f} — Excellent. The model clearly separates the two classes."
    if val >= 0.85:
        return f"AUC = {val:.3f} — Good. Works well but there is room to improve."
    if val >= 0.70:
        return f"AUC = {val:.3f} — Fair. Better than random but may struggle on hard cases."
    return f"AUC = {val:.3f} — Poor. Barely better than random. Consider feature engineering."
