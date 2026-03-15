"""
ml_evaluator — Model Evaluation Toolkit  v1.1
===============================================
Every piece of evaluation is its own function.

Single model — metrics & interpretation
----------------------------------------
    ml_evaluator.metrics(model, X_test, y_test)
    ml_evaluator.interpret(model, X_test, y_test)

Single model — individual plots
--------------------------------
    ml_evaluator.plot_confusion_matrix(model, X_test, y_test)
    ml_evaluator.plot_roc_curve(model, X_test, y_test)
    ml_evaluator.plot_metrics_bar(model, X_test, y_test)
    ml_evaluator.classification_report(model, X_test, y_test)

Single model — all-in-one shortcut
------------------------------------
    ml_evaluator.model_summary(model, X_test, y_test)

Bias–Variance — single model
------------------------------
    ml_evaluator.bv_stats(model, X_train, y_train)
    ml_evaluator.plot_learning_curve(model, X_train, y_train)
    ml_evaluator.bias_variance(model, X_train, y_train)

Multi-model — metrics & interpretation
----------------------------------------
    ml_evaluator.compare_metrics(models, X_test, y_test)
    ml_evaluator.compare_interpret(models, X_test, y_test)

Multi-model — individual plots
--------------------------------
    ml_evaluator.plot_confusion_matrices(models, X_test, y_test)
    ml_evaluator.compare_roc_curves(models, X_test, y_test)
    ml_evaluator.plot_metrics_comparison(models, X_test, y_test)

Multi-model — all-in-one shortcuts
------------------------------------
    ml_evaluator.comparison_dashboard(models, X_test, y_test)
    ml_evaluator.compare_bias_variance(models, X_train, y_train)
"""

from .evaluation import (
    metrics,
    interpret,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metrics_bar,
    classification_report,
    model_summary,
)

from .bias_variance import (
    bv_stats,
    plot_learning_curve,
    bias_variance,
    compare_bias_variance,
)

from .comparison import (
    compare_metrics,
    compare_interpret,
    plot_confusion_matrices,
    compare_roc_curves,
    plot_metrics_comparison,
    comparison_dashboard,
)

__version__ = "1.1.0"

__all__ = [
    # single model — metrics & interpretation
    "metrics",
    "interpret",
    # single model — plots
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_metrics_bar",
    "classification_report",
    # single model — shortcut
    "model_summary",
    # bias-variance — single model
    "bv_stats",
    "plot_learning_curve",
    "bias_variance",
    # multi-model — metrics & interpretation
    "compare_metrics",
    "compare_interpret",
    # multi-model — plots
    "plot_confusion_matrices",
    "compare_roc_curves",
    "plot_metrics_comparison",
    # multi-model — shortcuts
    "comparison_dashboard",
    "compare_bias_variance",
]
