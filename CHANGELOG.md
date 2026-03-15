# Changelog

All notable changes to mleval will be documented here.

## [1.0.0] — 2025

### Added
- `model_summary(model, X_test, y_test)` — full single-model dashboard:
  confusion matrix, ROC curve, metrics bar chart, classification report,
  and plain-English interpretation printed to terminal and rendered inside the figure.
- `plot_roc_curve(model, X_test, y_test)` — standalone ROC curve with
  AUC interpretation box.
- `bias_variance(model, X_train, y_train)` — learning-curve analysis for one
  model with automatic Good Fit / Underfit / Overfit diagnosis, explanation,
  and actionable recommendations.
- `compare_models(models, X_test, y_test)` — grouped bar chart comparing
  Accuracy, F1, Precision, Recall and ROC-AUC across multiple models, with
  winner-per-metric and weighted overall recommendation.
- `compare_roc_curves(models, X_test, y_test)` — overlaid ROC curves for all
  models with AUC grades.
- `comparison_dashboard(models, X_test, y_test)` — full multi-model dashboard:
  confusion matrices (Row 1), ROC + metrics comparison (Row 2), colour-coded
  summary table (Row 3).
- `compare_bias_variance(models, X_train, y_train)` — two-row dashboard with
  per-model learning curves and a three-panel Bias / Variance / Gap comparison.
