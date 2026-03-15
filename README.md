<div align="center">

# ml-evaluator

**Stop copy-pasting evaluation code.**

[![PyPI version](https://img.shields.io/pypi/v/ml-evaluator?color=2E75B6&labelColor=1a1a2e&style=for-the-badge)](https://pypi.org/project/ml-evaluator/)
[![Python](https://img.shields.io/pypi/pyversions/ml-evaluator?color=2ECC71&labelColor=1a1a2e&style=for-the-badge)](https://pypi.org/project/ml-evaluator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-E67E22?labelColor=1a1a2e&style=for-the-badge)](LICENSE)

```bash
pip install ml-evaluator
```

</div>

---

## What is it?

`ml-evaluator` turns model evaluation from a chore into a single function call.

Every function is **standalone** — pass your model, `X_test`, and `y_test` and you're done. No pipelines to build, no intermediate objects to store, no boilerplate to copy.

Built for people who:
- know how to train a model but want clean, reproducible evaluation
- are learning ML and want plots + **plain-English interpretation**, not just numbers
- are tired of writing the same confusion-matrix / ROC / bias-variance code every project

---

## Install

```bash
pip install ml-evaluator
```

```python
import ml_evaluator as ev
```

> **Requirements:** Python 3.8+  ·  numpy  ·  pandas  ·  matplotlib  ·  scikit-learn

---

## API

Every piece of evaluation is its own function — use exactly what you need.

### 🔹 Single model — text output

```python
ev.metrics(model, X_test, y_test)           # numbers only, no plot
ev.interpret(model, X_test, y_test)         # plain-English interpretation, no plot
ev.classification_report(model, X_test, y_test)  # classification report table, no plot
```

<details>
<summary>Example output — <code>ev.metrics()</code></summary>

```
=======================================================
  Model Summary — Random Forest
=======================================================
  Accuracy  : 0.9625
  F1        : 0.9634
  Precision : 0.9518
  Recall    : 0.9753
  ROC-AUC   : 0.9930
```

</details>

<details>
<summary>Example output — <code>ev.interpret()</code></summary>

```
  Interpretation — Random Forest
  ─────────────────────────────────────────────────────
  ✅  Accuracy 0.963 — high overall correctness.
  ✅  F1 0.963 — strong balance between precision and recall.
  ✅  Precision (0.952) ≈ Recall (0.975) — well balanced.
       AUC = 0.993 — Excellent. The model clearly separates the two classes.
```

</details>

<details>
<summary>Example output — <code>ev.classification_report()</code></summary>

```
  Classification Report — Random Forest
  ────────────────────────────────────────────────────────
  Class             Precision   Recall  F1-Score  Support
  ────────────────────────────────────────────────────────
  0                     0.970    0.952     0.961      100
  1                     0.953    0.971     0.962      100
  ────────────────────────────────────────────────────────
  Accuracy                                 0.963      200
  macro avg             0.961    0.962     0.961
  weighted avg          0.961    0.962     0.961
```

</details>

---

### 🔹 Single model — individual plots

```python
ev.plot_confusion_matrix(model, X_test, y_test)   # confusion matrix only
ev.plot_roc_curve(model, X_test, y_test)           # ROC curve only
ev.plot_metrics_bar(model, X_test, y_test)         # metrics bar chart only
```

---

### 🔹 Single model — all-in-one

```python
ev.model_summary(model, X_test, y_test)
```

Produces a **2×2 dashboard** in one figure:

```
┌─────────────────────┬─────────────────────┐
│  A · Confusion      │  B · ROC Curve      │
│      Matrix         │                     │
├─────────────────────┼─────────────────────┤
│  C · Metrics        │  D · Classification │
│      Bar Chart      │      Report         │
└─────────────────────┴─────────────────────┘
```

Also prints metrics + interpretation to the terminal.

---

### 🔸 Bias–Variance — single model

```python
ev.bv_stats(model, X_train, y_train)          # stats + diagnosis only, no plot
ev.plot_learning_curve(model, X_train, y_train)   # learning curve plot only
ev.bias_variance(model, X_train, y_train)     # stats + plot together
```

<details>
<summary>Example output — <code>ev.bv_stats()</code></summary>

```
  ✅ Random Forest
     Train Acc : 1.0000
     Val   Acc : 0.9359
     Gap       : 0.0641   (threshold: 0.10)
     Val Std   : 0.0167   (variance proxy)
     Diagnosis : Good Fit
     Train–val gap is 0.064 and val error is 0.064.
     The model generalises well — no strong signs of overfit or underfit.
     → Next step: evaluate on the held-out test set.
```

</details>

**Diagnosis logic:**

| Condition | Diagnosis | What it means |
|---|---|---|
| `train − val gap > 0.10` | 🔴 Overfit | Model memorises training data, fails to generalise |
| `1 − val_accuracy > 0.15` | 🟡 Underfit | Model too simple to capture the pattern |
| otherwise | ✅ Good Fit | Model generalises well |

Thresholds are configurable:

```python
ev.bias_variance(
    model, X_train, y_train,
    overfit_threshold=0.05,      # stricter
    high_bias_threshold=0.10,
)
```

---

### 🔸 Multiple models — text output

```python
ev.compare_metrics(models, X_test, y_test)     # metrics table + winner per metric, no plot
ev.compare_interpret(models, X_test, y_test)   # interpretation per model, no plot
```

<details>
<summary>Example output — <code>ev.compare_metrics()</code></summary>

```
=================================================================
  Model Comparison
=================================================================
  Model                   Accuracy        F1 Precision    Recall   ROC-AUC
  ──────────────────────────────────────────────────────────────
  Random Forest             0.9625    0.9634    0.9518    0.9753    0.9930
  Logistic Regression       0.9375    0.9412    0.8989    0.9877    0.9833

  Winners by metric:
    Accuracy     → Random Forest        (0.9625)
    F1           → Random Forest        (0.9634)
    Precision    → Random Forest        (0.9518)
    Recall       → Logistic Regression  (0.9877)
    ROC-AUC      → Random Forest        (0.9930)

  ✅  Overall recommendation: Random Forest
      (weighted score — F1 & Recall weighted highest)
```

</details>

---

### 🔸 Multiple models — individual plots

```python
ev.plot_confusion_matrices(models, X_test, y_test)   # one matrix per model
ev.compare_roc_curves(models, X_test, y_test)         # overlaid ROC curves
ev.plot_metrics_comparison(models, X_test, y_test)    # grouped bar chart
```

---

### 🔸 Multiple models — all-in-one shortcuts

```python
ev.comparison_dashboard(models, X_test, y_test)
ev.compare_bias_variance(models, X_train, y_train)
```

`comparison_dashboard` — full **3-row dashboard**:
```
Row 1 — Confusion matrix per model
Row 2 — Overlaid ROC curves  +  Metrics bar chart
Row 3 — Colour-coded summary table (green = top, red = bottom)
```

`compare_bias_variance` — two-row B-V dashboard:
```
Row 1 — Learning curve per model with diagnosis label
Row 2 — Bias proxy  ·  Variance  ·  Overfitting gap comparison
```

---

## All parameters

### Single model functions

```python
ev.metrics(model, X_test, y_test, model_name="Model", verbose=True)
ev.interpret(model, X_test, y_test, model_name="Model", verbose=True)
ev.classification_report(model, X_test, y_test, model_name="Model", class_labels=None)

ev.plot_confusion_matrix(model, X_test, y_test,
    model_name="Model",
    class_labels=None,       # e.g. ["Not Churned", "Churned"]
    color="#2E75B6",
    figsize=(5, 4.5),
    save_path=None,
)

ev.plot_roc_curve(model, X_test, y_test,
    model_name="Model",
    color="#2E75B6",
    figsize=(6, 5),
    save_path=None,
)

ev.plot_metrics_bar(model, X_test, y_test,
    model_name="Model",
    color="#2E75B6",
    figsize=(7, 4),
    save_path=None,
)

ev.model_summary(model, X_test, y_test,
    model_name="Model",
    class_labels=None,
    color="#2E75B6",
    figsize=(14, 10),
    save_path=None,
)
```

### Bias–Variance functions

```python
ev.bv_stats(model, X_train, y_train,
    model_name="Model",
    n_splits=5,
    random_state=42,
    train_sizes=None,            # default: linspace(0.1, 1.0, 8)
    scoring="accuracy",
    overfit_threshold=0.10,
    high_bias_threshold=0.15,
    verbose=True,
)

ev.plot_learning_curve(model, X_train, y_train,
    model_name="Model",
    n_splits=5,
    random_state=42,
    train_sizes=None,
    scoring="accuracy",
    overfit_threshold=0.10,
    high_bias_threshold=0.15,
    color="#2E75B6",
    figsize=(8, 5),
    save_path=None,
)

# bias_variance() accepts all parameters above
```

### Multi-model functions

```python
# All multi-model functions accept:
models    = {"name": fitted_estimator, ...}   # required
colors    = None     # list of colours, one per model
figsize   = None     # auto-sized if not given
save_path = None     # saves figure to this path

# compare_metrics / compare_interpret also accept:
verbose   = True

# plot_confusion_matrices / comparison_dashboard also accept:
class_labels = None  # e.g. ["No", "Yes"]

# compare_bias_variance also accepts all bv_stats parameters
```

---

## Return values

| Function | Returns |
|---|---|
| `metrics()` | `dict` — accuracy, f1, precision, recall, roc_auc, y_pred, y_prob, report |
| `interpret()` | `str` — full interpretation text |
| `bv_stats()` | `dict` — lc_sizes, lc_train, lc_val, mean_train, mean_val, val_std, gap, bias_proxy, diagnosis, explanation |
| `compare_metrics()` | `pandas.DataFrame` — one row per model |
| `compare_interpret()` | `dict` — {model_name: interpretation_string} |
| all `plot_*` functions | `None` |
| `model_summary()` | `None` |
| `comparison_dashboard()` | `None` |
| `bias_variance()` | `None` |
| `compare_bias_variance()` | `None` |

---

## Roadmap

- [ ] v1.2 — Threshold optimiser (best decision threshold for F1 / Recall)
- [ ] v1.3 — Probability calibration plot (reliability diagram)
- [ ] v1.4 — Feature importance comparison across models
- [ ] v2.0 — Auto PDF report

---

## Contributing

Issues and pull requests are welcome.

---

## License

MIT
