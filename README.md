# ml-evaluator · Model Evaluation Toolkit

> Stop copy-pasting evaluation code. One import, one call.

```bash
pip install ml-evaluator
```

---

## What is it?

`ml-evaluator` turns model evaluation from a chore into a single function call.

Every function is **standalone** — pass your model, `X_test`, and `y_test` and you're done. Works for **binary and multiclass** problems automatically.

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
ev.metrics(model, X_test, y_test)                # numbers only, no plot
ev.interpret(model, X_test, y_test)              # plain-English interpretation, no plot
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
<summary>Example output — multiclass <code>ev.metrics()</code></summary>

```
=======================================================
  Model Summary — RF (4-class)
  Task: Multiclass (4 classes: 0, 1, 2, 3)
=======================================================
  Accuracy  : 0.8708
  F1        : 0.8524  (macro avg)
  Precision : 0.8601  (macro avg)
  Recall    : 0.8490  (macro avg)
  ROC-AUC   : 0.9712  (macro OvR)
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
│      Matrix         │   (OvR if multiclass│
├─────────────────────┼─────────────────────┤
│  C · Metrics        │  D · Classification │
│      Bar Chart      │      Report         │
└─────────────────────┴─────────────────────┘
```

Also prints metrics + interpretation to the terminal.

---

### 🔸 Bias–Variance — single model

```python
ev.bv_stats(model, X_train, y_train)           # stats + diagnosis only, no plot
ev.plot_learning_curve(model, X_train, y_train) # learning curve plot only
ev.bias_variance(model, X_train, y_train)      # stats + plot together
```

<details>
<summary>Example output — <code>ev.bv_stats()</code></summary>

```
  ✅ Random Forest
     Train Acc : 1.0000
     Val   Acc : 0.9400
     Gap       : 0.0600   (threshold: 0.10)
     Val Std   : 0.0151   (variance proxy)
     Diagnosis : Good Fit
     Train–val gap is 0.060 and val error is 0.060.
     The model generalises well.
     → Next step: evaluate on the held-out test set.
```

</details>

**Diagnosis logic:**

| Condition | Diagnosis | What it means |
|---|---|---|
| `train − val gap > 0.10` | 🔴 Overfit | Model memorises training data, fails to generalise |
| `1 − val_accuracy > 0.15` | 🟡 Underfit | Model too simple to capture the pattern |
| otherwise | ✅ Good Fit | Model generalises well |

---

### 🔸 Multiple models — text output

```python
ev.compare_metrics(models, X_test, y_test)     # metrics table + winner per metric
ev.compare_interpret(models, X_test, y_test)   # interpretation per model
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
    Recall       → Logistic Regression  (0.9877)

  ✅  Overall recommendation: Random Forest
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

---

### 🔺 Data utilities

```python
ev.is_imbalanced(y_train)
ev.is_imbalanced(y_train, threshold=0.15, class_labels=["Not Churned", "Churned"])
```

Detects class imbalance, prints a distribution table with a diagnosis, and draws a bar chart + pie chart.

<details>
<summary>Example output — <code>ev.is_imbalanced()</code></summary>

```
=======================================================
  Class Distribution
=======================================================
  Class                Count        %
  ──────────────────────────────────────
  0                      850    85.0%  █████████████████████
  1                      150    15.0%  ███

  Min / Max ratio : 0.176  (threshold: 0.20)
  🔴  Dataset is IMBALANCED

  ⚠️   Accuracy can be misleading on imbalanced data.
       → Use F1, Precision, Recall, or ROC-AUC instead.
       → Consider: class_weight='balanced', SMOTE, or threshold tuning.
```

</details>

---

## Multiclass support

All functions **automatically detect** binary vs. multiclass — no extra parameters needed.

| What changes | Binary | Multiclass |
|---|---|---|
| F1 / Precision / Recall | `average="binary"` | `average="macro"` |
| ROC-AUC | from `predict_proba[:, 1]` | One-vs-Rest, macro |
| ROC Curve | single curve | one per class + macro avg |
| Confusion Matrix | 2×2 | N×N, auto-scaled |
| Interpretation | precision vs recall balance | flags classes with low F1 |

---

## Controlling interpretation output

Every function that prints results accepts `show_interpretation=True/False`:

```python
# Print interpretation alongside metrics
ev.metrics(rf, X_test, y_test, show_interpretation=True)

# Skip interpretation — just the numbers
ev.bv_stats(rf, X_train, y_train, show_interpretation=False)

# model_summary without the interpretation block
ev.model_summary(rf, X_test, y_test, show_interpretation=False)
```

**Defaults:**

| Function | `show_interpretation` default |
|---|---|
| `model_summary`, `bv_stats`, `bias_variance`, `plot_roc_curve` | `True` |
| `metrics`, `plot_confusion_matrix`, `plot_metrics_bar`, `compare_metrics`, `comparison_dashboard` | `False` |

---

## All parameters

### Single model

```python
ev.metrics(model, X_test, y_test,
    model_name="Model",
    verbose=True,
    show_interpretation=False,
)

ev.plot_confusion_matrix(model, X_test, y_test,
    model_name="Model",
    class_labels=None,        # e.g. ["Not Churned", "Churned"]
    color="#2E75B6",
    figsize=None,             # auto-scaled based on number of classes
    show_interpretation=False,
    save_path=None,
)

ev.plot_roc_curve(model, X_test, y_test,
    model_name="Model",
    color="#2E75B6",
    figsize=(6, 5),
    show_interpretation=True,
    save_path=None,
)

ev.plot_metrics_bar(model, X_test, y_test,
    model_name="Model",
    color="#2E75B6",
    figsize=(7, 4),
    show_interpretation=False,
    save_path=None,
)

ev.model_summary(model, X_test, y_test,
    model_name="Model",
    class_labels=None,
    color="#2E75B6",
    figsize=(14, 10),
    show_interpretation=True,
    save_path=None,
)
```

### Bias–Variance

```python
ev.bv_stats(model, X_train, y_train,
    model_name="Model",
    n_splits=5,
    random_state=42,
    train_sizes=None,             # default: linspace(0.1, 1.0, 8)
    scoring="accuracy",
    overfit_threshold=0.10,
    high_bias_threshold=0.15,
    verbose=True,
    show_interpretation=True,
)

# bias_variance() and plot_learning_curve() accept the same parameters
```

### Multi-model functions

```python
# All multi-model functions accept:
models    = {"name": fitted_estimator, ...}   # required
colors    = None      # list of colours, one per model
figsize   = None      # auto-sized if not given
save_path = None

# comparison_dashboard and plot_confusion_matrices also accept:
class_labels         = None
show_interpretation  = False   # comparison_dashboard only

# compare_metrics also accepts:
show_interpretation  = False
```

### Data utilities

```python
ev.is_imbalanced(y,
    threshold=0.20,           # minority/majority ratio below which → imbalanced
    class_labels=None,
    show_interpretation=True,
    figsize=(10, 4),
    save_path=None,
)
```

---

## Return values

| Function | Returns |
|---|---|
| `metrics()` | `Result` dict — `accuracy`, `f1`, `precision`, `recall`, `roc_auc`, `y_pred`, `y_prob`, `y_prob_multi`, `classes`, `multiclass`, `averaging`, `report` |
| `interpret()` | `Result` — `{"text": ...}` |
| `bv_stats()` / `bias_variance()` | `Result` dict — `mean_train`, `mean_val`, `gap`, `bias_proxy`, `val_std`, `diagnosis`, `explanation`, `lc_sizes`, `lc_train`, `lc_val` |
| `compare_metrics()` | `Result` (DataFrame-like) — `df["F1"]`, `df["F1"].idxmax()`, etc. |
| `compare_interpret()` | `Result` — `{model_name: text}` |
| `compare_bias_variance()` | `Result` of Results — `cb["RF"]["diagnosis"]`, `cb["LR"]["gap"]`, etc. |
| all `plot_*` and `*_dashboard` functions | `None` |
| `is_imbalanced()` | `None` |

> **Note:** Result objects are silent in Jupyter — they never auto-display. Access data with `m["accuracy"]`, `bv["gap"]`, etc.

---

## Contributing

Issues and pull requests are welcome.

---

## License

MIT
