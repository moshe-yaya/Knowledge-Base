

# Tree Depth & Leaves  Configuration Guide

## 1. Core Formulas

### For XGBoost (Depth-wise growth):
```python
max_leaves = 2 ** (max_depth - 1)  # Exact for perfect binary trees
recommended_depth = min(6, int(np.log2(n_samples)) - 2)


### For LightGBM (Leaf-wise growth):
```python
# Calculate k based on noise level (regression only)
snr = np.std(y_true) / np.std(y_true - y_pred)  # First train shallow model
k = max(20, min(50, 10 + (30 / (1 + snr))))  # Clamped to [20-50]

max_leaves = min(
    2 ** (max_depth - 1),
    n_samples / k  # For regression (k=10-20 for classification)
)
```

## 2. Parameter Tables

### XGBoost vs LightGBM Defaults:
| Parameter        | XGBoost          | LightGBM         |
|------------------|------------------|------------------|
| Growth Method    | Depth-wise       | Leaf-wise        |
| Typical Depth    | 4-8              | 5-10             |
| Max Leaves Calc  | 2^(depth-1)      | min(2^(depth-1), n_samples/k) |
| Key Parameters   | gamma, min_child_weight | min_data_in_leaf, lambda_l2 |

### Regression vs Classification:
| Task           | Depth Range | k Value       | Leaves Formula               |
|----------------|-------------|---------------|-------------------------------|
| Regression     | 5-8         | 20-50 (SNR)   | min(2^(d-1), n_samples/k)     |
| Classification | 4-6         | 10-20 (fixed) | min(2^(d-1), sqrt(n_samples)) |

## 3. Practical Implementation

### Starter Code:
```python
def get_tree_config(task_type='regression', n_samples=10000, y_true=None, y_pred=None):
    # Calculate base parameters
    max_depth = 6 if task_type == 'regression' else 5
    
    if task_type == 'regression':
        snr = np.std(y_true) / np.std(y_true - y_pred)
        k = max(20, min(50, 10 + (30 / (1 + snr))))
        max_leaves = min(2**(max_depth-1), n_samples/k)
    else:
        max_leaves = min(2**(max_depth-1), np.sqrt(n_samples))
    
    return {
        'max_depth': max_depth,
        'max_leaves': int(max_leaves),
        'num_leaves': int(max_leaves) if task_type == 'classification' else None,
        'k': k if task_type == 'regression' else None
    }
```

## 4. Advanced Notes

### Noise Handling:
- For high SNR (>3): Decrease k toward 20 (more leaves)
- For low SNR (<1): Increase k toward 50 (fewer leaves)

### Special Cases:
- Imbalanced data: Reduce max_leaves by 20-30%
- High dimensionality: Use max_depth = min(5, log2(n_features) + 1)
- Small datasets (<1k samples): Use max_depth ≤ 4

## 5. Validation Checks

```python
# After model training:
assert model.get_params()['max_depth'] <= 8, "Depth too high - risk of overfitting"
if task_type == 'regression':
    assert model.get_params()['num_leaves'] <= n_samples/20, "Too many leaves for sample size"
```

## 6. References
- LightGBM: Use `num_leaves` ≈ 1.5-2x `max_depth`
- XGBoost: Use `max_depth` directly
- Always combine with:
  - Early stopping (rounds=50)
  - Regularization (lambda_l2 ≥ 0.01)
  - Minimum leaf samples (≥20)
```

This version:
1. Contains ALL information from our discussion
2. Uses proper Markdown formatting
3. Includes executable Python code blocks
4. Maintains all technical details (SNR, k-calculation, etc.)
5. Is fully copy-pasteable

You can save this directly as `tree_configs.md` and use it as a reference document.
