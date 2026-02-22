# ============================================================
# NOTEBOOK 3 — XGBoost Model Training & Hyperparameter Tuning
# ============================================================

# ── CELL 1: Imports ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import joblib
import warnings, os, sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
print("✅ Imports OK")

def finalize_plot():
    if 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close()

# ── CELL 2: Load Preprocessed Data ───────────────────────────
X_train = pd.read_csv('../data/X_train.csv')
X_val   = pd.read_csv('../data/X_val.csv')
X_test  = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv').squeeze()
y_val   = pd.read_csv('../data/y_val.csv').squeeze()
y_test  = pd.read_csv('../data/y_test.csv').squeeze()

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ── CELL 3: Baseline Model (Default XGBoost) ─────────────────
baseline = xgb.XGBRegressor(
    n_estimators=100, random_state=42,
    tree_method='hist', verbosity=0
)
baseline.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_pred_base = baseline.predict(X_val)
rmse_base = np.sqrt(mean_squared_error(y_val, y_pred_base))
r2_base   = r2_score(y_val, y_pred_base)
print(f"Baseline (default)  →  RMSE: {rmse_base:.4f}  |  R²: {r2_base:.4f}")

# ── CELL 4: Hyperparameter Tuning with Optuna ────────────────
def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 800),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 3.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'gamma':             trial.suggest_float('gamma', 0.0, 0.5),
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=False
    )
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

print("\n🔍 Running Optuna hyperparameter search (20 trials) ...")
print("   This takes about 1–2 minutes — go grab a coffee ☕")

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

best_params = study.best_params
best_rmse   = study.best_value
print(f"\n✅ Best validation RMSE: {best_rmse:.4f}")
print("Best hyperparameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ── CELL 5: Train Final Model with Best Params ───────────────
best_params.update({'tree_method': 'hist', 'random_state': 42, 'verbosity': 0})

model = xgb.XGBRegressor(**best_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=30,
    verbose=False
)
print(f"✅ Final model trained | Best iteration: {model.best_iteration}")

# ── CELL 6: Evaluation Metrics Function ──────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    val_true = np.expm1(y_true)
    val_pred = np.expm1(y_pred)
    mape = np.mean(np.abs((val_true - val_pred) / (val_true + 1e-9))) * 100
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  RMSE (log scale):  {rmse:.4f}")
    print(f"  MAE  (log scale):  {mae:.4f}")
    print(f"  R² Score:          {r2:.4f}")
    print(f"  MAPE:              {mape:.2f}%")
    print(f"{'─'*45}")
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

train_metrics = evaluate("TRAINING SET",   y_train, model.predict(X_train))
val_metrics   = evaluate("VALIDATION SET", y_val,   model.predict(X_val))
test_metrics  = evaluate("TEST SET ★",    y_test,   model.predict(X_test))

# ── CELL 7: Improvement Over Baseline ────────────────────────
improvement = (rmse_base - test_metrics['rmse']) / rmse_base * 100
print(f"\n📈 Improvement over baseline: {improvement:.1f}% RMSE reduction")

# ── CELL 8: Learning Curves ───────────────────────────────────
results = model.evals_result()
train_rmse_curve = results['validation_0']['rmse']
val_rmse_curve   = results['validation_1']['rmse']

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train_rmse_curve, label='Train RMSE', color='steelblue', linewidth=2)
ax.plot(val_rmse_curve,   label='Validation RMSE', color='darkorange', linewidth=2)
ax.axvline(model.best_iteration, color='red', linestyle='--', alpha=0.7,
           label=f'Best Iteration ({model.best_iteration})')
ax.set_title('XGBoost Learning Curves', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Trees (n_estimators)')
ax.set_ylabel('RMSE (log scale)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../data/fig_learning_curves.png', bbox_inches='tight')
finalize_plot()
print("📊 Figure saved: fig_learning_curves.png")

# ── CELL 9: Actual vs Predicted Plot ─────────────────────────
y_pred_test = model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted (log scale)
axes[0].scatter(y_test, y_pred_test, alpha=0.3, s=10, color='steelblue')
min_val, max_val = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
axes[0].set_title('Actual vs Predicted Market Value\n(log scale)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Actual log(Value)')
axes[0].set_ylabel('Predicted log(Value)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residuals
residuals = y_test - y_pred_test
axes[1].hist(residuals, bins=50, color='darkorange', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
axes[1].set_title('Residual Distribution\n(should be centred at 0)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Residual (Actual − Predicted)')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data/fig_predictions.png', bbox_inches='tight')
finalize_plot()
print("📊 Figure saved: fig_predictions.png")

# ── CELL 10: Metrics Summary Table ───────────────────────────
import pandas as pd
metrics_df = pd.DataFrame({
    'Metric':   ['RMSE (log)', 'MAE (log)', 'R² Score', 'MAPE (%)'],
    'Baseline': [rmse_base, mean_absolute_error(y_val, baseline.predict(X_val)),
                 r2_score(y_val, baseline.predict(X_val)), 'N/A'],
    'XGBoost (Tuned)': [
        f"{test_metrics['rmse']:.4f}",
        f"{test_metrics['mae']:.4f}",
        f"{test_metrics['r2']:.4f}",
        f"{test_metrics['mape']:.2f}%"
    ]
})
print("\n📊 Metrics Summary Table:")
print(metrics_df.to_string(index=False))

# ── CELL 11: Save Model ───────────────────────────────────────
os.makedirs('../models', exist_ok=True)
model.save_model('../models/xgb_model.json')
joblib.dump(best_params, '../models/best_params.pkl')

print("\n✅ Model saved: models/xgb_model.json")
print("✅ Best params saved: models/best_params.pkl")
print("\n➡️  Next: Run notebook 04_evaluation.py")
