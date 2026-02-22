# ============================================================
# NOTEBOOK 4 — Model Evaluation (Detailed Plots & Analysis)
# ============================================================

# ── CELL 1: Imports ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings, sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
print("✅ Imports OK")

def finalize_plot():
    # Show plots in notebooks, close when running as a script
    if 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close()

# ── CELL 2: Load Data & Model ────────────────────────────────
X_test       = pd.read_csv('../data/X_test.csv')
X_test_raw   = pd.read_csv('../data/X_test_raw.csv')   # unscaled
y_test       = pd.read_csv('../data/y_test.csv').squeeze()
player_names = pd.read_csv('../data/player_names_test.csv').squeeze()

model = xgb.XGBRegressor()
model.load_model('../models/xgb_model.json')

y_pred = model.predict(X_test)
print(f"✅ Loaded {len(X_test):,} test samples")

# ── CELL 3: Core Metrics ─────────────────────────────────────
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

# Back-transform for interpretable Euro value error (in € millions)
val_true_m = np.expm1(y_test) / 1e6
val_pred_m = np.expm1(y_pred) / 1e6
mae_eur    = mean_absolute_error(val_true_m, val_pred_m)
mape       = np.mean(np.abs((val_true_m - val_pred_m) / (val_true_m + 1e-9))) * 100

print(f"\n{'═'*45}")
print(f"  FINAL TEST SET EVALUATION")
print(f"{'═'*45}")
print(f"  RMSE (log scale):      {rmse:.4f}")
print(f"  MAE  (log scale):      {mae:.4f}")
print(f"  R² Score:              {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"  MAE  (€ millions):     €{mae_eur:.2f}M")
print(f"  MAPE:                  {mape:.2f}%")
print(f"{'═'*45}")

# ── CELL 4: Actual vs Predicted (€ Millions) ─────────────────
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(val_true_m, val_pred_m, alpha=0.25, s=12,
                     c=val_true_m, cmap='viridis', norm=plt.Normalize(0, 80))
plt.colorbar(scatter, ax=ax, label='Actual Value (€M)')

lim = max(val_true_m.quantile(0.98), val_pred_m.max())
ax.plot([0, lim], [0, lim], 'r--', linewidth=2, label='Perfect prediction')
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_title('Actual vs Predicted Transfer Value (€ Millions)', fontsize=14, fontweight='bold')
ax.set_xlabel('Actual Market Value (€M)')
ax.set_ylabel('Predicted Market Value (€M)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Annotate R²
ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = €{mae_eur:.1f}M\nMAPE = {mape:.1f}%',
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
plt.savefig('../data/fig_actual_vs_predicted_eur.png', bbox_inches='tight')
finalize_plot()

# ── CELL 5: Residual Analysis ─────────────────────────────────
residuals = y_test.values - y_pred

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Histogram
axes[0].hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].axvline(residuals.mean(), color='orange', linestyle='-', linewidth=1.5,
                label=f'Mean = {residuals.mean():.3f}')
axes[0].set_title('Residual Distribution', fontweight='bold')
axes[0].set_xlabel('Residual')
axes[0].set_ylabel('Count')
axes[0].legend()

# Residuals vs Predicted
axes[1].scatter(y_pred, residuals, alpha=0.2, s=8, color='steelblue')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Residuals vs Predicted', fontweight='bold')
axes[1].set_xlabel('Predicted log(Value)')
axes[1].set_ylabel('Residual')

# Q-Q Plot (normality check)
from scipy import stats
(osm, osr), (slope, intercept, r) = stats.probplot(residuals)
axes[2].scatter(osm, osr, alpha=0.3, s=8, color='steelblue')
axes[2].plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2)
axes[2].set_title('Q-Q Plot (Normality of Residuals)', fontweight='bold')
axes[2].set_xlabel('Theoretical Quantiles')
axes[2].set_ylabel('Sample Quantiles')

plt.tight_layout()
plt.savefig('../data/fig_residual_analysis.png', bbox_inches='tight')
finalize_plot()

# ── CELL 6: Prediction Error by Position ─────────────────────
position_cols = [c for c in X_test.columns if c.startswith('position_group_')]
if position_cols:
    pos_labels = []
    for _, row in X_test[position_cols].iterrows():
        if row.max() == 0:
            pos_labels.append('MID')
        else:
            pos_labels.append(row.idxmax().replace('position_group_', ''))
    positions = pd.Series(pos_labels)
else:
    positions = pd.Series(['Unknown'] * len(X_test))

error_df = pd.DataFrame({
    'position': positions,
    'abs_error_log': np.abs(residuals),
    'abs_error_eur': np.abs(np.asarray(val_true_m) - np.asarray(val_pred_m))
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
error_df.groupby('position')['abs_error_log'].median().sort_values().plot(
    kind='barh', ax=axes[0], color='steelblue', edgecolor='white'
)
axes[0].set_title('Median Absolute Error by Position\n(log scale)', fontweight='bold')
axes[0].set_xlabel('Median |Residual|')

error_df.groupby('position')['abs_error_eur'].median().sort_values().plot(
    kind='barh', ax=axes[1], color='darkorange', edgecolor='white'
)
axes[1].set_title('Median Absolute Error by Position\n(€ Millions)', fontweight='bold')
axes[1].set_xlabel('Median |Error| (€M)')
plt.tight_layout()
plt.savefig('../data/fig_error_by_position.png', bbox_inches='tight')
finalize_plot()

# ── CELL 7: Top Predicted Players ────────────────────────────
result_df = pd.DataFrame({
    'player': player_names.values,
    'actual_eur_m': np.asarray(val_true_m),
    'predicted_eur_m': np.asarray(val_pred_m),
    'error_eur_m': np.abs(np.asarray(val_true_m) - np.asarray(val_pred_m))
})

print("\n🌟 Top 10 Highest Actual Value Players (Test Set):")
top10 = result_df.nlargest(10, 'actual_eur_m')[['player','actual_eur_m','predicted_eur_m','error_eur_m']]
print(top10.to_string(index=False))

print("\n✅ Best Predictions (lowest absolute error):")
best10 = result_df.nsmallest(10, 'error_eur_m')[['player','actual_eur_m','predicted_eur_m','error_eur_m']]
print(best10.to_string(index=False))

# ── CELL 8: Hyperparameter Importance from Optuna ─────────────

print("\n✅ Evaluation complete! All figures saved to data/")
print("  • fig_actual_vs_predicted_eur.png")
print("  • fig_residual_analysis.png")
print("  • fig_error_by_position.png")
print("\n➡️  Next: Run notebook 05_explainability.py")
