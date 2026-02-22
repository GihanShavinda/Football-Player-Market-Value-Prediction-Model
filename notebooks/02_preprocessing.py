# ============================================================
# NOTEBOOK 2 — Data Preprocessing Pipeline
# ============================================================

# ── CELL 1: Imports ──────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings, os

warnings.filterwarnings('ignore')
print("✅ Imports OK")

# ── CELL 2: Load Raw Data ────────────────────────────────────
df = pd.read_csv('../data/players_22.csv', low_memory=False)
print(f"Raw data shape: {df.shape}")

# ── CELL 3: Select & Rename Features ─────────────────────────
KEEP_COLS = [
    'short_name', 'age', 'height_cm', 'weight_kg',
    'overall', 'potential', 'value_eur', 'wage_eur',
    'player_positions', 'league_level', 'nationality_name',
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'attacking_finishing', 'skill_long_passing', 'movement_sprint_speed',
    'power_shot_power', 'mentality_vision', 'mentality_composure',
    'international_reputation', 'weak_foot', 'skill_moves',
    'contract_valid_until', 'release_clause_eur'
]
available = [c for c in KEEP_COLS if c in df.columns]
df = df[available].copy()
print(f"After column selection: {df.shape}")

# ── CELL 4: Drop Invalid Rows ─────────────────────────────────
# Must have a target value
df = df[df['value_eur'].notna() & (df['value_eur'] > 0)].copy()

# Remove free agents (value = 0) and extreme outliers using IQR on log scale
log_val = np.log1p(df['value_eur'])
Q1, Q3 = log_val.quantile(0.01), log_val.quantile(0.99)
IQR = Q3 - Q1
df = df[(log_val >= Q1 - 1.5 * IQR) & (log_val <= Q3 + 1.5 * IQR)].copy()
print(f"After removing nulls & outliers: {df.shape}")

# ── CELL 5: Position Grouping ─────────────────────────────────
def map_position(pos_str):
    if pd.isna(pos_str):
        return 'MID'
    pos = pos_str.split(',')[0].strip()
    if pos == 'GK':
        return 'GK'
    elif pos in ['CB', 'LB', 'RB', 'LWB', 'RWB']:
        return 'DEF'
    elif pos in ['LW', 'RW', 'CF', 'ST']:
        return 'FWD'
    else:
        return 'MID'

df['position_group'] = df['player_positions'].apply(map_position)
print("Position distribution:")
print(df['position_group'].value_counts())

# ── CELL 6: Contract Years Remaining ─────────────────────────
if 'contract_valid_until' in df.columns:
    df['contract_valid_until'] = pd.to_numeric(df['contract_valid_until'], errors='coerce')
    df['contract_years_left'] = (df['contract_valid_until'] - 2022).clip(0, 5)
    df['contract_years_left'] = df['contract_years_left'].fillna(df['contract_years_left'].median())
else:
    # Some dataset versions do not include contract_valid_until
    df['contract_years_left'] = 2

# ── CELL 7: Log-transform Target & Wage ─────────────────────
df['log_value'] = np.log1p(df['value_eur'])
df['log_wage'] = np.log1p(df['wage_eur'].fillna(0))

# ── CELL 8: Nationality Grouping ─────────────────────────────
# Group rare nationalities as 'Other' to avoid high cardinality
top_nations = df['nationality_name'].value_counts().head(30).index
df['nationality_group'] = df['nationality_name'].apply(
    lambda x: x if x in top_nations else 'Other'
)

# ── CELL 9: Handle Missing Numerics ───────────────────────────
NUMERIC_FEATURES = [
    'age', 'height_cm', 'weight_kg', 'overall', 'potential',
    'log_wage', 'pace', 'shooting', 'passing', 'dribbling',
    'defending', 'physic', 'attacking_finishing', 'skill_long_passing',
    'movement_sprint_speed', 'power_shot_power', 'mentality_vision',
    'mentality_composure', 'international_reputation', 'weak_foot',
    'skill_moves', 'contract_years_left', 'league_level'
]
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in df.columns]

for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# ── CELL 10: One-Hot Encode Categoricals ──────────────────────
CATEGORICAL_FEATURES = ['position_group', 'nationality_group']

df_encoded = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=False)
cat_cols = df_encoded.columns.tolist()
print(f"One-hot encoded columns: {len(cat_cols)}")

# ── CELL 11: Final Feature Matrix ────────────────────────────
ALL_FEATURES = NUMERIC_FEATURES + cat_cols

X = pd.concat([df[NUMERIC_FEATURES].reset_index(drop=True),
               df_encoded.reset_index(drop=True)], axis=1)
y = df['log_value'].reset_index(drop=True)

print(f"\nFinal feature matrix: {X.shape}")
print(f"Target variable (log_value) stats:")
print(y.describe())

# ── CELL 12: Train / Validation / Test Split ──────────────────
# 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"\nSplit sizes:")
print(f"  Train:      {X_train.shape[0]:>6,} rows ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation: {X_val.shape[0]:>6,} rows ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:       {X_test.shape[0]:>6,} rows ({len(X_test)/len(X)*100:.1f}%)")

# ── CELL 13: Scale Features ───────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Convert back to DataFrames with feature names (needed for SHAP)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_val_scaled   = pd.DataFrame(X_val_scaled,   columns=X.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X.columns)

# ── CELL 14: Save Everything ──────────────────────────────────
os.makedirs('../data', exist_ok=True)
os.makedirs('../models', exist_ok=True)

X_train_scaled.to_csv('../data/X_train.csv', index=False)
X_val_scaled.to_csv('../data/X_val.csv',     index=False)
X_test_scaled.to_csv('../data/X_test.csv',   index=False)
y_train.to_csv('../data/y_train.csv', index=False)
y_val.to_csv('../data/y_val.csv',     index=False)
y_test.to_csv('../data/y_test.csv',   index=False)

# Save unscaled test set for display (SHAP needs real feature values)
X_test.to_csv('../data/X_test_raw.csv', index=False)

# Save scaler and feature list
joblib.dump(scaler, '../models/scaler.pkl')
joblib.dump(list(X.columns), '../models/feature_names.pkl')

# Save player names aligned to test set for display
player_names = df['short_name'].reset_index(drop=True)
player_names_test = player_names.iloc[X_test.index].reset_index(drop=True)
player_names_test.to_csv('../data/player_names_test.csv', index=False)

print("\n✅ Preprocessing complete! Files saved:")
print("  data/X_train.csv, X_val.csv, X_test.csv")
print("  data/y_train.csv, y_val.csv, y_test.csv")
print("  data/X_test_raw.csv  (unscaled, for SHAP display)")
print("  models/scaler.pkl")
print("  models/feature_names.pkl")
print("\n➡️  Next: Run notebook 03_model_training.py")
