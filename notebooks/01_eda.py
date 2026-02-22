# ============================================================
# NOTEBOOK 1 — Data Download & Exploratory Data Analysis
# ============================================================


# ── CELL 2: Imports ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, sys

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

def finalize_plot():
    # Show plots in notebooks, close when running as a script
    if 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close()

print("✅ Imports OK")

# ── CELL 3: Load Dataset ─────────────────────────────────────

DATA_PATH = '../data/players_22.csv'

df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head(3)

# ── CELL 4: Select Relevant Columns ──────────────────────────
FEATURES = [
    'short_name', 'age', 'height_cm', 'weight_kg',
    'overall', 'potential', 'value_eur', 'wage_eur',
    'release_clause_eur',
    'club_team_id', 'league_name', 'league_level',
    'nationality_name',
    'player_positions',
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'attacking_finishing', 'skill_long_passing', 'movement_sprint_speed',
    'power_shot_power', 'mentality_vision', 'mentality_composure',
    'contract_valid_until', 'international_reputation', 'weak_foot', 'skill_moves'
]

# Keep only columns that exist in this version of the dataset
available = [c for c in FEATURES if c in df.columns]
df = df[available].copy()
print(f"✅ Kept {len(available)} columns")
print("Missing columns:", set(FEATURES) - set(available))

# ── CELL 5: Target Variable — Value Distribution ─────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw distribution
axes[0].hist(df['value_eur'].dropna() / 1e6, bins=60, color='steelblue', edgecolor='white')
axes[0].set_title('Market Value Distribution (Raw)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Value (€ Millions)')
axes[0].set_ylabel('Count')

# Log-transformed
log_vals = np.log1p(df['value_eur'].dropna() / 1e6)
axes[1].hist(log_vals, bins=60, color='darkorange', edgecolor='white')
axes[1].set_title('Market Value Distribution (Log-Transformed)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('log(Value + 1)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('../data/fig_value_distribution.png', bbox_inches='tight')
finalize_plot()
print("📊 The log transform makes the target approximately normal — good for regression!")

# ── CELL 6: Missing Values ───────────────────────────────────
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Columns with missing values:")
print(missing)

fig, ax = plt.subplots(figsize=(10, 4))
missing.plot(kind='bar', ax=ax, color='coral', edgecolor='white')
ax.set_title('Missing Values per Column', fontsize=13, fontweight='bold')
ax.set_ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../data/fig_missing_values.png', bbox_inches='tight')
finalize_plot()

# ── CELL 7: Position Distribution ───────────────────────────
df['position_group'] = df['player_positions'].str.split(',').str[0].str.strip()
position_map = {
    'GK': 'GK',
    'CB': 'DEF', 'LB': 'DEF', 'RB': 'DEF', 'LWB': 'DEF', 'RWB': 'DEF',
    'CDM': 'MID', 'CM': 'MID', 'CAM': 'MID', 'LM': 'MID', 'RM': 'MID',
    'LW': 'FWD', 'RW': 'FWD', 'CF': 'FWD', 'ST': 'FWD'
}
df['position_group'] = df['position_group'].map(position_map).fillna('MID')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['position_group'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Players per Position Group', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

# Value by position
df_valid = df[df['value_eur'] > 0]
df_valid.groupby('position_group')['value_eur'].median().div(1e6).sort_values().plot(
    kind='barh', ax=axes[1], color='darkorange', edgecolor='white'
)
axes[1].set_title('Median Market Value by Position (€M)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Median Value (€ Millions)')
plt.tight_layout()
plt.savefig('../data/fig_position_analysis.png', bbox_inches='tight')
finalize_plot()

# ── CELL 8: Age vs Value ─────────────────────────────────────
df_valid = df[(df['value_eur'] > 0) & (df['age'] <= 40)]
fig, ax = plt.subplots(figsize=(12, 5))
age_val = df_valid.groupby('age')['value_eur'].median() / 1e6
ax.plot(age_val.index, age_val.values, marker='o', linewidth=2.5, color='steelblue', markersize=5)
ax.fill_between(age_val.index, age_val.values, alpha=0.15, color='steelblue')
ax.set_title('Median Market Value by Age — Peak Around 25–27', fontsize=13, fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Median Market Value (€ Millions)')
plt.tight_layout()
plt.savefig('../data/fig_age_value.png', bbox_inches='tight')
finalize_plot()
print("📊 Non-linear age effect clearly visible — perfect motivation for XGBoost!")

# ── CELL 9: Correlation Heatmap ──────────────────────────────
numeric_cols = ['age', 'overall', 'potential', 'wage_eur',
                'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
                'international_reputation', 'value_eur']
numeric_cols = [c for c in numeric_cols if c in df.columns]

corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(11, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/fig_correlation.png', bbox_inches='tight')
finalize_plot()

print("\n✅ EDA Complete! All figures saved to data/")
print("📌 Key findings:")
print("  • Market value is heavily right-skewed → log transform needed")
print("  • Strong correlations: overall, potential, wage_eur with value_eur")
print("  • Clear non-linear age-value relationship → motivates XGBoost")
print("  • Forwards have highest median value, GK lowest")
print("\n➡️  Next: Run notebook 02_preprocessing.py")
