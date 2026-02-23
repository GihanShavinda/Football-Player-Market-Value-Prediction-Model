# ============================================================
# STREAMLIT APP — Football Player Market Value Predictor
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Football Player Value Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
        color: #111111;
    }
    .stApp {
        background: radial-gradient(circle at 10% 10%, #FFF3E6 0%, #F7FAFF 40%, #EAF6FF 100%);
    }
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stText, .stTextInput label,
    .stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label,
    .stCheckbox label, .stTextArea label, .stMultiSelect label {
        color: #111111;
    }

    .stSidebar, .stSidebar * {
        color: #FFFFFF;
    }
    .stSidebar .stMarkdown, .stSidebar .stMarkdown p, .stSidebar .stMarkdown li,
    .stSidebar .stText, .stSidebar .stTextInput label, .stSidebar .stSelectbox label,
    .stSidebar .stSlider label, .stSidebar .stNumberInput label, .stSidebar .stRadio label,
    .stSidebar .stCheckbox label, .stSidebar .stTextArea label, .stSidebar .stMultiSelect label {
        color: #FFFFFF;
    }

    /* Force white on all sidebar label elements regardless of Streamlit version */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] .stSlider > div > div > div > div,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }

    .st-emotion-cache-k7vsyb h1, .st-emotion-cache-k7vsyb h2, .st-emotion-cache-k7vsyb h3, 
    .st-emotion-cache-k7vsyb h4, .st-emotion-cache-k7vsyb h5, .st-emotion-cache-k7vsyb h6,
    .st-emotion-cache-k7vsyb span {
    scroll-margin-top: 2rem;
    color: black;
    }
    div[data-testid="stAlert"] {
        color: #111111;
    }
    div[data-testid="stAlert"] p {
        color: #111111;
        font-weight: 600;
    }
    div[data-testid="stAlert"][data-baseweb="notification"] {
        background: #E7F4E8;
        border: 1px solid #B6E2BE;
    }
    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        color: #0B3D91;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #39536B;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #0B3D91, #FF8C42);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 8px 0;
        box-shadow: 0 8px 24px rgba(11, 61, 145, 0.15);
        animation: fadeIn 0.6s ease-in;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFF5D6;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .info-box {
        background: #F3F7FF;
        border-left: 4px solid #FF8C42;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #0B3D91;
        color: white;
        font-size: 0.8rem;
        margin-right: 6px;
    }
    .stButton > button {
        background-color: #0B3D91;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 30px;
        border: none;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #FF8C42;
        transform: translateY(-1px);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL & SCALER (cached for speed)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_model.json')
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

@st.cache_resource
def load_scaler_and_features():
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    scaler   = joblib.load(os.path.join(base, 'scaler.pkl'))
    features = joblib.load(os.path.join(base, 'feature_names.pkl'))
    return scaler, features

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚽ Football Player Market Value Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by XGBoost + SHAP Explainability</p>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# PRESETS & SESSION DEFAULTS
# ─────────────────────────────────────────────────────────────
DEFAULT_PROFILE = {
    "age": 24,
    "height_cm": 181,
    "weight_kg": 75,
    "overall": 72,
    "potential": 80,
    "intl_rep": 1,
    "weak_foot": 3,
    "skill_moves": 3,
    "pace": 70,
    "shooting": 55,
    "passing": 65,
    "dribbling": 68,
    "defending": 45,
    "physic": 68,
    "weekly_wage": 15000,
    "contract_years": 2,
    "league_level": 1,
    "position": "MID",
    "nationality": "Other",
}

PRESETS = {
    "Custom": None,
    "Wonderkid": {
        "age": 19, "overall": 76, "potential": 90, "pace": 82,
        "shooting": 70, "passing": 74, "dribbling": 80, "defending": 40,
        "physic": 62, "weekly_wage": 25000, "position": "FWD", "league_level": 1,
        "intl_rep": 2, "skill_moves": 4, "weak_foot": 4
    },
    "Elite Star": {
        "age": 27, "overall": 90, "potential": 92, "pace": 88,
        "shooting": 89, "passing": 86, "dribbling": 90, "defending": 50,
        "physic": 78, "weekly_wage": 250000, "position": "FWD", "league_level": 1,
        "intl_rep": 5, "skill_moves": 5, "weak_foot": 4
    },
    "Budget Defender": {
        "age": 28, "overall": 75, "potential": 77, "pace": 62,
        "shooting": 38, "passing": 60, "dribbling": 58, "defending": 80,
        "physic": 82, "weekly_wage": 18000, "position": "DEF", "league_level": 2,
        "intl_rep": 1, "skill_moves": 2, "weak_foot": 2
    },
}

for k, v in DEFAULT_PROFILE.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("preset_name", "Custom")
st.session_state.setdefault("preset_applied", "Custom")

# ─────────────────────────────────────────────────────────────
# SIDEBAR — PLAYER INPUTS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎮 Player Profile")
    st.markdown("Adjust the sliders to build your player:")
    preset_name = st.selectbox("Preset Profiles", list(PRESETS.keys()), key="preset_name")
    if preset_name != st.session_state.get("preset_applied"):
        preset = PRESETS.get(preset_name)
        if preset:
            for key, value in preset.items():
                st.session_state[key] = value
        st.session_state["preset_applied"] = preset_name
    st.markdown("---")

    st.markdown("### 📋 Basic Info")
    age          = st.slider("Age",            15, 40, st.session_state["age"], key="age")
    height_cm    = st.slider("Height (cm)",   155, 205, st.session_state["height_cm"], key="height_cm")
    weight_kg    = st.slider("Weight (kg)",    55, 110, st.session_state["weight_kg"], key="weight_kg")

    st.markdown("### ⭐ Ratings")
    overall      = st.slider("Overall Rating",     40, 99, st.session_state["overall"], key="overall")
    potential    = st.slider("Potential Rating",   40, 99, st.session_state["potential"], key="potential")
    intl_rep     = st.select_slider("International Reputation",
                                    options=[1, 2, 3, 4, 5], value=st.session_state["intl_rep"], key="intl_rep")
    weak_foot    = st.select_slider("Weak Foot",   options=[1,2,3,4,5], value=st.session_state["weak_foot"], key="weak_foot")
    skill_moves  = st.select_slider("Skill Moves", options=[1,2,3,4,5], value=st.session_state["skill_moves"], key="skill_moves")

    st.markdown("### 🏃 Attributes")
    pace         = st.slider("Pace",       30, 99, st.session_state["pace"], key="pace")
    shooting     = st.slider("Shooting",   15, 99, st.session_state["shooting"], key="shooting")
    passing      = st.slider("Passing",    20, 99, st.session_state["passing"], key="passing")
    dribbling    = st.slider("Dribbling",  20, 99, st.session_state["dribbling"], key="dribbling")
    defending    = st.slider("Defending",  10, 99, st.session_state["defending"], key="defending")
    physic       = st.slider("Physicality",30, 99, st.session_state["physic"], key="physic")

    st.markdown("### 💰 Contract & Wage")
    weekly_wage       = st.number_input("Weekly Wage (€)", min_value=500, max_value=500000,
                                        value=st.session_state["weekly_wage"], step=500, key="weekly_wage")
    contract_years    = st.slider("Contract Years Remaining", 0, 5, st.session_state["contract_years"], key="contract_years")
    league_level      = st.select_slider("League Level (1=Top)",
                                         options=[1, 2, 3, 4, 5], value=st.session_state["league_level"], key="league_level")

    st.markdown("### 🎽 Position & Nationality")
    position     = st.selectbox(
        "Position Group",
        ["GK", "DEF", "MID", "FWD"],
        index=["GK", "DEF", "MID", "FWD"].index(st.session_state["position"]),
        key="position"
    )
    nationality  = st.selectbox("Nationality Group", [
        "Brazil", "France", "Spain", "Germany", "Argentina",
        "England", "Portugal", "Italy", "Netherlands", "Belgium",
        "Croatia", "Uruguay", "Colombia", "Senegal", "Morocco",
        "Japan", "South Korea", "United States", "Mexico", "Other"
    ], index=[
        "Brazil", "France", "Spain", "Germany", "Argentina",
        "England", "Portugal", "Italy", "Netherlands", "Belgium",
        "Croatia", "Uruguay", "Colombia", "Senegal", "Morocco",
        "Japan", "South Korea", "United States", "Mexico", "Other"
    ].index(st.session_state["nationality"]), key="nationality")

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Market Value")


# ─────────────────────────────────────────────────────────────
# BUILD INPUT VECTOR
# ─────────────────────────────────────────────────────────────
def build_input(scaler, feature_names):
    log_wage = np.log1p(weekly_wage)
    # Derived attributes (approximate from FIFA logic)
    atk_finishing  = shooting * 0.9 + overall * 0.1
    skill_lp       = passing * 0.85 + dribbling * 0.1
    mvt_sprint     = pace * 0.92 + overall * 0.05
    pwr_shot       = shooting * 0.88 + physic * 0.05
    mnt_vision     = passing * 0.80 + overall * 0.1
    mnt_composure  = (overall * 0.70 + potential * 0.20 + age * 0.10)

    # Clip derived stats to valid FIFA range
    def clip_stat(v): return np.clip(v, 10, 99)

    numeric_vals = {
        'age': age, 'height_cm': height_cm, 'weight_kg': weight_kg,
        'overall': overall, 'potential': potential,
        'log_wage': log_wage,
        'pace': pace, 'shooting': shooting, 'passing': passing,
        'dribbling': dribbling, 'defending': defending, 'physic': physic,
        'attacking_finishing':  clip_stat(atk_finishing),
        'skill_long_passing':   clip_stat(skill_lp),
        'movement_sprint_speed':clip_stat(mvt_sprint),
        'power_shot_power':     clip_stat(pwr_shot),
        'mentality_vision':     clip_stat(mnt_vision),
        'mentality_composure':  clip_stat(mnt_composure),
        'international_reputation': intl_rep,
        'weak_foot': weak_foot,
        'skill_moves': skill_moves,
        'contract_years_left': contract_years,
        'league_level': league_level,
    }

    # Build full feature vector from feature names
    row = {}
    for f in feature_names:
        if f in numeric_vals:
            row[f] = numeric_vals[f]
        elif f.startswith('position_group_'):
            pos_name = f.replace('position_group_', '')
            row[f] = 1.0 if pos_name == position else 0.0
        elif f.startswith('nationality_group_'):
            nat_name = f.replace('nationality_group_', '')
            row[f] = 1.0 if nat_name == nationality else 0.0
        else:
            row[f] = 0.0

    X_raw = pd.DataFrame([row], columns=feature_names)
    X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=feature_names)
    return X_raw, X_scaled


def format_currency(value_millions):
    return f"€{value_millions:,.1f}M"

def build_report_text(value_m, tier, growth_label):
    lines = [
        "Football Player Market Value Predictor",
        "======================================",
        "",
        f"Predicted Market Value: {format_currency(value_m)}",
        f"Tier: {tier}",
        f"Growth Potential: {growth_label}",
        "",
        "\nInputs",
        "------",
        f"Age: {age}",
        f"Height (cm): {height_cm}",
        f"Weight (kg): {weight_kg}",
        f"Position: {position}",
        f"Nationality Group: {nationality}",
        f"Overall: {overall}",
        f"Potential: {potential}",
        f"Pace: {pace}",
        f"Shooting: {shooting}",
        f"Passing: {passing}",
        f"Dribbling: {dribbling}",
        f"Defending: {defending}",
        f"Physicality: {physic}",
        f"Weekly Wage: €{weekly_wage:,}",
        f"Contract Years Left: {contract_years}",
        f"League Level: {league_level}",
    ]
    return "\n".join(lines)

def radar_plot(values, labels, title):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values = np.concatenate([values, [values[0]]])
    angles = np.concatenate([angles, [angles[0]]])
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#0B3D91", linewidth=2.5)
    ax.fill(angles, values, color="#FF8C42", alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=11, fontweight="bold")
    return fig

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

if predict_btn:
    try:
        model = load_model()
        scaler, feature_names = load_scaler_and_features()
        explainer = load_explainer(model)
        X_raw, X_scaled = build_input(scaler, feature_names)
        log_pred = model.predict(X_scaled)[0]
        value_m = np.expm1(log_pred)

        if value_m < 1:
            tier = "🔵 Reserve / Youth"
        elif value_m < 5:
            tier = "🟢 Lower Division"
        elif value_m < 20:
            tier = "🟡 Mid-Table Player"
        elif value_m < 60:
            tier = "🟠 Top Club Player"
        else:
            tier = "🔴 World Class / Elite"

        potential_diff = potential - overall
        if potential_diff >= 8:
            growth_label = "🚀 High Growth"
        elif potential_diff >= 4:
            growth_label = "📈 Good Growth"
        elif potential_diff >= 1:
            growth_label = "➡️ Steady"
        else:
            growth_label = "📉 Peak / Declining"

        shap_explanation = explainer(X_scaled)
        sv = shap_explanation[0].values
        fn = shap_explanation[0].feature_names

        contrib_df = pd.DataFrame({'feature': fn, 'shap': sv})
        contrib_df['abs'] = contrib_df['shap'].abs()
        contrib_df = contrib_df.nlargest(12, 'abs').sort_values('shap')

        def clean_name(n):
            n = n.replace('position_group_', 'Position: ')
            n = n.replace('nationality_group_', 'Nation: ')
            n = n.replace('_', ' ').title()
            return n

        contrib_df['clean'] = contrib_df['feature'].apply(clean_name)

        st.session_state["prediction"] = {
            "value_m": value_m,
            "tier": tier,
            "growth_label": growth_label,
            "X_raw": X_raw,
            "X_scaled": X_scaled,
            "contrib_df": contrib_df,
            "shap_explanation": shap_explanation,
            "feature_names": feature_names,
            "model": model,
            "scaler": scaler,
        }
        st.success("✅ Prediction Complete!")
    except FileNotFoundError as e:
        st.session_state["prediction"] = None
        st.error(f"""
        **Model files not found!**

        Please make sure you have:
        1. Downloaded `players_22.csv` to the `data/` folder
        2. Run notebooks 01 → 02 → 03 in order to train and save the model

        Missing file: `{e.filename}`
        """)
    except Exception as e:
        st.session_state["prediction"] = None
        st.error(f"**Error:** {str(e)}")
        st.exception(e)

pred = st.session_state.get("prediction")
prediction_ready = pred is not None

tab_pred, tab_xai, tab_scenario, tab_eda, tab_perf, tab_features, tab_report = st.tabs([
    "🎯 Prediction",
    "🔍 Explainability",
    "📊 Scenarios",
    "📈 EDA & Dataset",
    "📉 Performance",
    "🔬 Features",
    "🧾 Report"
])

with tab_pred:
    if not prediction_ready:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Model Algorithm</div>
                <div class="metric-value">XGBoost</div>
                <div class="metric-label">Extreme Gradient Boosting</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Test R² Score</div>
                <div class="metric-value">~0.85</div>
                <div class="metric-label">85% Variance Explained</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Explainability</div>
                <div class="metric-value">SHAP</div>
                <div class="metric-label">Per-Feature Attribution</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 👈 Adjust the player profile in the sidebar and click **Predict Market Value**")
        st.markdown("""
        <div class="info-box">
        <b>How it works:</b> This app uses a trained XGBoost model to predict a football player's
        transfer market value based on their FIFA attributes. It then uses SHAP (SHapley Additive exPlanations)
        to show you <em>which features most influenced that specific prediction</em> — making the AI transparent and interpretable.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <span class="pill">Tip</span> Try the <b>Wonderkid</b> and <b>Elite Star</b> presets for your demo.
        """, unsafe_allow_html=True)
    else:
        value_m = pred["value_m"]
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Market Value</div>
                <div class="metric-value">{format_currency(value_m)}</div>
                <div class="metric-label">{pred['tier']}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Player Profile</div>
                <div class="metric-value">{position}</div>
                <div class="metric-label">Age {age} | Rating {overall} | Pot. {potential}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Growth Potential</div>
                <div class="metric-value">+{potential - overall}</div>
                <div class="metric-label">{pred['growth_label']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚽ Attribute Radar")
        radar_vals = np.array([pace, shooting, passing, dribbling, defending, physic])
        radar_labels = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physical"]
        fig = radar_plot(radar_vals, radar_labels, "Player Attribute Profile")
        st.pyplot(fig)

with tab_xai:
    if not prediction_ready:
        st.info("Generate a prediction to see SHAP explanations.")
    else:
        st.markdown("### 🔍 SHAP Explanation — What Drove This Prediction?")
        st.markdown("""
        <div class="info-box">
        The waterfall chart shows how each feature <b>pushed the prediction up (red/right)</b>
        or <b>down (blue/left)</b> from the model's average baseline prediction.
        </div>
        """, unsafe_allow_html=True)

        shap_explanation = pred["shap_explanation"]
        contrib_df = pred["contrib_df"].copy()

        col_shap, col_bar = st.columns([3, 2])
        with col_shap:
            st.markdown("#### Waterfall Plot — This Player")
            fig_wf, ax_wf = plt.subplots(figsize=(9, 7))
            shap.waterfall_plot(shap_explanation[0], max_display=14, show=False)
            plt.title(f"SHAP Waterfall | Predicted: {format_currency(pred['value_m'])}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_wf)
            plt.close()

        with col_bar:
            st.markdown("#### Top Feature Contributions")
            colors = ['#FF8C42' if v > 0 else '#0B3D91' for v in contrib_df['shap']]
            fig_bar, ax_bar = plt.subplots(figsize=(6, 7))
            ax_bar.barh(contrib_df['clean'], contrib_df['shap'],
                        color=colors, edgecolor='white', height=0.7)
            ax_bar.axvline(0, color='black', linewidth=1)
            ax_bar.set_title('Feature SHAP Values\n(Orange = boosts value, Blue = reduces)', fontsize=11, fontweight='bold')
            ax_bar.set_xlabel('SHAP Value')
            ax_bar.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close()

        st.markdown("---")
        st.markdown("### 🌍 Global SHAP Analysis")
        col_summary, col_shap_dep = st.columns([1, 1])
        with col_summary:
            if os.path.exists('../data/fig_shap_summary.png'):
                st.markdown("#### SHAP Summary Plot")
                st.image('../data/fig_shap_summary.png', use_column_width=True)
        with col_shap_dep:
            if os.path.exists('../data/fig_shap_bar.png'):
                st.markdown("#### Global Feature Importance (SHAP)")
                st.image('../data/fig_shap_bar.png', use_column_width=True)

        st.markdown("### 📈 Feature Dependencies (How Features Affect Predictions)")
        col_dep1, col_dep2 = st.columns(2)
        with col_dep1:
            if os.path.exists('../data/fig_shap_dependence_age.png'):
                st.markdown("#### Age Impact on Value")
                st.image('../data/fig_shap_dependence_age.png', use_column_width=True)
            if os.path.exists('../data/fig_shap_dependence_overall.png'):
                st.markdown("#### Overall Rating Impact on Value")
                st.image('../data/fig_shap_dependence_overall.png', use_column_width=True)
        with col_dep2:
            st.markdown("#### High-Value Player SHAP Waterfall")
            if os.path.exists('../data/fig_shap_waterfall_high.png'):
                st.image('../data/fig_shap_waterfall_high.png', use_column_width=True)
            st.markdown("#### Median-Value Player SHAP Waterfall")
            if os.path.exists('../data/fig_shap_waterfall_avg.png'):
                st.image('../data/fig_shap_waterfall_avg.png', use_column_width=True)
        top_positive = contrib_df[contrib_df['shap'] > 0].nlargest(3, 'shap')
        top_negative = contrib_df[contrib_df['shap'] < 0].nsmallest(3, 'shap')

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("**✅ Value-Boosting Features:**")
            for _, row in top_positive.iterrows():
                st.markdown(f"- **{row['clean']}**: +{row['shap']:.3f} SHAP")
        with col_neg:
            st.markdown("**⚠️ Value-Reducing Features:**")
            for _, row in top_negative.iterrows():
                st.markdown(f"- **{row['clean']}**: {row['shap']:.3f} SHAP")

with tab_scenario:
    if not prediction_ready:
        st.info("Generate a prediction to explore scenario comparisons.")
    else:
        st.markdown("### 📊 Compare — How Does This Player's Value Change?")
        X_raw = pred["X_raw"]
        scaler = pred["scaler"]
        feature_names = pred["feature_names"]
        model = pred["model"]
        value_m = pred["value_m"]

        age_range = list(range(18, 38, 2))
        values_by_age = []
        for a in age_range:
            row_copy = X_raw.copy()
            row_copy['age'] = a
            scaled = pd.DataFrame(scaler.transform(row_copy), columns=feature_names)
            v = np.expm1(model.predict(scaled)[0])
            values_by_age.append(v)

        overall_range = list(range(55, 95, 3))
        values_by_overall = []
        for o in overall_range:
            row_copy = X_raw.copy()
            row_copy['overall'] = o
            scaled = pd.DataFrame(scaler.transform(row_copy), columns=feature_names)
            v = np.expm1(model.predict(scaled)[0])
            values_by_overall.append(v)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_age, ax_age = plt.subplots(figsize=(7, 4))
            ax_age.plot(age_range, values_by_age, marker='o', color='#0B3D91', linewidth=2.5)
            ax_age.axvline(age, color='#FF8C42', linestyle='--', linewidth=2, label=f'Current Age ({age})')
            ax_age.axhline(value_m, color='#FF8C42', linestyle=':', alpha=0.7, label=f'Current Value ({format_currency(value_m)})')
            ax_age.fill_between(age_range, values_by_age, alpha=0.1, color='#0B3D91')
            ax_age.set_title('Predicted Value by Age\n(all other stats fixed)', fontweight='bold')
            ax_age.set_xlabel('Age')
            ax_age.set_ylabel('Predicted Value (€M)')
            ax_age.legend(fontsize=9)
            ax_age.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_age)
            plt.close()

        with col_b:
            fig_ov, ax_ov = plt.subplots(figsize=(7, 4))
            ax_ov.plot(overall_range, values_by_overall, marker='s', color='#FF8C42', linewidth=2.5)
            ax_ov.axvline(overall, color='#0B3D91', linestyle='--', linewidth=2, label=f'Current Rating ({overall})')
            ax_ov.axhline(value_m, color='#0B3D91', linestyle=':', alpha=0.7, label=f'Current Value ({format_currency(value_m)})')
            ax_ov.fill_between(overall_range, values_by_overall, alpha=0.1, color='#FF8C42')
            ax_ov.set_title('Predicted Value by Overall Rating\n(all other stats fixed)', fontweight='bold')
            ax_ov.set_xlabel('Overall Rating')
            ax_ov.set_ylabel('Predicted Value (€M)')
            ax_ov.legend(fontsize=9)
            ax_ov.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_ov)
            plt.close()

with tab_eda:
    st.markdown("### 📈 Exploratory Data Analysis (EDA)")
    st.markdown("""
    This section shows deep insights into the FIFA 22 player dataset:
    - Distribution of player market values (actual vs. log-transformed)
    - Missing values analysis
    - Position-based analysis
    - Age vs. Market Value relationship
    - Feature correlations
    """)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Player Market Value Distribution")
        if os.path.exists('../data/fig_value_distribution.png'):
            st.image('../data/fig_value_distribution.png', use_column_width=True)
        
        st.markdown("#### Missing Values Analysis")
        if os.path.exists('../data/fig_missing_values.png'):
            st.image('../data/fig_missing_values.png', use_column_width=True)
        
        st.markdown("#### Position-Based Analysis")
        if os.path.exists('../data/fig_position_analysis.png'):
            st.image('../data/fig_position_analysis.png', use_column_width=True)
    
    with col_b:
        st.markdown("#### Age vs. Market Value Trend")
        if os.path.exists('../data/fig_age_value.png'):
            st.image('../data/fig_age_value.png', use_column_width=True)
        
        st.markdown("#### Feature Correlations Heatmap")
        if os.path.exists('../data/fig_correlation.png'):
            st.image('../data/fig_correlation.png', use_column_width=True)
        
        st.markdown("#### Prediction Distribution (EDA)")
        if os.path.exists('../data/fig_predictions.png'):
            st.image('../data/fig_predictions.png', use_column_width=True)

with tab_perf:
    st.markdown("### 📉 Model Performance & Diagnostics")
    st.markdown("""
    Detailed evaluation of the XGBoost model on the test set:
    - Actual vs. Predicted values (to check model accuracy)
    - Residual distribution (to check for bias)
    - Residual analysis (to check assumptions)
    - Error by position group
    - Learning curves (training progress)
    """)
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### Actual vs. Predicted Market Value")
        if os.path.exists('../data/fig_actual_vs_predicted_eur.png'):
            st.image('../data/fig_actual_vs_predicted_eur.png', use_column_width=True)
        
        st.markdown("#### Learning Curves")
        if os.path.exists('../data/fig_learning_curves.png'):
            st.image('../data/fig_learning_curves.png', use_column_width=True)
    
    with col_b:
        st.markdown("#### Residual Analysis (Errors)")
        if os.path.exists('../data/fig_residual_analysis.png'):
            st.image('../data/fig_residual_analysis.png', use_column_width=True)
        
        st.markdown("#### Prediction Error by Position")
        if os.path.exists('../data/fig_error_by_position.png'):
            st.image('../data/fig_error_by_position.png', use_column_width=True)

with tab_features:
    st.markdown("### 🔬 Feature Importance & Partial Dependence Analysis")
    st.markdown("""
    Understanding which features are most important for the model and how they affect predictions:
    - Feature importance (XGBoost Gain-based)
    - Partial Dependence Plots (PDP) — marginal effect of key features
    """)
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### XGBoost Feature Importance (Gain)")
        st.markdown("Higher gain = feature contributes more to prediction accuracy")
        if os.path.exists('../data/fig_feature_importance_gain.png'):
            st.image('../data/fig_feature_importance_gain.png', use_column_width=True)
    
    with col_b:
        st.markdown("#### Partial Dependence Plots (PDP)")
        st.markdown("Shows marginal effect of key features on predicted value (holding others constant)")
        if os.path.exists('../data/fig_pdp.png'):
            st.image('../data/fig_pdp.png', use_column_width=True)

with tab_report:
    if not prediction_ready:
        st.info("Generate a prediction to download a summary report.")
    else:
        report_text = build_report_text(pred["value_m"], pred["tier"], pred["growth_label"])
        st.markdown("### 🧾 Prediction Summary")
        st.markdown(f"""
        <div style='background: #F5F5F5; padding: 20px; border-radius: 8px; border-left: 4px solid #0B3D91; color: #000000; font-family: monospace; line-height: 2.0; white-space: pre-wrap; word-break: break-word;'>{report_text}</div>
        """, unsafe_allow_html=True)
        st.download_button(
            "Download Report (.txt)",
            data=report_text,
            file_name="player_value_report.txt",
            mime="text/plain"
        )

st.markdown("---")
st.markdown("""
<div class="info-box">
<b>⚠️ Disclaimer:</b> This prediction is generated by an XGBoost model trained on FIFA 22 data.
Actual transfer market values depend on many additional factors including club negotiations,
player agent relationships, injury history, and market demand at the time of transfer.
This tool is intended as a <em>decision support aid</em>, not a definitive valuation.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; color:#888; font-size:0.85rem; margin-top:20px;'>
Machine Learning Assignment<br>
Model: XGBoost | Explainability: SHAP | Dataset: FIFA 22 (Kaggle)
</p>
""", unsafe_allow_html=True)
