import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle, torch, numpy as np, pandas as pd
from src.models import MultiOmicsAutoencoder

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Multi-Omics BRCA Subtype Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ DESIGN SYSTEM ============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-deep: #0b1220;
    --bg-panel: #131c2e;
    --bg-panel-light: #1a2640;
    --border: #253349;
    --teal: #14b8a6;
    --teal-dim: #0d9488;
    --amber: #f59e0b;
    --rose: #f43f5e;
    --text-primary: #e8edf5;
    --text-secondary: #8b9bb4;
    --text-muted: #5a6b85;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg-deep);
}

/* Hide default streamlit chrome clutter */
#MainMenu, footer, header {visibility: hidden;}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

/* ---------- HERO ---------- */
.hero {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.4rem;
    margin-bottom: 1.8rem;
    flex-wrap: wrap;
    gap: 1rem;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.4rem;
    display: block;
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.15;
    letter-spacing: -0.01em;
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--text-secondary);
    margin-top: 0.45rem;
    max-width: 560px;
}
.hero-cohort {
    font-family: 'JetBrains Mono', monospace;
    text-align: right;
    color: var(--text-muted);
    font-size: 0.78rem;
    line-height: 1.6;
}
.hero-cohort b { color: var(--text-secondary); font-weight: 600; }

/* ---------- READOUT METRIC ROW ---------- */
.readout-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.8rem;
}
.readout-cell {
    background: var(--bg-panel);
    padding: 1.1rem 1.3rem;
    position: relative;
}
.readout-cell.accent::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: var(--teal);
}
.readout-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}
.readout-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.readout-value.dim { color: var(--text-secondary); font-size: 1.7rem; }
.readout-delta {
    font-size: 0.76rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
}
.readout-delta.up { color: var(--teal); }
.readout-delta.down { color: var(--rose); }

/* ---------- SECTION LABEL ---------- */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--teal);
    margin: 1.6rem 0 0.7rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ---------- PANEL ---------- */
.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
}

/* ---------- FINDING ROW ---------- */
.finding {
    display: flex;
    gap: 0.85rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid var(--border);
}
.finding:last-child { border-bottom: none; }
.finding-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--teal);
    background: rgba(20, 184, 166, 0.1);
    border: 1px solid rgba(20, 184, 166, 0.25);
    border-radius: 5px;
    padding: 0.15rem 0.5rem;
    height: fit-content;
    white-space: nowrap;
    margin-top: 0.1rem;
}
.finding-tag.warn {
    color: var(--amber);
    background: rgba(245, 158, 11, 0.1);
    border-color: rgba(245, 158, 11, 0.25);
}
.finding-text {
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.55;
}
.finding-text b { color: var(--text-primary); font-weight: 600; }

/* ---------- SUBTYPE CHIPS ---------- */
.subtype-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.7rem;
}
.subtype-chip {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--chip-color, var(--teal));
    border-radius: 8px;
    padding: 0.8rem 0.9rem;
}
.subtype-name {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}
.subtype-desc {
    font-size: 0.74rem;
    color: var(--text-muted);
    line-height: 1.4;
}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    background: var(--bg-panel);
    padding: 0.3rem;
    border-radius: 10px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 0.85rem;
    height: 2.4rem;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-panel-light) !important;
    color: var(--text-primary) !important;
}

/* ---------- IMAGE FRAME ---------- */
.img-frame {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem;
}
.img-frame img { border-radius: 6px; }

/* Streamlit native overrides */
[data-testid="stMetricValue"] { color: var(--text-primary); }
.stRadio > label, .stSlider > label { color: var(--text-secondary) !important; }
hr { border-color: var(--border) !important; }
[data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }
</style>
""", unsafe_allow_html=True)

# ============ LOAD EVERYTHING ============
@st.cache_resource
def load_artifacts():
    base = os.path.join(os.path.dirname(__file__), '..')

    with open(f'{base}/results/models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    data = np.load(f'{base}/data/processed/test_data.npz')

    with open(f'{base}/data/processed/metadata.pkl', 'rb') as f:
        meta = pickle.load(f)

    config = torch.load(f'{base}/results/models/autoencoder_config.pt')
    ae_model = MultiOmicsAutoencoder(expr_dim=config['expr_dim'], mut_dim=config['mut_dim'])
    ae_model.load_state_dict(torch.load(f'{base}/results/models/autoencoder_state.pt'))
    ae_model.eval()

    return xgb_model, ae_model, data, meta

xgb_model, ae_model, data, meta = load_artifacts()
X_expr_test, X_mut_test, y_test, X_test = data['X_expr_test'], data['X_mut_test'], data['y_test'], data['X_test_concat']
le_classes = meta['label_classes']
acc_xgb = (xgb_model.predict(X_test) == y_test).mean()

# ============ HERO ============
st.markdown(f"""
<div class="hero">
    <div>
        <span class="hero-eyebrow">Precision Oncology · Multi-Omics ML</span>
        <h1 class="hero-title">Breast Cancer Subtype Classifier</h1>
        <p class="hero-sub">Predicting PAM50 molecular subtype from gene expression and somatic mutation profiles. Two architectures, compared head-to-head, on real TCGA-BRCA patient data.</p>
    </div>
    <div class="hero-cohort">
        COHORT &nbsp;<b>n=593</b><br>
        SOURCE &nbsp;<b>TCGA-BRCA</b><br>
        CLASSES &nbsp;<b>5 (PAM50)</b>
    </div>
</div>
""", unsafe_allow_html=True)

# ============ TABS ============
tab1, tab2, tab3 = st.tabs(["OVERVIEW", "LIVE PREDICTION", "INTERPRETABILITY"])

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    st.markdown(f"""
    <div class="readout-row">
        <div class="readout-cell">
            <div class="readout-label">XGBoost</div>
            <div class="readout-value">{acc_xgb:.1%}</div>
            <div class="readout-delta">late fusion, 4,716 features</div>
        </div>
        <div class="readout-cell accent">
            <div class="readout-label">Autoencoder Fusion</div>
            <div class="readout-value">{meta['acc_full']:.1%}</div>
            <div class="readout-delta up">best of two architectures</div>
        </div>
        <div class="readout-cell">
            <div class="readout-label">Expression-Only Ablation</div>
            <div class="readout-value dim">{meta['acc_expr_only']:.1%}</div>
            <div class="readout-delta">matches full model exactly</div>
        </div>
        <div class="readout-cell">
            <div class="readout-label">Mutation-Only Ablation</div>
            <div class="readout-value dim">{meta['acc_mut_only']:.1%}</div>
            <div class="readout-delta down">−40.5pp vs. full model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.15, 1])

    with col_left:
        st.markdown('<div class="section-label">Confusion Matrices</div>', unsafe_allow_html=True)
        cm_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'confusion_matrices.png')
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.caption("Run notebook 05 to generate this figure.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-label">Key Findings</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="panel">
            <div class="finding">
                <span class="finding-tag">FEATURE ENG.</span>
                <span class="finding-text">Adding back <b>ERBB2</b> and <b>MKI67</b> — clinically critical markers dropped by pure variance selection — improved Her2 recall <b>0.57→0.71</b> and LumB precision <b>0.83→0.92</b>.</span>
            </div>
            <div class="finding">
                <span class="finding-tag">ARCHITECTURE</span>
                <span class="finding-text">Autoencoder fusion matched XGBoost overall but with <b>complementary strengths</b> — stronger LumB recall, slightly weaker Her2.</span>
            </div>
            <div class="finding">
                <span class="finding-tag">ABLATION</span>
                <span class="finding-text">Expression alone reproduces full-model accuracy <b>exactly</b>. Mutation data added <b>no measurable signal</b> for this task.</span>
            </div>
            <div class="finding">
                <span class="finding-tag warn">LIMITATION</span>
                <span class="finding-text">Normal-like (19/593 patients) was <b>not reliably classified</b> by either model — a known instability of this PAM50 category, not a modeling failure.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">PAM50 Subtypes</div>', unsafe_allow_html=True)
    subtype_info = [
        ("Basal", "Triple-negative, aggressive, chemo-responsive", "#f43f5e"),
        ("Her2", "HER2-amplified, targeted therapy (trastuzumab)", "#f59e0b"),
        ("LumA", "Hormone receptor+, low proliferation, good prognosis", "#14b8a6"),
        ("LumB", "Hormone receptor+, higher proliferation than LumA", "#38bdf8"),
        ("Normal", "Normal-like profile, biologically debated subtype", "#8b9bb4"),
    ]
    chips_html = '<div class="subtype-grid">'
    for name, desc, color in subtype_info:
        chips_html += f'<div class="subtype-chip" style="--chip-color:{color}"><div class="subtype-name">{name}</div><div class="subtype-desc">{desc}</div></div>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)

# ---------------- TAB 2: LIVE PREDICTION DEMO ----------------
with tab2:
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    mode = st.radio("Choose input method:", ["Select test patient (demo)", "Upload patient data (CSV)"], label_visibility="collapsed")

    if mode == "Select test patient (demo)":
        idx = st.slider("Patient index", 0, len(y_test) - 1, 0)
        true_label = le_classes[y_test[idx]]
        expr_vec = X_expr_test[idx:idx+1]
        mut_vec = X_mut_test[idx:idx+1]
        xgb_vec = X_test[idx:idx+1]
        st.caption(f"True subtype for this held-out test patient: **{true_label}**")

    else:
        st.caption(f"Upload one row: {len(meta['expr_features'])} expression + {len(meta['mut_features'])} mutation columns = {len(meta['expr_features']) + len(meta['mut_features'])} total, in order.")

        template_cols = meta['expr_features'] + meta['mut_features']
        template_df = pd.DataFrame(columns=template_cols)
        st.download_button("Download CSV template", template_df.to_csv(index=False),
                            "patient_template.csv", "text/csv")

        uploaded = st.file_uploader("Upload patient CSV", type="csv", label_visibility="collapsed")

        if uploaded is not None:
            patient_df = pd.read_csv(uploaded)
            expected_cols = len(meta['expr_features']) + len(meta['mut_features'])
            if patient_df.shape[1] != expected_cols or patient_df.shape[0] != 1:
                st.error(f"Expected 1 row with {expected_cols} columns, got {patient_df.shape}")
                st.stop()

            n_expr = len(meta['expr_features'])
            expr_vec = patient_df.iloc[:, :n_expr].values.astype(np.float32)
            mut_vec = patient_df.iloc[:, n_expr:].values.astype(np.float32)
            xgb_vec = np.hstack([expr_vec, mut_vec])
            true_label = None
        else:
            st.info("Upload a CSV to get a prediction, or switch to demo mode.")
            st.stop()

    expr_input = torch.tensor(expr_vec, dtype=torch.float32)
    mut_input = torch.tensor(mut_vec, dtype=torch.float32)

    with torch.no_grad():
        ae_logits, _, _ = ae_model(expr_input, mut_input)
        ae_probs = torch.softmax(ae_logits, dim=1).numpy()[0]

    xgb_probs = xgb_model.predict_proba(xgb_vec)[0]

    st.markdown('<div class="section-label">Predictions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="panel"><b style="color:#e8edf5">XGBoost</b> → <span style="color:#14b8a6;font-weight:700">{le_classes[xgb_probs.argmax()]}</span></div>', unsafe_allow_html=True)
        xgb_df = pd.DataFrame({'Subtype': le_classes, 'Probability': xgb_probs}).sort_values('Probability', ascending=False)
        st.bar_chart(xgb_df.set_index('Subtype'), color="#14b8a6")

    with col2:
        st.markdown(f'<div class="panel"><b style="color:#e8edf5">Autoencoder Fusion</b> → <span style="color:#14b8a6;font-weight:700">{le_classes[ae_probs.argmax()]}</span></div>', unsafe_allow_html=True)
        ae_df = pd.DataFrame({'Subtype': le_classes, 'Probability': ae_probs}).sort_values('Probability', ascending=False)
        st.bar_chart(ae_df.set_index('Subtype'), color="#38bdf8")

    if true_label:
        agree_xgb = "agrees" if le_classes[xgb_probs.argmax()] == true_label else "disagrees"
        agree_ae = "agrees" if le_classes[ae_probs.argmax()] == true_label else "disagrees"
        st.caption(f"Ground truth: **{true_label}** — XGBoost {agree_xgb}, Autoencoder Fusion {agree_ae}.")

# ---------------- TAB 3: INTERPRETABILITY ----------------
with tab3:
    st.markdown('<div class="section-label">SHAP — Top Genes per Subtype (XGBoost)</div>', unsafe_allow_html=True)
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    cols = st.columns(2)
    for i, subtype in enumerate(le_classes):
        img_path = os.path.join(fig_dir, f'shap_{subtype}.png')
        with cols[i % 2]:
            st.markdown(f'<div class="img-frame" style="margin-bottom:1rem"><div style="font-family:JetBrains Mono;font-size:0.75rem;color:#8b9bb4;margin-bottom:0.4rem;padding:0 0.2rem">{subtype.upper()}</div>', unsafe_allow_html=True)
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.caption(f"SHAP plot for {subtype} not found.")
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #253349;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#5a6b85">
TCGA-BRCA via UCSC Xena · XGBoost + PyTorch Autoencoder Fusion · Built for precision oncology R&D portfolio
</div>
""", unsafe_allow_html=True)
