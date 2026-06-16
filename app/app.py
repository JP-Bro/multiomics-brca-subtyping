import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle, torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import MultiOmicsAutoencoder

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Multi-Omics BRCA Subtype Classifier",
    page_icon="🧬",
    layout="wide"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1f3a5f;
    margin-bottom: 0;
}
.sub-header {
    font-size: 1rem;
    color: #5a6c7d;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 1.2rem;
    border-left: 4px solid #2e7bcf;
}
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

# ============ HEADER ============
st.markdown('<p class="main-header">🧬 Multi-Omics Breast Cancer Subtype Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predicting PAM50 molecular subtypes from gene expression + mutation data (TCGA-BRCA, n=593)</p>', unsafe_allow_html=True)

# ============ TABS ============
tab1, tab2, tab3 = st.tabs(["📊 Overview & Results", "🔬 Live Prediction Demo", "🧠 Model Interpretability"])

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    acc_xgb = (xgb_model.predict(X_test) == y_test).mean()

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("XGBoost Accuracy", f"{acc_xgb:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Autoencoder Fusion Accuracy", f"{meta['acc_full']:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Expression-only Accuracy", f"{meta['acc_expr_only']:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Mutation-only Accuracy", f"{meta['acc_mut_only']:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Confusion Matrices")
        cm_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'confusion_matrices.png')
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            st.info("Run notebook 05 to generate confusion matrix figure.")

    with col_right:
        st.subheader("Key Findings")
        st.markdown("""
        - **Domain-informed feature selection**: Adding ERBB2 and MKI67 (clinically critical PAM50 markers
          missed by variance-based selection) improved Her2 recall (0.57→0.71) and LumB precision (0.83→0.92).
        - **Architecture comparison**: Autoencoder fusion matched XGBoost overall (85% vs 84%), with
          complementary per-class strengths — better LumB, slightly weaker Her2.
        - **Ablation finding**: Expression alone matches the full fusion model exactly (85.4%), while
          mutation alone reaches only 45%. Gene expression carries nearly all the predictive signal for
          PAM50 — consistent with PAM50 being an expression-derived classifier.
        - **Normal-like subtype** (19/593 patients) was not reliably classified by either model,
          consistent with its known instability as a PAM50 category in the literature.
        """)

    st.markdown("---")
    st.subheader("About PAM50 Subtypes")
    info_cols = st.columns(5)
    subtype_info = {
        "Basal": "Triple-negative, aggressive, chemo-responsive",
        "Her2": "HER2-amplified, targeted therapy (trastuzumab)",
        "LumA": "Hormone receptor+, low proliferation, good prognosis",
        "LumB": "Hormone receptor+, higher proliferation than LumA",
        "Normal": "Normal-like profile, biologically debated subtype"
    }
    for col, (subtype, desc) in zip(info_cols, subtype_info.items()):
        with col:
            st.markdown(f"**{subtype}**")
            st.caption(desc)

# ---------------- TAB 2: LIVE PREDICTION DEMO ----------------
with tab2:
    st.subheader("Get a Prediction")

    mode = st.radio("Choose input method:", ["Select test patient (demo)", "Upload patient data (CSV)"])

    if mode == "Select test patient (demo)":
        idx = st.slider("Patient index", 0, len(y_test) - 1, 0)
        true_label = le_classes[y_test[idx]]
        expr_vec = X_expr_test[idx:idx+1]
        mut_vec = X_mut_test[idx:idx+1]
        xgb_vec = X_test[idx:idx+1]
        st.markdown(f"**True Subtype (known for this demo patient)**: {true_label}")

    else:
        st.markdown("""
        Upload a CSV with **one row** containing:
        - Expression values for the model's expression genes (in order)
        - Mutation values (0/1) for the model's mutation genes (in order)
        - Total columns expected: **{} expression + {} mutation = {}**
        """.format(len(meta['expr_features']), len(meta['mut_features']),
                   len(meta['expr_features']) + len(meta['mut_features'])))

        # Provide a downloadable template
        template_cols = meta['expr_features'] + meta['mut_features']
        template_df = pd.DataFrame(columns=template_cols)
        st.download_button("Download CSV template", template_df.to_csv(index=False),
                            "patient_template.csv", "text/csv")

        uploaded = st.file_uploader("Upload patient CSV", type="csv")

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

    # --- Run predictions (same for both modes) ---
    expr_input = torch.tensor(expr_vec, dtype=torch.float32)
    mut_input = torch.tensor(mut_vec, dtype=torch.float32)

    with torch.no_grad():
        ae_logits, _, _ = ae_model(expr_input, mut_input)
        ae_probs = torch.softmax(ae_logits, dim=1).numpy()[0]

    xgb_probs = xgb_model.predict_proba(xgb_vec)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**XGBoost Prediction**")
        xgb_df = pd.DataFrame({'Subtype': le_classes, 'Probability': xgb_probs}).sort_values('Probability', ascending=False)
        st.bar_chart(xgb_df.set_index('Subtype'))
        st.success(f"Predicted: **{le_classes[xgb_probs.argmax()]}**")

    with col2:
        st.markdown("**Autoencoder Fusion Prediction**")
        ae_df = pd.DataFrame({'Subtype': le_classes, 'Probability': ae_probs}).sort_values('Probability', ascending=False)
        st.bar_chart(ae_df.set_index('Subtype'))
        st.success(f"Predicted: **{le_classes[ae_probs.argmax()]}**")

    if true_label:
        st.markdown("---")
        agree_xgb = "✅" if le_classes[xgb_probs.argmax()] == true_label else "❌"
        agree_ae = "✅" if le_classes[ae_probs.argmax()] == true_label else "❌"
        st.markdown(f"True label: **{true_label}** | XGBoost correct: {agree_xgb} | Autoencoder correct: {agree_ae}")
# ---------------- TAB 3: INTERPRETABILITY ----------------
with tab3:
    st.subheader("SHAP Feature Importance by Subtype")
    st.caption("Top genes driving each subtype's predictions (XGBoost model)")

    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    cols = st.columns(2)
    for i, subtype in enumerate(le_classes):
        img_path = os.path.join(fig_dir, f'shap_{subtype}.png')
        with cols[i % 2]:
            st.markdown(f"**{subtype}**")
            if os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
            else:
                st.info(f"SHAP plot for {subtype} not found — run notebook 05.")

st.markdown("---")
st.caption("Built with TCGA-BRCA data via UCSC Xena | XGBoost + PyTorch Autoencoder Fusion")