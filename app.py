"""
app.py — QML-IDS-IoT Full Streamlit Application
================================================
Multi-page, deployment-ready Streamlit app for the Quantum Machine Learning
Intrusion Detection System for IoT Networks project.

Pages:
  Home      — Dashboard with key stats and project overview
  Dataset   — Upload & explore NSL-KDD data interactively
  Results   — Pre-computed metrics, plots, and model comparison
  Predict   — Live intrusion prediction using trained RF model
  About     — Research context, architecture, and future work
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="QML-IDS | Quantum Intrusion Detection",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #061020 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

.metric-card {
    background: linear-gradient(135deg, rgba(13,27,42,0.9), rgba(6,16,32,0.95));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,179,237,0.15);
    border-color: rgba(99,179,237,0.4);
}
.metric-card .value {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .label {
    color: #94a3b8; font-size: 0.85rem; font-weight: 500;
    margin-top: 6px; text-transform: uppercase; letter-spacing: 0.08em;
}
.metric-card .sub { color: #64748b; font-size: 0.75rem; margin-top: 4px; }

.section-header {
    background: linear-gradient(135deg, rgba(99,179,237,0.08), rgba(167,139,250,0.08));
    border-left: 3px solid #63b3ed;
    border-radius: 0 12px 12px 0;
    padding: 14px 20px;
    margin: 24px 0 16px 0;
}
.section-header h2 { color: #e2e8f0; margin: 0; font-size: 1.3rem; font-weight: 600; }
.section-header p  { color: #64748b; margin: 4px 0 0 0; font-size: 0.85rem; }

.glass-panel {
    background: rgba(13,27,42,0.7);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
}

.hero-banner {
    background: linear-gradient(135deg, rgba(99,179,237,0.12), rgba(167,139,250,0.12), rgba(236,72,153,0.08));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 40px 36px;
    text-align: center;
    margin-bottom: 32px;
}
.hero-banner h1 {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #63b3ed 0%, #a78bfa 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 12px 0;
}
.hero-banner p { color: #94a3b8; font-size: 1rem; margin: 0; }

.badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin: 3px;
}
.badge-blue   { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
.badge-purple { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.badge-green  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-red    { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

.status-ok   { color: #34d399; font-weight: 600; }
.status-warn { color: #fbbf24; font-weight: 600; }

h1, h2, h3, h4 { color: #e2e8f0 !important; }
p, li { color: #94a3b8; }

.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,27,42,0.6); border-radius: 12px; padding: 4px;
    border: 1px solid rgba(99,179,237,0.1);
}
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: #64748b !important; font-weight: 500; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99,179,237,0.2), rgba(167,139,250,0.2)) !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
MODELS_DIR  = os.path.join(BASE_DIR, "results", "models")
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw")

NSL_KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files",
    "num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

# ─── Cached helpers ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_metrics(filename):
    path = os.path.join(METRICS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data(show_spinner=False)
def load_comparison():
    path = os.path.join(METRICS_DIR, "comparison_table.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource(show_spinner=False)
def load_rf_model():
    import joblib
    paths = {
        "model":  os.path.join(MODELS_DIR, "random_forest.joblib"),
        "scaler": os.path.join(MODELS_DIR, "scaler.joblib"),
        "pca":    os.path.join(MODELS_DIR, "pca.joblib"),
        "fn":     os.path.join(MODELS_DIR, "feature_names.joblib"),
    }
    if not os.path.exists(paths["model"]):
        return None, None, None, None
    rf      = joblib.load(paths["model"])
    scaler  = joblib.load(paths["scaler"]) if os.path.exists(paths["scaler"]) else None
    pca     = joblib.load(paths["pca"])    if os.path.exists(paths["pca"])    else None
    f_names = joblib.load(paths["fn"])     if os.path.exists(paths["fn"])     else None
    return rf, scaler, pca, f_names

def models_ready():
    return os.path.exists(os.path.join(MODELS_DIR, "random_forest.joblib"))

def results_ready():
    return os.path.exists(os.path.join(METRICS_DIR, "classical_metrics.json"))

def dark_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,0.6)",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(99,179,237,0.08)", linecolor="rgba(99,179,237,0.15)"),
        yaxis=dict(gridcolor="rgba(99,179,237,0.08)", linecolor="rgba(99,179,237,0.15)"),
    )

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px;'>
        <div style='font-size:3rem;'>⚛️</div>
        <div style='font-size:1.1rem; font-weight:700; color:#63b3ed;'>QML-IDS</div>
        <div style='font-size:0.75rem; color:#475569; margin-top:4px;'>Quantum Intrusion Detection</div>
    </div>
    <hr style='border-color:rgba(99,179,237,0.15); margin:12px 0;'>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigate",
        ["🏠 Home", "📊 Dataset Explorer", "📈 Model Results", "🔮 Live Prediction", "📖 About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(99,179,237,0.1); margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("**System Status**")

    rf_ok  = models_ready()
    res_ok = results_ready()
    qml_ok = os.path.exists(os.path.join(METRICS_DIR, "quantum_metrics.json"))

    for label, ok in [("RF Model", rf_ok), ("Quantum VQC", qml_ok), ("Metrics", res_ok)]:
        icon  = "✅" if ok else "⏳"
        cls   = "status-ok" if ok else "status-warn"
        st.markdown(f"<span class='{cls}'>{icon} {label}</span>", unsafe_allow_html=True)

    if not rf_ok:
        st.info("Run `python main.py` to train models first.")

    st.markdown("<hr style='border-color:rgba(99,179,237,0.1); margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem; color:#334155;'>NSL-KDD • PennyLane VQC<br>Random Forest • Streamlit</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='hero-banner'>
        <h1>⚛️ Quantum Machine Learning<br>Intrusion Detection System</h1>
        <p>Benchmarking Classical Random Forest vs Quantum Variational Classifier<br>
        on the NSL-KDD network intrusion detection dataset</p>
        <div style='margin-top:20px;'>
            <span class='badge badge-blue'>PennyLane VQC</span>
            <span class='badge badge-purple'>Random Forest</span>
            <span class='badge badge-green'>NSL-KDD Dataset</span>
            <span class='badge badge-red'>Binary Classification</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("125K",  "Training Samples",    "NSL-KDD Train Set"),
        ("8",     "Qubits Used",         "AngleEmbedding + SEL"),
        ("122",   "Engineered Features", "After one-hot encoding"),
        ("72",    "Quantum Params",      "3 layers × 8 qubits × 3"),
    ]
    for col, (val, lbl, sub) in zip([col1,col2,col3,col4], cards):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='value'>{val}</div>
                <div class='label'>{lbl}</div>
                <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance summary if results exist
    rf_m  = load_metrics("classical_metrics.json")
    qml_m = load_metrics("quantum_metrics.json")

    if rf_m or qml_m:
        st.markdown("<div class='section-header'><h2>📊 Model Performance Summary</h2><p>Pre-computed results from trained models</p></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        for col, (m, name, icon) in zip([col1,col2],
                [(rf_m, "Random Forest", "🌲"), (qml_m, "Quantum VQC", "⚛️")]):
            with col:
                if m:
                    st.markdown(f"""<div class='glass-panel'>
                        <div style='font-size:1.1rem; font-weight:600; color:#e2e8f0; margin-bottom:16px;'>{icon} {name}</div>
                        <div style='display:grid; grid-template-columns:1fr 1fr; gap:12px;'>
                            <div><div style='font-size:1.6rem; font-weight:700; color:#63b3ed;'>{m['accuracy']:.1%}</div><div style='font-size:0.75rem; color:#475569;'>Accuracy</div></div>
                            <div><div style='font-size:1.6rem; font-weight:700; color:#a78bfa;'>{m['f1_score']:.1%}</div><div style='font-size:0.75rem; color:#475569;'>F1-Score</div></div>
                            <div><div style='font-size:1.6rem; font-weight:700; color:#34d399;'>{m['precision']:.1%}</div><div style='font-size:0.75rem; color:#475569;'>Precision</div></div>
                            <div><div style='font-size:1.6rem; font-weight:700; color:#f59e0b;'>{m['recall']:.1%}</div><div style='font-size:0.75rem; color:#475569;'>Recall</div></div>
                        </div>
                        <div style='margin-top:14px; padding-top:14px; border-top:1px solid rgba(99,179,237,0.1);'>
                            <span style='color:#64748b; font-size:0.8rem;'>ROC-AUC: </span>
                            <span style='color:#e2e8f0; font-weight:600;'>{m['roc_auc']:.4f}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='glass-panel' style='text-align:center; padding:40px;'>
                        <div style='font-size:2rem;'>⏳</div>
                        <div style='color:#475569; margin-top:8px;'>{icon} {name} not yet trained</div>
                        <div style='font-size:0.8rem; color:#334155; margin-top:4px;'>Run python main.py</div>
                    </div>""", unsafe_allow_html=True)

    # Pipeline steps
    st.markdown("<div class='section-header'><h2>🔄 7-Step Pipeline</h2></div>", unsafe_allow_html=True)
    steps = [
        ("1","Data Loading",   "Reads KDDTrain+.txt & KDDTest+.txt","63b3ed"),
        ("2","Preprocessing",  "Binary labels, one-hot, StandardScaler","a78bfa"),
        ("3","Random Forest",  "200 trees on full 122-feature set","34d399"),
        ("4","PCA Reduction",  "122 → 8 components, scaled [0, π]","f59e0b"),
        ("5","Quantum VQC",    "AngleEmbedding + SEL + Adam","ec4899"),
        ("6","Evaluation",     "Acc, Prec, Recall, F1, AUC","06b6d4"),
        ("7","Visualization",  "ROC, CM, comparison charts","8b5cf6"),
    ]
    cols = st.columns(7)
    for col, (num, title, desc, color) in zip(cols, steps):
        r, g, b = int(color[0:2],16), int(color[2:4],16), int(color[4:6],16)
        with col:
            st.markdown(f"""<div style='text-align:center; padding:14px 6px;
                background:rgba(13,27,42,0.7); border:1px solid rgba({r},{g},{b},0.3);
                border-radius:12px; height:130px; display:flex; flex-direction:column; justify-content:center;'>
                <div style='font-size:1.8rem; font-weight:800; color:#{color};'>{num}</div>
                <div style='font-size:0.78rem; font-weight:600; color:#e2e8f0; margin:4px 0;'>{title}</div>
                <div style='font-size:0.65rem; color:#475569; line-height:1.3;'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Explorer":
    st.markdown("<div class='section-header'><h2>📊 NSL-KDD Dataset Explorer</h2><p>Interactive exploration of the network intrusion dataset</p></div>", unsafe_allow_html=True)

    tab_auto, tab_upload = st.tabs(["📁 Local Dataset", "📤 Upload File"])

    with tab_auto:
        train_path = os.path.join(DATA_RAW, "KDDTrain+.txt")
        if os.path.exists(train_path):
            @st.cache_data(show_spinner="Loading NSL-KDD data…")
            def load_local():
                df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
                df["binary_label"] = df["label"].apply(
                    lambda x: "normal" if str(x).strip() == "normal" else "attack"
                )
                return df

            df = load_local()
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f"""<div class='metric-card'><div class='value'>{len(df):,}</div><div class='label'>Total Rows</div></div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class='metric-card'><div class='value'>{(df.binary_label=='normal').sum():,}</div><div class='label'>Normal</div></div>""", unsafe_allow_html=True)
            with c3: st.markdown(f"""<div class='metric-card'><div class='value'>{(df.binary_label=='attack').sum():,}</div><div class='label'>Attacks</div></div>""", unsafe_allow_html=True)
            with c4: st.markdown(f"""<div class='metric-card'><div class='value'>{df['label'].nunique()}</div><div class='label'>Attack Types</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            cola, colb = st.columns(2)

            with cola:
                counts = df["binary_label"].value_counts()
                fig = px.pie(
                    values=counts.values, names=counts.index,
                    title="Class Distribution",
                    color_discrete_map={"normal":"#34d399","attack":"#f87171"},
                    hole=0.5,
                )
                fig.update_layout(**dark_layout(), title_font_color="#e2e8f0", height=300,
                                  margin=dict(t=50,b=20,l=20,r=20))
                fig.update_traces(textposition="outside", textinfo="percent+label",
                                  textfont_color="#94a3b8")
                st.plotly_chart(fig, use_container_width=True)

            with colb:
                top_attacks = df[df.binary_label=="attack"]["label"].value_counts().head(10)
                fig2 = px.bar(
                    x=top_attacks.values, y=top_attacks.index, orientation="h",
                    title="Top 10 Attack Types",
                    color=top_attacks.values,
                    color_continuous_scale=["#3b82f6","#8b5cf6","#ec4899"],
                )
                fig2.update_layout(**dark_layout(), title_font_color="#e2e8f0", height=300,
                                   showlegend=False, coloraxis_showscale=False,
                                   margin=dict(t=50,b=20,l=20,r=20))
                st.plotly_chart(fig2, use_container_width=True)

            proto = df.groupby(["protocol_type","binary_label"]).size().reset_index(name="count")
            fig3 = px.bar(
                proto, x="protocol_type", y="count", color="binary_label",
                barmode="group", title="Traffic by Protocol",
                color_discrete_map={"normal":"#34d399","attack":"#f87171"},
            )
            fig3.update_layout(**dark_layout(), title_font_color="#e2e8f0", height=280,
                               margin=dict(t=50,b=20,l=20,r=20))
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("**Data Preview (first 100 rows)**")
            st.dataframe(
                df.head(100).drop(columns=["difficulty"], errors="ignore"),
                use_container_width=True, height=280,
            )

            with st.expander("📐 Numeric Feature Statistics"):
                num_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]
                            and c not in ["binary_label"]]
                st.dataframe(df[num_cols].describe().round(3), use_container_width=True)
        else:
            st.warning("No local dataset found at `data/raw/KDDTrain+.txt`. Upload a file below.")

    with tab_upload:
        uploaded = st.file_uploader("Upload NSL-KDD file (.txt or .csv)", type=["txt","csv"])
        if uploaded:
            with st.spinner("Loading…"):
                df_up = (pd.read_csv(uploaded, header=None, names=NSL_KDD_COLUMNS)
                         if uploaded.name.endswith(".txt") else pd.read_csv(uploaded))
            st.success(f"✅ {uploaded.name} — {len(df_up):,} rows × {df_up.shape[1]} cols")
            st.dataframe(df_up.head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Results":
    st.markdown("<div class='section-header'><h2>📈 Model Results & Comparison</h2><p>Classical Random Forest vs Quantum Variational Classifier</p></div>", unsafe_allow_html=True)

    rf_m  = load_metrics("classical_metrics.json")
    qml_m = load_metrics("quantum_metrics.json")
    comp  = load_comparison()

    if not rf_m and not qml_m:
        st.markdown("""<div class='glass-panel' style='text-align:center; padding:60px;'>
            <div style='font-size:3rem;'>⏳</div>
            <h3 style='color:#63b3ed;'>No Results Yet</h3>
            <p>Train the models first:</p>
            <code style='background:rgba(99,179,237,0.1); padding:8px 16px; border-radius:8px; color:#63b3ed;'>python main.py</code>
        </div>""", unsafe_allow_html=True)
    else:
        # Grouped bar chart comparison
        if comp is not None:
            metric_cols = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
            colors_map  = {"Random Forest":"#63b3ed","Quantum VQC":"#a78bfa"}
            fig = go.Figure()
            for _, row in comp.iterrows():
                model = row["Model"]
                vals  = [row[c] for c in metric_cols]
                fig.add_trace(go.Bar(
                    name=model, x=metric_cols, y=vals,
                    marker_color=colors_map.get(model,"#64748b"),
                    text=[f"{v:.3f}" for v in vals],
                    textposition="outside",
                    textfont=dict(color="#94a3b8", size=11),
                ))
            fig.update_layout(
                **dark_layout(), barmode="group",
                title=dict(text="Classical vs Quantum — All Metrics", font=dict(color="#e2e8f0",size=16)),
                yaxis_range=[0,1.12], height=380,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
                margin=dict(t=60,b=40,l=20,r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Per-model cards
        col1, col2 = st.columns(2)
        for col, (m, name, icon) in zip(
            [col1,col2],
            [(rf_m,"Random Forest","🌲"),(qml_m,"Quantum VQC","⚛️")]
        ):
            with col:
                if m:
                    st.markdown(f"##### {icon} {name}")
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Accuracy",  f"{m['accuracy']:.2%}")
                    c2.metric("Precision", f"{m['precision']:.2%}")
                    c3.metric("Recall",    f"{m['recall']:.2%}")
                    c1.metric("F1-Score",  f"{m['f1_score']:.4f}")
                    c2.metric("ROC-AUC",   f"{m['roc_auc']:.4f}")

                    # Radar
                    cats = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
                    vals = [m["accuracy"],m["precision"],m["recall"],m["f1_score"],m["roc_auc"]]
                    c_line = "#63b3ed" if "Forest" in name else "#a78bfa"
                    c_fill = "rgba(99,179,237,0.15)" if "Forest" in name else "rgba(167,139,250,0.15)"
                    fig_r = go.Figure(go.Scatterpolar(
                        r=vals+[vals[0]], theta=cats+[cats[0]],
                        fill="toself", name=name,
                        line_color=c_line, fillcolor=c_fill,
                    ))
                    fig_r.update_layout(
                        polar=dict(
                            bgcolor="rgba(13,27,42,0.6)",
                            radialaxis=dict(visible=True,range=[0,1],
                                gridcolor="rgba(99,179,237,0.1)",
                                tickfont=dict(color="#475569")),
                            angularaxis=dict(gridcolor="rgba(99,179,237,0.1)",
                                tickfont=dict(color="#94a3b8")),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8"),
                        height=290, margin=dict(t=20,b=20,l=40,r=40),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_r, use_container_width=True)

                    # Confusion matrix heatmap
                    cm = np.array(m["confusion_matrix"])
                    fig_cm = px.imshow(
                        cm, text_auto=True, aspect="auto",
                        labels=dict(x="Predicted",y="Actual"),
                        x=["Normal","Attack"], y=["Normal","Attack"],
                        color_continuous_scale="Blues",
                        title=f"Confusion Matrix — {name}",
                    )
                    fig_cm.update_layout(
                        **dark_layout(), height=280,
                        title_font_color="#e2e8f0",
                        margin=dict(t=50,b=30,l=60,r=20),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.info(f"{icon} {name} not yet trained. Run `python main.py`.")

        # Saved figures
        st.markdown("<div class='section-header'><h2>🖼️ Generated Figures</h2></div>", unsafe_allow_html=True)
        figure_files = {
            "rf_roc_curve.png":          "RF ROC Curve",
            "qml_roc_curve.png":         "QML ROC Curve",
            "combined_roc_curve.png":    "Combined ROC",
            "rf_confusion_matrix.png":   "RF Confusion Matrix",
            "qml_confusion_matrix.png":  "QML Confusion Matrix",
            "metric_comparison_bar.png": "Metric Comparison",
            "quantum_training_loss.png": "QML Training Loss",
        }
        avail = {k:v for k,v in figure_files.items()
                 if os.path.exists(os.path.join(FIGURES_DIR, k))}
        if avail:
            fig_cols = st.columns(min(3, len(avail)))
            for i, (fname, label) in enumerate(avail.items()):
                with fig_cols[i % 3]:
                    st.image(os.path.join(FIGURES_DIR, fname), caption=label,
                             use_container_width=True)
        else:
            st.info("No figures yet. Run `python main.py` to generate them.")

        # Comparison table
        if comp is not None:
            st.markdown("<div class='section-header'><h2>📋 Comparison Table</h2></div>", unsafe_allow_html=True)
            st.dataframe(
                comp.set_index("Model").style.format("{:.4f}").background_gradient(
                    cmap="Blues", axis=None, subset=["Accuracy","F1-Score","ROC-AUC"]
                ),
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.markdown("<div class='section-header'><h2>🔮 Live Intrusion Prediction</h2><p>Classify new network traffic using the trained Random Forest model</p></div>", unsafe_allow_html=True)

    rf_model, scaler, pca, feature_names = load_rf_model()

    if rf_model is None:
        st.markdown("""<div class='glass-panel' style='text-align:center; padding:50px;'>
            <div style='font-size:2.5rem;'>🤖</div>
            <h3 style='color:#63b3ed;'>Model Not Ready</h3>
            <p>Please train the model first:</p>
            <code style='background:rgba(99,179,237,0.1); padding:8px 16px; border-radius:8px; color:#63b3ed;'>python main.py</code>
        </div>""", unsafe_allow_html=True)
    else:
        tab_manual, tab_csv = st.tabs(["✍️ Manual Input", "📄 Batch CSV Upload"])

        with tab_manual:
            st.markdown("Fill in network traffic features:")
            with st.form("predict_form"):
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**Connection**")
                    duration      = st.number_input("Duration (sec)", value=0, min_value=0)
                    src_bytes     = st.number_input("Source Bytes",   value=0, min_value=0)
                    dst_bytes     = st.number_input("Dest Bytes",     value=0, min_value=0)
                    protocol_type = st.selectbox("Protocol", ["tcp","udp","icmp"])
                    service       = st.selectbox("Service", [
                        "http","ftp","smtp","ssh","dns","ftp_data","telnet","finger",
                        "domain_u","auth","other","private","pop_3","imap4","time",
                        "netstat","exec","login","shell","sunrpc","domain","bgp","IRC","X11",
                    ])
                    flag = st.selectbox("Flag", ["SF","S0","REJ","RSTO","RSTR","S1","S2","S3","OTH","SH"])

                with c2:
                    st.markdown("**Traffic Rates**")
                    count         = st.number_input("Count",        value=1, min_value=0)
                    srv_count     = st.number_input("Srv Count",    value=1, min_value=0)
                    serror_rate   = st.slider("Serror Rate",   0.0,1.0,0.0,0.01)
                    rerror_rate   = st.slider("Rerror Rate",   0.0,1.0,0.0,0.01)
                    same_srv_rate = st.slider("Same Srv Rate", 0.0,1.0,1.0,0.01)
                    diff_srv_rate = st.slider("Diff Srv Rate", 0.0,1.0,0.0,0.01)
                    logged_in     = st.selectbox("Logged In", [0,1])

                with c3:
                    st.markdown("**Host Behavior**")
                    dst_host_count         = st.number_input("Dst Host Count",         value=255,min_value=0,max_value=255)
                    dst_host_srv_count     = st.number_input("Dst Host Srv Count",     value=255,min_value=0,max_value=255)
                    dst_host_same_srv_rate = st.slider("Dst Same Srv Rate",0.0,1.0,1.0,0.01)
                    dst_host_diff_srv_rate = st.slider("Dst Diff Srv Rate",0.0,1.0,0.0,0.01)
                    hot                    = st.number_input("Hot Indicators",  value=0, min_value=0)
                    num_failed_logins      = st.number_input("Failed Logins",   value=0, min_value=0)
                    root_shell             = st.selectbox("Root Shell", [0,1])

                submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

            if submitted and feature_names:
                # Build feature vector
                row = {col: 0 for col in feature_names}
                numeric_map = {
                    "duration":duration,"src_bytes":src_bytes,"dst_bytes":dst_bytes,
                    "count":count,"srv_count":srv_count,"serror_rate":serror_rate,
                    "rerror_rate":rerror_rate,"same_srv_rate":same_srv_rate,
                    "diff_srv_rate":diff_srv_rate,"logged_in":logged_in,
                    "dst_host_count":dst_host_count,"dst_host_srv_count":dst_host_srv_count,
                    "dst_host_same_srv_rate":dst_host_same_srv_rate,
                    "dst_host_diff_srv_rate":dst_host_diff_srv_rate,
                    "hot":hot,"num_failed_logins":num_failed_logins,"root_shell":root_shell,
                }
                for k,v in numeric_map.items():
                    if k in row: row[k] = v

                for proto_val in ["tcp","udp","icmp"]:
                    k = f"protocol_type_{proto_val}"
                    if k in row: row[k] = 1 if protocol_type==proto_val else 0
                svc_k = f"service_{service}"
                if svc_k in row: row[svc_k] = 1
                flag_k = f"flag_{flag}"
                if flag_k in row: row[flag_k] = 1

                X_in = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=0)
                X_sc = scaler.transform(X_in) if scaler else X_in.values

                pred     = rf_model.predict(X_sc)[0]
                prob_atk = rf_model.predict_proba(X_sc)[0][1]
                label    = "🚨 ATTACK DETECTED" if pred==1 else "✅ NORMAL TRAFFIC"
                color    = "#f87171" if pred==1 else "#34d399"
                bg       = "rgba(248,113,113,0.1)" if pred==1 else "rgba(52,211,153,0.1)"
                border   = "rgba(248,113,113,0.3)" if pred==1 else "rgba(52,211,153,0.3)"

                st.markdown(f"""
                <div style='background:{bg}; border:2px solid {border}; border-radius:16px;
                    padding:32px; text-align:center; margin-top:24px;'>
                    <div style='font-size:2.2rem; font-weight:800; color:{color};'>{label}</div>
                    <div style='color:#94a3b8; margin-top:8px;'>
                        Attack probability: <span style='color:{color}; font-weight:700;'>{prob_atk:.1%}</span>
                        &nbsp;|&nbsp; Normal probability:
                        <span style='color:#34d399; font-weight:700;'>{1-prob_atk:.1%}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Gauge chart
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob_atk*100,
                    title={"text":"Attack Probability (%)", "font":{"color":"#94a3b8"}},
                    number={"font":{"color":color},"suffix":"%"},
                    gauge={
                        "axis":{"range":[0,100],"tickcolor":"#475569"},
                        "bar": {"color":color},
                        "bgcolor":"rgba(13,27,42,0.8)",
                        "steps":[
                            {"range":[0,40],   "color":"rgba(52,211,153,0.15)"},
                            {"range":[40,70],  "color":"rgba(251,191,36,0.15)"},
                            {"range":[70,100], "color":"rgba(248,113,113,0.15)"},
                        ],
                        "threshold":{"line":{"color":color,"width":4},"value":prob_atk*100},
                    }
                ))
                fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#94a3b8"), height=280)
                st.plotly_chart(fig_g, use_container_width=True)

        with tab_csv:
            st.markdown("Upload a CSV file for batch prediction. Columns should match NSL-KDD schema.")
            batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
            if batch_file and rf_model and scaler and feature_names:
                batch_df = pd.read_csv(batch_file)
                for drop_col in ["label","difficulty"]:
                    if drop_col in batch_df.columns:
                        batch_df.drop(columns=[drop_col], inplace=True)
                try:
                    cat_present = [c for c in CATEGORICAL if c in batch_df.columns]
                    batch_enc   = pd.get_dummies(batch_df, columns=cat_present)
                    batch_enc   = batch_enc.reindex(columns=feature_names, fill_value=0)
                    X_b         = scaler.transform(batch_enc)
                    preds       = rf_model.predict(X_b)
                    probas      = rf_model.predict_proba(X_b)[:,1]
                    n_att = (preds==1).sum()
                    n_nor = (preds==0).sum()
                    c1,c2 = st.columns(2)
                    c1.metric("🚨 Attacks Detected", f"{n_att} ({n_att/len(preds):.1%})")
                    c2.metric("✅ Normal Traffic",    f"{n_nor} ({n_nor/len(preds):.1%})")
                    result_df = pd.DataFrame({
                        "Prediction":        ["Attack" if p==1 else "Normal" for p in preds],
                        "Attack_Probability":(probas*100).round(2),
                    })
                    st.dataframe(result_df.head(300), use_container_width=True)
                    st.download_button("⬇️ Download Results CSV",
                                       result_df.to_csv(index=False),
                                       "predictions.csv","text/csv")
                except Exception as e:
                    st.error(f"Prediction error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📖 About":
    st.markdown("<div class='section-header'><h2>📖 About This Project</h2><p>Research context, methodology, and architecture</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown("""<div class='glass-panel'>
        <h3 style='color:#63b3ed;'>🎯 Research Goal</h3>
        <p>This project benchmarks a <strong style='color:#a78bfa;'>Quantum Machine Learning (QML)</strong>
        Variational Quantum Classifier (VQC) against a classical
        <strong style='color:#63b3ed;'>Random Forest</strong> for binary network intrusion detection
        on <strong style='color:#34d399;'>NSL-KDD</strong>.</p>
        <p>Both models use identical preprocessing, PCA-reduced features, and evaluation metrics —
        the standard rigorous methodology for a research paper.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='glass-panel'>
        <h3 style='color:#a78bfa;'>⚛️ Quantum Model Architecture</h3>
        <table style='width:100%; border-collapse:collapse; color:#94a3b8;'>
        <tr style='border-bottom:1px solid rgba(99,179,237,0.1);'>
            <td style='padding:8px 0; color:#63b3ed; font-weight:600; width:35%;'>Encoding</td>
            <td>AngleEmbedding — each feature → RY rotation on 1 qubit</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(99,179,237,0.1);'>
            <td style='padding:8px 0; color:#63b3ed; font-weight:600;'>Ansatz</td>
            <td>StronglyEntanglingLayers (3 layers, CNOT ring topology)</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(99,179,237,0.1);'>
            <td style='padding:8px 0; color:#63b3ed; font-weight:600;'>Measurement</td>
            <td>⟨Z₀⟩ Pauli-Z on qubit 0 → sigmoid → P(attack)</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(99,179,237,0.1);'>
            <td style='padding:8px 0; color:#63b3ed; font-weight:600;'>Optimizer</td>
            <td>Adam (PyTorch) + Binary Cross-Entropy, parameter-shift rule</td>
        </tr>
        <tr>
            <td style='padding:8px 0; color:#63b3ed; font-weight:600;'>Parameters</td>
            <td>72 trainable weights (3 layers × 8 qubits × 3 angles)</td>
        </tr>
        </table>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='glass-panel'>
        <h3 style='color:#34d399;'>🚀 Suggested Research Extensions</h3>
        <ul style='color:#94a3b8;'>
            <li><strong style='color:#e2e8f0;'>Feature map ablation</strong> — AngleEmbedding vs AmplitudeEmbedding vs IQPEmbedding</li>
            <li><strong style='color:#e2e8f0;'>Qubit-count sweep</strong> — accuracy vs N_QUBITS ∈ {4,6,8,10,12}</li>
            <li><strong style='color:#e2e8f0;'>Hybrid model</strong> — VQC feature extractor → logistic regression head</li>
            <li><strong style='color:#e2e8f0;'>Noise simulation</strong> — default.mixed or Qiskit noise models</li>
            <li><strong style='color:#e2e8f0;'>IoT datasets</strong> — TON_IoT, BoT-IoT, CICIoT2023</li>
            <li><strong style='color:#e2e8f0;'>Quantum Kernel SVM</strong> — third model class</li>
            <li><strong style='color:#e2e8f0;'>Adversarial robustness</strong> — compare RF vs VQC degradation</li>
        </ul>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-panel'><h3 style='color:#f59e0b;'>📦 Tech Stack</h3></div>", unsafe_allow_html=True)
        techs = [
            ("⚛️","PennyLane 0.44","Quantum circuit simulation","a78bfa"),
            ("🔥","PyTorch 2.x","Autograd + Adam optimizer","f87171"),
            ("🌲","scikit-learn","Random Forest + PCA + Scaler","34d399"),
            ("🐼","Pandas / NumPy","Data processing pipeline","63b3ed"),
            ("📊","Plotly","Interactive visualizations","f59e0b"),
            ("🌐","Streamlit","Web application UI","06b6d4"),
        ]
        for icon,name,desc,color in techs:
            st.markdown(f"""<div style='display:flex; align-items:center; gap:12px; padding:10px;
                background:rgba(13,27,42,0.5); border-radius:8px; border-left:3px solid #{color};
                margin:4px 0;'>
                <span style='font-size:1.2rem;'>{icon}</span>
                <div><div style='color:#e2e8f0; font-weight:600; font-size:0.9rem;'>{name}</div>
                <div style='color:#475569; font-size:0.75rem;'>{desc}</div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='glass-panel' style='margin-top:16px;'>
        <h3 style='color:#ec4899;'>⚠️ Known Limitations</h3>
        <ul style='color:#94a3b8; font-size:0.87rem;'>
            <li>Quantum circuit runs on a <strong style='color:#fbbf24;'>classical simulator</strong> — no true quantum hardware</li>
            <li>QML trained on <strong style='color:#fbbf24;'>subsampled data</strong> (~800 rows) for tractable runtime</li>
            <li>NSL-KDD is a <strong style='color:#fbbf24;'>1999-derived dataset</strong>, not raw IoT traffic</li>
            <li>PCA dimensionality reduction is a NISQ-era necessity, not a quantum advantage</li>
        </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='glass-panel' style='text-align:center; margin-top:16px;'>
        <p style='color:#475569; font-size:0.85rem;'>
        ⚛️ QML-IDS-IoT &nbsp;·&nbsp; NSL-KDD Dataset by University of New Brunswick &nbsp;·&nbsp;
        Built with PennyLane, PyTorch &amp; Streamlit
        </p>
    </div>""", unsafe_allow_html=True)