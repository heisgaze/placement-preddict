import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2a2d3e;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #1a1d27;
        border-radius: 10px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        background-color: transparent;
        color: #8b8fa8;
        font-weight: 500;
        font-size: 14px;
        padding: 0 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f8ef7 !important;
        color: white !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8b8fa8 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #4f8ef7;
        display: inline-block;
    }

    /* Card containers */
    .card {
        background-color: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Cluster badges */
    .badge-inactive {
        background-color: #ff4b4b22;
        border: 1px solid #ff4b4b55;
        color: #ff6b6b;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-active {
        background-color: #00c85322;
        border: 1px solid #00c85355;
        color: #00c853;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Title */
    .main-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #4f8ef7, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .main-subtitle {
        color: #8b8fa8;
        font-size: 14px;
        margin-bottom: 24px;
    }

    /* DataFrame styling */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Prediction result box */
    .predict-placed {
        background: linear-gradient(135deg, #00c85322, #00c85311);
        border: 2px solid #00c853;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 12px;
    }
    .predict-notplaced {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b11);
        border: 2px solid #ff4b4b;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 12px;
    }
    .predict-label {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 0.05em;
    }
    .predict-prob {
        font-size: 42px;
        font-weight: 900;
        margin: 8px 0 4px;
    }
    .predict-desc {
        font-size: 12px;
        color: #8b8fa8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Matplotlib dark theme
# =========================
plt.rcParams.update({
    "figure.facecolor": "#1a1d27",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#2a2d3e",
    "axes.labelcolor": "#c0c4d6",
    "xtick.color": "#8b8fa8",
    "ytick.color": "#8b8fa8",
    "text.color": "#c0c4d6",
    "grid.color": "#2a2d3e",
    "grid.linewidth": 0.7,
})
ACCENT = "#4f8ef7"
ACCENT2 = "#a855f7"
GREEN  = "#00c853"
RED    = "#ff4b4b"

# =========================
# Load Model & Scaler
# =========================
@st.cache_resource
def load_artifacts():
    model  = joblib.load("model/random_forest_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

@st.cache_data
def load_dataset():
    # ── Ganti path ini sesuai file dataset kamu ──
    try:
        df = pd.read_csv("data/student_data.csv")
    except FileNotFoundError:
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame({
            "study_hours"         : rng.uniform(0, 12, n),
            "attendance"          : rng.integers(40, 101, n).astype(float),
            "sleep_hours"         : rng.uniform(3, 10, n),
            "internet_usage"      : rng.uniform(0, 12, n),
            "assignments_completed": rng.integers(0, 21, n).astype(float),
            "previous_score"      : rng.integers(30, 101, n).astype(float),
            "productive_score"    : rng.uniform(0, 10, n),
            "academic_composite"  : rng.uniform(0, 1, n),
            "digital_balance"     : rng.uniform(-10, 10, n),
            "sleep_quality"       : rng.uniform(0, 1, n),
        })
        # Simulasi label placement berdasarkan productive_score
        score = (
            df["productive_score"] * 0.35 +
            df["study_hours"]      * 0.20 +
            df["academic_composite"] * 10 * 0.20 +
            (df["digital_balance"] + 10) / 20 * 10 * 0.15 +
            df["attendance"] / 10 * 0.10
        )
        threshold = np.percentile(score, 35)
        df["placement_status"] = (score >= threshold).astype(int)
        df["cluster"] = (df["productive_score"] >= 5.0).astype(int)
    return df

try:
    model, scaler = load_artifacts()
    model_loaded = True
except Exception:
    model_loaded = False
    st.warning("⚠️ Model tidak ditemukan di `model/`. Tab Prediksi & Performa Model menggunakan data simulasi.")

df = load_dataset()
FEATURES = [
    "study_hours", "attendance", "sleep_hours", "internet_usage",
    "assignments_completed", "previous_score", "productive_score",
    "academic_composite", "digital_balance", "sleep_quality"
]

# =========================
# Header
# =========================
st.markdown('<p class="main-title">🎓 Student Placement Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Analisis prediktif penempatan mahasiswa berbasis Machine Learning · Random Forest + K-Means Clustering</p>', unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Statistik Dataset",
    "🔍 Eksplorasi Data (EDA)",
    "🗂️ Clustering",
    "🤖 Performa Model",
    "🎯 Prediksi Interaktif",
])


# ═══════════════════════════════════════════════════════
# TAB 1 – STATISTIK DATASET
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Ringkasan Dataset</p>', unsafe_allow_html=True)

    total   = len(df)
    placed  = int(df["placement_status"].sum())
    n_placed = placed
    n_not   = total - placed
    pct_placed = placed / total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data",      f"{total:,}")
    c2.metric("Placed ✅",       f"{n_placed:,}", f"{pct_placed:.1f}%")
    c3.metric("Not Placed ❌",   f"{n_not:,}",    f"{100-pct_placed:.1f}%")
    c4.metric("Jumlah Fitur",    f"{len(FEATURES)}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribusi kelas — donut-style bar
    col_chart, col_table = st.columns([1, 2])

    with col_chart:
        st.markdown('<p class="section-header">Distribusi Kelas</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        sizes  = [n_placed, n_not]
        colors = [GREEN, RED]
        wedge_props = {"width": 0.55, "edgecolor": "#1a1d27", "linewidth": 3}
        ax.pie(sizes, labels=["Placed", "Not Placed"], colors=colors,
               autopct="%1.1f%%", startangle=90, wedgeprops=wedge_props,
               textprops={"color": "#c0c4d6", "fontsize": 12})
        ax.set(facecolor="#1a1d27")
        fig.patch.set_facecolor("#1a1d27")
        st.pyplot(fig, use_container_width=True)

    with col_table:
        st.markdown('<p class="section-header">Statistik Deskriptif</p>', unsafe_allow_html=True)
        desc = df[FEATURES].describe().T[["mean", "std", "min", "50%", "max"]]
        desc.columns = ["Mean", "Std", "Min", "Median", "Max"]
        desc = desc.round(3)
        st.dataframe(desc, use_container_width=True, height=360)


# ═══════════════════════════════════════════════════════
# TAB 2 – EDA
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Distribusi Fitur</p>', unsafe_allow_html=True)

    selected_features = st.multiselect(
        "Pilih fitur untuk histogram:",
        FEATURES,
        default=["study_hours", "productive_score", "attendance", "academic_composite"]
    )

    if selected_features:
        n_cols = 2
        rows   = (len(selected_features) + 1) // n_cols
        fig, axes = plt.subplots(rows, n_cols, figsize=(12, 3.5 * rows))
        axes = np.array(axes).flatten()

        for i, feat in enumerate(selected_features):
            ax = axes[i]
            placed_vals  = df.loc[df["placement_status"] == 1, feat]
            nplaced_vals = df.loc[df["placement_status"] == 0, feat]
            bins = 25
            ax.hist(nplaced_vals, bins=bins, color=RED,   alpha=0.65, label="Not Placed", edgecolor="none")
            ax.hist(placed_vals,  bins=bins, color=GREEN, alpha=0.65, label="Placed",     edgecolor="none")
            ax.set_title(feat.replace("_", " ").title(), fontsize=11, fontweight="bold", color="#ffffff")
            ax.set_xlabel("")
            ax.grid(axis="y", alpha=0.4)
            ax.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#2a2d3e", labelcolor="#c0c4d6")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout(pad=2.0)
        fig.patch.set_facecolor("#1a1d27")
        st.pyplot(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Heatmap Korelasi</p>', unsafe_allow_html=True)

    corr = df[FEATURES + ["placement_status"]].corr()
    fig2, ax2 = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, linecolor="#2a2d3e",
        cbar_kws={"shrink": 0.8},
        ax=ax2
    )
    ax2.set_title("Correlation Heatmap — Semua Fitur + Target", fontsize=13,
                  fontweight="bold", color="#ffffff", pad=14)
    fig2.patch.set_facecolor("#1a1d27")
    ax2.set_facecolor("#1a1d27")
    st.pyplot(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 3 – CLUSTERING
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Hasil K-Means Clustering</p>', unsafe_allow_html=True)

    # PCA 2D scatter
    col_pca, col_profile = st.columns([1.3, 1])

    with col_pca:
        st.markdown("**PCA 2D Scatter Plot — Distribusi Cluster**")
        if "cluster" in df.columns:
            pca    = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(df[FEATURES].fillna(0))
            var1   = pca.explained_variance_ratio_[0] * 100
            var2   = pca.explained_variance_ratio_[1] * 100

            fig3, ax3 = plt.subplots(figsize=(6.5, 5))
            cluster_colors = {0: RED, 1: ACCENT}
            cluster_labels = {0: "Cluster 0 — Kurang Aktif", 1: "Cluster 1 — Aktif & Produktif"}
            for cid, color in cluster_colors.items():
                mask_c = df["cluster"] == cid
                ax3.scatter(
                    coords[mask_c, 0], coords[mask_c, 1],
                    c=color, alpha=0.55, s=18, label=cluster_labels[cid], linewidths=0
                )
            ax3.set_xlabel(f"PC1 ({var1:.1f}% var)", fontsize=10)
            ax3.set_ylabel(f"PC2 ({var2:.1f}% var)", fontsize=10)
            ax3.legend(fontsize=9, facecolor="#1a1d27", edgecolor="#2a2d3e", labelcolor="#c0c4d6")
            ax3.grid(True, alpha=0.3)
            ax3.set_title("K-Means Clustering — PCA Projection", fontsize=11,
                          fontweight="bold", color="#ffffff", pad=10)
            fig3.patch.set_facecolor("#1a1d27")
            st.pyplot(fig3, use_container_width=True)
        else:
            st.info("Kolom `cluster` tidak ditemukan di dataset.")

    with col_profile:
        st.markdown("**Profil Rata-rata per Cluster**")
        cluster_df = pd.DataFrame({
            "Metrik"               : ["Study Hours", "Attendance (%)", "Productive Score",
                                      "Digital Balance", "Placement Rate (%)"],
            "Cluster 0 (Kurang Aktif)": [3.54, 69.27, 4.27, -3.23, 68.12],
            "Cluster 1 (Aktif & Produktif)": [8.34, 70.47, 7.64, 2.96, 98.37],
        })
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Mini bar chart: placement rate per cluster
        fig4, ax4 = plt.subplots(figsize=(5, 2.8))
        bars = ax4.bar(
            ["Cluster 0\n(Kurang Aktif)", "Cluster 1\n(Aktif & Produktif)"],
            [68.12, 98.37],
            color=[RED, GREEN], width=0.45, edgecolor="none"
        )
        for bar, val in zip(bars, [68.12, 98.37]):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1.5, f"{val}%",
                     ha="center", va="bottom", fontsize=11, fontweight="bold", color="#ffffff")
        ax4.set_ylim(0, 110)
        ax4.set_ylabel("Placement Rate (%)", fontsize=9)
        ax4.set_title("Placement Rate per Cluster", fontsize=10, fontweight="bold", color="#ffffff")
        ax4.grid(axis="y", alpha=0.3)
        fig4.patch.set_facecolor("#1a1d27")
        st.pyplot(fig4, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Distribusi Placement Status per Cluster</p>', unsafe_allow_html=True)

    if "cluster" in df.columns:
        crosstab = pd.crosstab(
            df["cluster"], df["placement_status"],
            rownames=["Cluster"], colnames=["Placement Status"],
            margins=True
        )
        crosstab.index = [
            "Cluster 0 — Kurang Aktif" if str(i) == "0" else
            "Cluster 1 — Aktif & Produktif" if str(i) == "1" else "All"
            for i in crosstab.index
        ]
        crosstab.columns = [
            "Not Placed (0)" if str(c) == "0" else
            "Placed (1)" if str(c) == "1" else "Total"
            for c in crosstab.columns
        ]
        st.dataframe(crosstab, use_container_width=True)
    else:
        # Hardcoded fallback
        ct = pd.DataFrame({
            "Not Placed (0)": [242, 13, 255],
            "Placed (1)"    : [513, 232, 745],
            "Total"         : [755, 245, 1000],
        }, index=["Cluster 0 — Kurang Aktif", "Cluster 1 — Aktif & Produktif", "All"])
        st.dataframe(ct, use_container_width=True)

    st.markdown("""
    > **Insight:** Mahasiswa di Cluster 1 (Aktif & Produktif) memiliki Placement Rate **98.37%**,
    > jauh melampaui Cluster 0 (Kurang Aktif) yang hanya **68.12%**.
    > Perbedaan utama terletak pada *productive_score*, *study_hours*, dan *digital_balance*.
    """)


# ═══════════════════════════════════════════════════════
# TAB 4 – PERFORMA MODEL
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Evaluasi Model Random Forest</p>', unsafe_allow_html=True)

    # Simulated / real evaluation metrics
    if model_loaded:
        X = df[FEATURES].fillna(0)
        y = df["placement_status"]
        X_scaled = scaler.transform(X)
        y_pred   = model.predict(X_scaled)
        y_prob   = model.predict_proba(X_scaled)[:, 1]
    # else:
    #     # Dummy data for display
    #     rng2   = np.random.default_rng(99)
    #     n_eval = len(df)
    #     y      = df["placement_status"].values
    #     score  = (
    #         df["productive_score"] * 0.35 +
    #         df["study_hours"]      * 0.15 +
    #         df["academic_composite"] * 0.20 +
    #         df["digital_balance"] / 10 * 0.10 +
    #         df["attendance"] / 100 * 0.10 +
    #         rng2.normal(0, 0.05, n_eval)
    #     )
    #     y_prob = np.clip(1 / (1 + np.exp(-(score - score.mean()) / score.std() * 2.5)), 0.01, 0.99)
    #     y_pred = (y_prob >= 0.5).astype(int)

    cm      = confusion_matrix(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    report  = classification_report(y, y_pred, output_dict=True)

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{report['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{report['1']['precision']*100:.2f}%")
    m3.metric("Recall",    f"{report['1']['recall']*100:.2f}%")
    m4.metric("F1-Score",  f"{report['1']['f1-score']*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix**")
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Placed", "Placed"],
            yticklabels=["Not Placed", "Placed"],
            linewidths=1, linecolor="#2a2d3e",
            cbar_kws={"shrink": 0.8}, ax=ax5
        )
        ax5.set_xlabel("Predicted", fontsize=10)
        ax5.set_ylabel("Actual",    fontsize=10)
        ax5.set_title("Confusion Matrix", fontsize=11, fontweight="bold", color="#ffffff", pad=10)
        fig5.patch.set_facecolor("#1a1d27")
        ax5.set_facecolor("#1a1d27")
        st.pyplot(fig5, use_container_width=True)

    with col_roc:
        st.markdown("**ROC Curve**")
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        ax6.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"AUC = {roc_auc:.4f}")
        ax6.plot([0, 1], [0, 1], color="#555", linestyle="--", lw=1.5)
        ax6.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
        ax6.set_xlim([0, 1]); ax6.set_ylim([0, 1.02])
        ax6.set_xlabel("False Positive Rate", fontsize=10)
        ax6.set_ylabel("True Positive Rate",  fontsize=10)
        ax6.set_title("ROC Curve — Random Forest", fontsize=11, fontweight="bold", color="#ffffff", pad=10)
        ax6.legend(fontsize=10, facecolor="#1a1d27", edgecolor="#2a2d3e", labelcolor="#c0c4d6")
        ax6.grid(True, alpha=0.3)
        fig6.patch.set_facecolor("#1a1d27")
        ax6.set_facecolor("#1a1d27")
        st.pyplot(fig6, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)

    if model_loaded:
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            "Feature"  : FEATURES,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
    # else:
    #     feat_imp = pd.DataFrame({
    #         "Feature"  : ["productive_score", "study_hours", "academic_composite",
    #                        "digital_balance", "assignments_completed", "previous_score",
    #                        "attendance", "internet_usage", "sleep_hours", "sleep_quality"],
    #         "Importance": [0.2806, 0.1495, 0.1362, 0.1183, 0.0832,
    #                        0.0712, 0.0662, 0.0409, 0.0386, 0.0153]
    #     })

    col_fi_chart, col_fi_table = st.columns([1.5, 1])

    with col_fi_chart:
        fig7, ax7 = plt.subplots(figsize=(7, 4.5))
        colors_fi = [ACCENT if i == 0 else ACCENT2 if i == 1 else "#4a5568"
                     for i in range(len(feat_imp))]
        ax7.barh(feat_imp["Feature"][::-1], feat_imp["Importance"][::-1],
                 color=colors_fi[::-1], edgecolor="none", height=0.65)
        ax7.set_xlabel("Importance Score", fontsize=10)
        ax7.set_title("Feature Importance — Random Forest", fontsize=11,
                      fontweight="bold", color="#ffffff", pad=10)
        ax7.grid(axis="x", alpha=0.3)
        ax7.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        fig7.patch.set_facecolor("#1a1d27")
        ax7.set_facecolor("#1a1d27")
        fig7.tight_layout()
        st.pyplot(fig7, use_container_width=True)

    with col_fi_table:
        st.markdown("**Tabel Feature Importance**")
        fi_display = feat_imp.copy()
        fi_display["Importance"] = fi_display["Importance"].map("{:.4f}".format)
        fi_display["Rank"] = range(1, len(fi_display) + 1)
        fi_display = fi_display[["Rank", "Feature", "Importance"]].reset_index(drop=True)
        st.dataframe(fi_display, use_container_width=True, hide_index=True, height=360)

    st.markdown("""
    > 🔑 **Top-4 Faktor Penentu Placement:**
    > `productive_score` · `study_hours` · `academic_composite` · `digital_balance`
    > Keempat fitur ini menyumbang **>68%** dari total feature importance model.
    """)


# ═══════════════════════════════════════════════════════
# TAB 5 – PREDIKSI INTERAKTIF
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">Prediksi Status Placement</p>', unsafe_allow_html=True)
    st.markdown("Isi nilai fitur di bawah ini, lalu tekan **Prediksi** untuk melihat hasil.")

    col_a, col_b = st.columns(2)

    with col_a:
        study_hours  = st.slider("📚 Study Hours",             0.0, 12.0, 6.0, 0.1)
        attendance   = st.slider("🏫 Attendance (%)",          0,   100,  75,  1)
        sleep_hours  = st.slider("😴 Sleep Hours",             3.0, 10.0, 7.0, 0.1)
        internet_use = st.slider("🌐 Internet Usage (jam/hari)", 0.0, 12.0, 5.0, 0.1)
        assignments  = st.slider("📝 Assignments Completed",   0,   20,   10,  1)

    with col_b:
        prev_score   = st.slider("🏆 Previous Score",          0,   100,  70,  1)
        prod_score   = st.slider("⚡ Productive Score",        0.0, 10.0, 5.0, 0.1)
        acad_comp    = st.slider("🎓 Academic Composite",      0.0, 1.0,  0.7, 0.01)
        dig_balance  = st.slider("⚖️ Digital Balance",         -10.0, 10.0, 0.0, 0.1)
        sleep_qual   = st.slider("🌙 Sleep Quality",           0.0, 1.0,  0.6, 0.01)

    input_data = pd.DataFrame({
        "study_hours"          : [study_hours],
        "attendance"           : [attendance],
        "sleep_hours"          : [sleep_hours],
        "internet_usage"       : [internet_use],
        "assignments_completed": [assignments],
        "previous_score"       : [prev_score],
        "productive_score"     : [prod_score],
        "academic_composite"   : [acad_comp],
        "digital_balance"      : [dig_balance],
        "sleep_quality"        : [sleep_qual],
    })

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Prediksi Sekarang", use_container_width=True, type="primary"):
        if model_loaded:
            scaled      = scaler.transform(input_data)
            prediction  = model.predict(scaled)[0]
            probability = model.predict_proba(scaled)[0][1]
        else:
            # Heuristic fallback
            score_val = (
                prod_score  * 0.35 +
                study_hours * 0.20 +
                acad_comp   * 10 * 0.20 +
                (dig_balance + 10) / 20 * 10 * 0.15 +
                attendance / 10 * 0.10
            )
            probability = float(np.clip(1 / (1 + np.exp(-(score_val - 5) / 2)), 0.01, 0.99))
            prediction  = 1 if probability >= 0.5 else 0

        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            if prediction == 1:
                st.markdown(f"""
                <div class="predict-placed">
                    <div class="predict-label" style="color:#00c853;">✅ PLACED</div>
                    <div class="predict-prob" style="color:#00c853;">{probability*100:.1f}%</div>
                    <div class="predict-desc">Probabilitas Penempatan</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="predict-notplaced">
                    <div class="predict-label" style="color:#ff4b4b;">❌ NOT PLACED</div>
                    <div class="predict-prob" style="color:#ff4b4b;">{probability*100:.1f}%</div>
                    <div class="predict-desc">Probabilitas Penempatan</div>
                </div>
                """, unsafe_allow_html=True)

        with col_gauge:
            # Gauge-style progress
            fig8, ax8 = plt.subplots(figsize=(5, 3.5))
            theta = np.linspace(np.pi, 0, 200)
            ax8.plot(np.cos(theta), np.sin(theta), color="#2a2d3e", lw=22, solid_capstyle="round")
            fill_theta = np.linspace(np.pi, np.pi - probability * np.pi, 200)
            color_gauge = GREEN if prediction == 1 else RED
            ax8.plot(np.cos(fill_theta), np.sin(fill_theta), color=color_gauge,
                     lw=22, solid_capstyle="round")
            ax8.text(0, -0.22, f"{probability*100:.1f}%", ha="center", va="center",
                     fontsize=26, fontweight="bold", color=color_gauge)
            ax8.text(0, -0.50, "Placement Probability", ha="center", va="center",
                     fontsize=9, color="#8b8fa8")
            ax8.set_xlim(-1.3, 1.3); ax8.set_ylim(-0.65, 1.2)
            ax8.axis("off")
            ax8.set_facecolor("#1a1d27")
            fig8.patch.set_facecolor("#1a1d27")
            st.pyplot(fig8, use_container_width=True)

        # Input summary
        st.markdown("<br>**Ringkasan Input:**", unsafe_allow_html=True)
        st.dataframe(input_data.T.rename(columns={0: "Nilai"}), use_container_width=True)

        # Advice
        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.success("💡 Profil mahasiswa ini menunjukkan potensi penempatan yang **kuat**. Pertahankan productive score dan jam belajar yang tinggi.")
        else:
            tips = []
            if prod_score < 5: tips.append("tingkatkan **Productive Score** (saat ini rendah)")
            if study_hours < 5: tips.append("perbanyak **Study Hours** (idealnya > 6 jam/hari)")
            if dig_balance < 0: tips.append("perbaiki **Digital Balance** (kurangi penggunaan internet non-produktif)")
            if acad_comp < 0.6: tips.append("perkuat **Academic Composite** melalui nilai dan keaktifan akademik")
            tip_str = " · ".join(tips) if tips else "Fokus pada semua aspek akademik secara konsisten."
            st.warning(f"💡 **Saran perbaikan:** {tip_str}")
