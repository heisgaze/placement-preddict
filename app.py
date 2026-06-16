import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .main { background-color: #F7F8FC; }
    .block-container { padding: 2rem 2.5rem 2rem 2.5rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1D2E 0%, #252840 100%);
        border-right: 1px solid #2E3250;
    }
    [data-testid="stSidebar"] * { color: #E0E4F8 !important; }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.9rem;
        padding: 0.4rem 0;
        color: #A8AED4 !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover { color: #FFFFFF !important; }

    /* Page header */
    .page-header {
        background: linear-gradient(135deg, #1A1D2E 0%, #2D3561 60%, #3D4A8A 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .page-header::after {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: rgba(100, 130, 255, 0.15);
        border-radius: 50%;
    }
    .page-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        color: #FFFFFF;
    }
    .page-header p {
        color: #A8B4E8;
        margin: 0;
        font-size: 0.9rem;
    }

    /* Section title */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin: 1.5rem 0 1rem 0;
        padding-left: 0.75rem;
        border-left: 4px solid #4361EE;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: 1px solid #E8EAFF;
        box-shadow: 0 2px 12px rgba(67, 97, 238, 0.06);
        text-align: center;
    }
    .metric-card .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1A1D2E;
        line-height: 1;
    }
    .metric-card .metric-label {
        font-size: 0.75rem;
        color: #7B82A8;
        margin-top: 0.4rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .metric-sub {
        font-size: 0.8rem;
        color: #4361EE;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    /* Chart card */
    .chart-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #E8EAFF;
        box-shadow: 0 2px 12px rgba(67, 97, 238, 0.06);
        margin-bottom: 1rem;
    }

    /* Prediction result */
    .pred-placed {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
    }
    .pred-not-placed {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
    }
    .pred-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .pred-sub { font-size: 0.85rem; opacity: 0.9; margin-top: 0.25rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #EDEEFF;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.85rem;
        color: #7B82A8;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #4361EE !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }

    /* Info box */
    .info-box {
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        font-size: 0.82rem;
        color: #3730A3;
        margin: 0.75rem 0;
        line-height: 1.5;
    }

    /* Hide Streamlit default elements */
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS (with graceful fallback)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        artifacts['model']          = joblib.load('model/best_model.pkl')
        artifacts['scaler']         = joblib.load('model/scaler.pkl')
        artifacts['le']             = joblib.load('model/label_encoder.pkl')
        artifacts['scaler_cluster'] = joblib.load('model/scaler_cluster.pkl')
        artifacts['kmeans']         = joblib.load('model/kmeans.pkl')
        artifacts['X_test']         = joblib.load('model/X_test.pkl')
        artifacts['y_test']         = joblib.load('model/y_test.pkl')
        with open('model/feature_names.json') as f:
            artifacts['feature_names'] = json.load(f)
        with open('model/model_info.json') as f:
            artifacts['model_info'] = json.load(f)
        artifacts['loaded'] = True
    except Exception as e:
        artifacts['loaded'] = False
        artifacts['error'] = str(e)
    return artifacts


@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('datasets/student_dataset_10000_rows.csv')
        return df
    except:
        return None


def apply_feature_engineering(df):
    df = df.copy()
    df['productive_score']   = df['study_hours'] * 0.5 + df['assignments_completed'] * 0.3
    df['academic_composite'] = (df['previous_score'] / 95) * 0.6 + (df['attendance'] / 100) * 0.4
    df['digital_balance']    = df['study_hours'] - df['internet_usage']
    df['sleep_quality']      = df['sleep_hours'].apply(
        lambda x: 1 if 7 <= x <= 9 else (0.5 if x in [6, 10] else 0)
    )
    return df


# ─────────────────────────────────────────────────────────────────
# PLOT HELPERS  (matplotlib — Streamlit-safe)
# ─────────────────────────────────────────────────────────────────
PALETTE     = ['#4361EE', '#F72585', '#7209B7', '#3A0CA3', '#4CC9F0', '#06D6A0']
PLACED_CLR  = '#10B981'
NOT_CLR     = '#EF4444'
BG          = '#FFFFFF'
GRID        = '#F0F2FF'

def _fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors='#555')
    for spine in ax.spines.values():
        spine.set_color('#E0E4F8')
    return fig, ax


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem'>
        <div style='font-size:2.5rem'>🎓</div>
        <div style='font-family: Space Grotesk, sans-serif; font-size:1.1rem; font-weight:700; color:#FFFFFF'>
            Student Placement
        </div>
        <div style='font-size:0.75rem; color:#6B7280; margin-top:0.25rem'>Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        [
            "📊  Ringkasan Dataset",
            "🔍  Exploratory Data Analysis",
            "🗂️  Clustering",
            "🤖  Performa Model",
            "🎯  Prediksi Interaktif"
        ],
        label_visibility="collapsed"
    )

    arts = load_artifacts()
    df_raw = load_dataset()

    st.markdown("---")
    if arts['loaded']:
        model_name = arts['model_info'].get('model_name', 'Unknown')
        st.markdown(f"""
        <div style='background:#1E2240; border-radius:8px; padding:0.75rem 1rem; font-size:0.78rem'>
            <div style='color:#A8AED4; margin-bottom:4px'>Model Aktif</div>
            <div style='color:#7DD3FC; font-weight:600'>{model_name}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#2D1B1B; border-radius:8px; padding:0.75rem 1rem; font-size:0.78rem'>
            <div style='color:#FCA5A5'>⚠ Model belum dimuat</div>
            <div style='color:#7B82A8; margin-top:4px'>Jalankan notebook terlebih dahulu</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE 1 — RINGKASAN DATASET
# ─────────────────────────────────────────────────────────────────
if "Ringkasan" in page:
    st.markdown("""
    <div class='page-header'>
        <h1>📊 Ringkasan Dataset</h1>
        <p>Overview statistik dan distribusi data Student Placement</p>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is None:
        st.error("Dataset tidak ditemukan di `datasets/student_dataset_10000_rows.csv`")
        st.stop()

    df = df_raw.copy()

    # ── KPI Row ──────────────────────────────────────────────────
    placed_pct = (df['placement_status'] == 'Placed').mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (f"{len(df):,}", "Total Data", f"{df.shape[1]} fitur"),
        (f"{df.duplicated().sum()}", "Duplikat", "Tidak ada duplikasi" if df.duplicated().sum() == 0 else "Perlu ditangani"),
        (f"{df.isna().sum().sum()}", "Missing Value", "Data bersih ✓" if df.isna().sum().sum() == 0 else "Perlu imputing"),
        (f"{placed_pct:.1f}%", "Placed Rate", f"{(df['placement_status']=='Placed').sum():,} mahasiswa")
    ]
    for col, (val, label, sub) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
                <div class='metric-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Distribusi Kelas Target</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])

    with col_a:
        vc = df['placement_status'].value_counts()
        fig, ax = _fig(5, 3.5)
        bars = ax.bar(vc.index, vc.values,
                      color=[PLACED_CLR if x == 'Placed' else NOT_CLR for x in vc.index],
                      width=0.5, zorder=3)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{bar.get_height():,}', ha='center', va='bottom',
                    fontsize=10, fontweight='600', color='#1A1D2E')
        ax.set_title('Jumlah per Kelas', fontweight='600', color='#1A1D2E', pad=10)
        ax.set_xlabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        fig2, ax2 = _fig(5, 3.5)
        wedges, texts, autotexts = ax2.pie(
            vc.values,
            labels=vc.index,
            autopct='%1.1f%%',
            colors=[PLACED_CLR, NOT_CLR],
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight('600')
        ax2.set_title('Proporsi Kelas', fontweight='600', color='#1A1D2E', pad=10)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("<div class='section-title'>Statistik Deskriptif</div>", unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    desc = df[num_cols].describe().T.round(2)
    desc.index.name = 'Fitur'
    st.dataframe(
        desc.style
            .format(precision=2)
            .background_gradient(subset=['mean'], cmap='Blues')
            .set_properties(**{'font-size': '0.8rem'}),
        use_container_width=True
    )

    st.markdown("<div class='section-title'>Preview Data</div>", unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, height=280)


# ─────────────────────────────────────────────────────────────────
# PAGE 2 — EDA
# ─────────────────────────────────────────────────────────────────
elif "Exploratory" in page:
    st.markdown("""
    <div class='page-header'>
        <h1>🔍 Exploratory Data Analysis</h1>
        <p>Distribusi fitur, korelasi, dan perbandingan antar kelas target</p>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is None:
        st.error("Dataset tidak ditemukan.")
        st.stop()

    df = df_raw.copy()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    tab1, tab2, tab3 = st.tabs(["📈 Distribusi Fitur", "🌡️ Korelasi Heatmap", "📦 Boxplot per Kelas"])

    with tab1:
        st.markdown("<div class='section-title'>Histogram Distribusi Fitur Numerik</div>", unsafe_allow_html=True)
        n_cols = 3
        n_rows = (len(num_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.2), facecolor=BG)
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            ax = axes[i]
            ax.set_facecolor(BG)
            placed   = df[df['placement_status'] == 'Placed'][col].dropna()
            not_placed = df[df['placement_status'] == 'Not Placed'][col].dropna()
            ax.hist(not_placed, bins=25, alpha=0.6, color=NOT_CLR, label='Not Placed', edgecolor='none')
            ax.hist(placed,     bins=25, alpha=0.6, color=PLACED_CLR, label='Placed', edgecolor='none')
            ax.set_title(col, fontsize=9, fontweight='600', color='#1A1D2E')
            ax.tick_params(labelsize=7, colors='#888')
            for sp in ax.spines.values(): sp.set_color('#E0E4F8')
            ax.yaxis.grid(True, color=GRID, zorder=0, alpha=0.7)
            ax.set_axisbelow(True)
        for j in range(len(num_cols), len(axes)):
            axes[j].set_visible(False)
        patch_p = mpatches.Patch(color=PLACED_CLR, alpha=0.7, label='Placed')
        patch_n = mpatches.Patch(color=NOT_CLR,    alpha=0.7, label='Not Placed')
        fig.legend(handles=[patch_p, patch_n], loc='lower right',
                   framealpha=0.9, fontsize=8, ncol=2)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        st.markdown("<div class='section-title'>Heatmap Korelasi Antar Fitur Numerik</div>", unsafe_allow_html=True)
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor='#F0F2FF',
            ax=ax, annot_kws={'size': 8}
        )
        ax.set_title('Correlation Matrix', fontweight='600', color='#1A1D2E', pad=12)
        ax.tick_params(labelsize=8, colors='#555')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div class='info-box'>
            💡 <b>Interpretasi:</b> Korelasi tinggi (&gt;0.7) antar fitur dapat mengindikasikan multikolinearitas.
            Fitur <code>exam_score</code> dihapus karena terindikasi <i>data leakage</i> (separasi sempurna terhadap target).
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='section-title'>Perbandingan Distribusi Fitur per Kelas</div>", unsafe_allow_html=True)
        n_rows2 = (len(num_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows2, n_cols, figsize=(14, n_rows2 * 3.5), facecolor=BG)
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            ax = axes[i]
            ax.set_facecolor(BG)
            sns.boxplot(
                data=df, x='placement_status', y=col,
                palette={'Placed': PLACED_CLR, 'Not Placed': NOT_CLR},
                width=0.5, linewidth=1.2, fliersize=3, ax=ax
            )
            ax.set_title(col, fontsize=9, fontweight='600', color='#1A1D2E')
            ax.set_xlabel(''); ax.set_ylabel('')
            ax.tick_params(labelsize=7, colors='#888')
            for sp in ax.spines.values(): sp.set_color('#E0E4F8')
            ax.yaxis.grid(True, color=GRID, zorder=0)
            ax.set_axisbelow(True)
        for j in range(len(num_cols), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ─────────────────────────────────────────────────────────────────
# PAGE 3 — CLUSTERING
# ─────────────────────────────────────────────────────────────────
elif "Clustering" in page:
    st.markdown("""
    <div class='page-header'>
        <h1>🗂️ K-Means Clustering</h1>
        <p>Profil kelompok mahasiswa berdasarkan perilaku akademik</p>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is None or not arts['loaded']:
        st.error("Dataset atau model belum tersedia. Pastikan semua file di folder `model/` sudah ada.")
        st.stop()

    df     = apply_feature_engineering(df_raw.copy())
    kmeans = arts['kmeans']
    scaler_c = arts['scaler_cluster']
    le = arts['le']

    # Encode target jika belum
    if df['placement_status'].dtype == object:
        df['placement_status_enc'] = le.transform(df['placement_status'])
    else:
        df['placement_status_enc'] = df['placement_status']

    drop_cols = ['placement_status', 'placement_status_enc', 'exam_score'] if 'exam_score' in df.columns else ['placement_status', 'placement_status_enc']
    X_full = df.drop(columns=drop_cols, errors='ignore')

    X_scaled = scaler_c.transform(X_full)
    cluster_labels = kmeans.predict(X_scaled)
    df['cluster'] = cluster_labels

    cluster_names = {0: 'Kurang Aktif', 1: 'Aktif & Produktif'}
    df['cluster_name'] = df['cluster'].map(cluster_names)

    tab1, tab2, tab3 = st.tabs(["🔵 PCA Scatter Plot", "📊 Profil Cluster", "🔗 Cross-tab Placement"])

    with tab1:
        st.markdown("<div class='section-title'>Visualisasi Cluster — PCA 2D</div>", unsafe_allow_html=True)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = _fig(8, 5.5)
        cluster_colors = [PALETTE[0], PALETTE[1]]
        for c_idx in sorted(df['cluster'].unique()):
            mask = df['cluster'] == c_idx
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=cluster_colors[c_idx],
                alpha=0.5, s=18,
                label=f"Cluster {c_idx}: {cluster_names.get(c_idx, str(c_idx))}",
                edgecolors='none'
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", color='#555', fontsize=9)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", color='#555', fontsize=9)
        ax.set_title('K-Means Cluster — PCA 2D Projection', fontweight='600', color='#1A1D2E', pad=10)
        ax.legend(framealpha=0.9, fontsize=9)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.xaxis.grid(True, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        col1, col2 = st.columns(2)
        for c_idx in sorted(df['cluster'].unique()):
            cnt = (df['cluster'] == c_idx).sum()
            pct = cnt / len(df) * 100
            with (col1 if c_idx == 0 else col2):
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='color:{cluster_colors[c_idx]}'>{cnt:,}</div>
                    <div class='metric-label'>Cluster {c_idx} — {cluster_names.get(c_idx, '')}</div>
                    <div class='metric-sub'>{pct:.1f}% dari total data</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='section-title'>Profil Rata-rata Fitur per Cluster</div>", unsafe_allow_html=True)
        profile = df.groupby('cluster')[X_full.columns].mean().round(2)

        fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=BG)
        sns.heatmap(
            profile,
            annot=True, fmt='.2f',
            cmap='Blues',
            linewidths=0.5, linecolor='#F0F2FF',
            ax=ax, annot_kws={'size': 8}
        )
        ax.set_title('Rata-rata Fitur per Cluster', fontweight='600', color='#1A1D2E', pad=12)
        ax.set_ylabel('Cluster', color='#555')
        ax.tick_params(labelsize=8, colors='#555')
        y_labels = [f"C{int(t.get_text())} — {cluster_names.get(int(t.get_text()), '')}"
                    for t in ax.get_yticklabels()]
        ax.set_yticklabels(y_labels, rotation=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            profile.rename(index=cluster_names).style
                .format(precision=2)
                .background_gradient(cmap='Blues', axis=0),
            use_container_width=True
        )

    with tab3:
        st.markdown("<div class='section-title'>Distribusi Placement Status per Cluster</div>", unsafe_allow_html=True)
        crosstab = pd.crosstab(df['cluster'], df['placement_status'], normalize='index') * 100
        crosstab.index = [cluster_names.get(i, f'Cluster {i}') for i in crosstab.index]

        fig, ax = _fig(8, 4.5)
        bottom = np.zeros(len(crosstab))
        colors_ct = [NOT_CLR, PLACED_CLR]
        for i, col_name in enumerate(crosstab.columns):
            vals = crosstab[col_name].values
            bars = ax.bar(crosstab.index, vals, bottom=bottom,
                          color=colors_ct[i], label=col_name, width=0.45, zorder=3)
            for bar, b in zip(bars, bottom):
                h = bar.get_height()
                if h > 5:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            b + h/2, f'{h:.1f}%',
                            ha='center', va='center',
                            fontsize=9, fontweight='600', color='white')
            bottom += vals

        ax.set_ylabel('Persentase (%)', color='#555', fontsize=9)
        ax.set_title('Placement Status per Cluster (%)', fontweight='600', color='#1A1D2E', pad=10)
        ax.set_ylim(0, 105)
        ax.legend(title='Status', framealpha=0.9, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            crosstab.round(2).style.format("{:.2f}%").background_gradient(cmap='Greens'),
            use_container_width=True
        )
        st.markdown("""
        <div class='info-box'>
            ℹ️ <b>Catatan:</b> Cluster label <b>tidak</b> digunakan sebagai fitur input model klasifikasi
            untuk mencegah data leakage. Clustering berfungsi sebagai analisis deskriptif/profiling saja.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE 4 — PERFORMA MODEL
# ─────────────────────────────────────────────────────────────────
elif "Performa" in page:
    st.markdown("""
    <div class='page-header'>
        <h1>🤖 Performa Model</h1>
        <p>Evaluasi dan analisis model klasifikasi terbaik</p>
    </div>
    """, unsafe_allow_html=True)

    if not arts['loaded']:
        st.error(f"Model belum dimuat. Error: {arts.get('error', 'Unknown')}")
        st.stop()

    model_obj    = arts['model']
    scaler_model = arts['scaler']
    X_test       = arts['X_test']
    y_test       = arts['y_test']
    model_name   = arts['model_info'].get('model_name', 'Model')
    feat_names   = arts['feature_names']

    X_test_scaled = scaler_model.transform(X_test)
    y_pred = model_obj.predict(X_test_scaled)
    y_prob = model_obj.predict_proba(X_test_scaled)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    f1   = f1_score(y_test, y_pred, average='weighted')
    auc  = roc_auc_score(y_test, y_prob)

    # ── Metric KPIs ──────────────────────────────────────────────
    st.markdown(f"<div class='section-title'>Metrik Evaluasi — {model_name}</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    kpis = [
        (f"{acc:.4f}",  "Accuracy"),
        (f"{prec:.4f}", "Precision (W)"),
        (f"{rec:.4f}",  "Recall (W)"),
        (f"{f1:.4f}",   "F1 Score (W)"),
        (f"{auc:.4f}",  "AUC Score"),
    ]
    for col, (val, label) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='font-size:1.5rem; color:#4361EE'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📉 Confusion Matrix", "📈 ROC Curve", "⭐ Feature Importance"])

    with tab1:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = _fig(5, 4.5)
            disp = ConfusionMatrixDisplay(cm, display_labels=['Not Placed', 'Placed'])
            disp.plot(cmap='Blues', ax=ax, colorbar=False)
            ax.set_title(f'Confusion Matrix\n{model_name}', fontweight='600', color='#1A1D2E', pad=10)
            ax.tick_params(labelsize=9)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            report = classification_report(y_test, y_pred,
                                           target_names=['Not Placed', 'Placed'],
                                           output_dict=True)
            report_df = pd.DataFrame(report).T.round(4)
            st.markdown("**Classification Report**")
            st.dataframe(
                report_df.style.format(precision=4).background_gradient(subset=['f1-score'], cmap='Blues'),
                use_container_width=True
            )

    with tab2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = _fig(7, 5)
        ax.plot(fpr, tpr, color=PALETTE[0], linewidth=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
        ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE[0])
        ax.plot([0, 1], [0, 1], '--', color='#AAAAAA', linewidth=1.2, label='Random Guess')
        ax.set_xlabel('False Positive Rate', color='#555', fontsize=10)
        ax.set_ylabel('True Positive Rate', color='#555', fontsize=10)
        ax.set_title(f'ROC Curve — {model_name}', fontweight='600', color='#1A1D2E', pad=10)
        ax.legend(framealpha=0.9, fontsize=10)
        ax.yaxis.grid(True, color=GRID); ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        if hasattr(model_obj, 'feature_importances_'):
            fi = pd.Series(model_obj.feature_importances_, index=feat_names).sort_values(ascending=True)
            fig, ax = _fig(8, max(4, len(fi) * 0.4))
            colors_fi = [PALETTE[0] if v >= fi.quantile(0.75) else '#A8C1FF' for v in fi.values]
            bars = ax.barh(fi.index, fi.values, color=colors_fi, height=0.6, zorder=3)
            for bar in bars:
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width():.4f}', va='center', fontsize=7.5, color='#444')
            ax.xaxis.grid(True, color=GRID, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel('Importance', color='#555', fontsize=9)
            ax.set_title(f'Feature Importance — {model_name}', fontweight='600', color='#1A1D2E', pad=10)
            ax.tick_params(labelsize=8, colors='#555')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("Feature importance tidak tersedia untuk model ini (XGBoost — gunakan SHAP untuk analisis lanjut).")


# ─────────────────────────────────────────────────────────────────
# PAGE 5 — PREDIKSI INTERAKTIF
# ─────────────────────────────────────────────────────────────────
elif "Prediksi" in page:
    st.markdown("""
    <div class='page-header'>
        <h1>🎯 Prediksi Interaktif</h1>
        <p>Masukkan data mahasiswa untuk memprediksi status penempatan kerja</p>
    </div>
    """, unsafe_allow_html=True)

    if not arts['loaded']:
        st.error(f"Model belum dimuat. Jalankan notebook dan pastikan folder `model/` lengkap.")
        st.stop()

    model_obj    = arts['model']
    scaler_model = arts['scaler']
    le           = arts['le']
    feat_names   = arts['feature_names']
    model_name   = arts['model_info'].get('model_name', 'Model')

    st.markdown(f"""
    <div class='info-box'>
        🤖 Prediksi menggunakan model: <b>{model_name}</b>.
        Isi semua input di bawah lalu klik tombol <b>Prediksi</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Input Form ───────────────────────────────────────────────
    st.markdown("<div class='section-title'>Data Akademik Mahasiswa</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        study_hours    = st.slider("📚 Study Hours / Day", 0.0, 12.0, 5.0, 0.5)
        attendance     = st.slider("📅 Attendance (%)", 0, 100, 80)

    with col2:
        previous_score = st.slider("📝 Previous Score", 0, 100, 70)
        sleep_hours    = st.slider("😴 Sleep Hours / Day", 4.0, 12.0, 7.0, 0.5)
        assignments    = st.slider("✅ Assignments Completed", 0, 20, 10)

    with col3:
        internet_usage = st.slider("🌐 Internet Usage (hrs/day)", 0.0, 12.0, 3.0, 0.5)

    # ── Feature Engineering (sama dengan notebook) ───────────────
    productive_score   = study_hours * 0.5 + assignments * 0.3
    academic_composite = (previous_score / 95) * 0.6 + (attendance / 100) * 0.4
    digital_balance    = study_hours - internet_usage
    sleep_quality      = 1 if 7 <= sleep_hours <= 9 else (0.5 if sleep_hours in [6.0, 10.0] else 0)

    input_dict = {
        'study_hours':          study_hours,
        'attendance':           attendance,
        'previous_score':       previous_score,
        'sleep_hours':          sleep_hours,
        'assignments_completed': assignments,
        'internet_usage':       internet_usage,
        'productive_score':     productive_score,
        'academic_composite':   academic_composite,
        'digital_balance':      digital_balance,
        'sleep_quality':        sleep_quality,
    }

    # Filter hanya kolom yang ada di feature_names
    input_filtered = {k: input_dict[k] for k in feat_names if k in input_dict}
    input_df = pd.DataFrame([input_filtered])[feat_names]

    # ── Fitur turunan preview ─────────────────────────────────────
    with st.expander("🔧 Lihat Fitur Turunan yang Dihitung Otomatis"):
        derived = {
            'productive_score':   f"{productive_score:.2f}  (study×0.5 + assignments×0.3)",
            'academic_composite': f"{academic_composite:.2f}  (prev_score/95×0.6 + attendance/100×0.4)",
            'digital_balance':    f"{digital_balance:.2f}  (study_hours − internet_usage)",
            'sleep_quality':      f"{sleep_quality}  (1=ideal 7–9 jam, 0.5=borderline, 0=kurang/berlebih)"
        }
        for k, v in derived.items():
            st.markdown(f"- **`{k}`** = {v}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict Button ────────────────────────────────────────────
    predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

    if predict_btn:
        input_scaled = scaler_model.transform(input_df)
        pred_enc     = model_obj.predict(input_scaled)[0]
        pred_proba   = model_obj.predict_proba(input_scaled)[0]
        pred_label   = le.inverse_transform([pred_enc])[0]

        prob_placed    = pred_proba[1] * 100
        prob_not       = pred_proba[0] * 100

        st.markdown("<div class='section-title'>Hasil Prediksi</div>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns([1, 1])

        with col_r1:
            if pred_label == 'Placed':
                st.markdown(f"""
                <div class='pred-placed'>
                    <div class='pred-label'>✅ {pred_label}</div>
                    <div class='pred-sub'>Mahasiswa diprediksi berhasil mendapatkan penempatan kerja</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='pred-not-placed'>
                    <div class='pred-label'>❌ {pred_label}</div>
                    <div class='pred-sub'>Mahasiswa diprediksi belum mendapatkan penempatan kerja</div>
                </div>
                """, unsafe_allow_html=True)

        with col_r2:
            fig, ax = _fig(5, 3.5)
            labels  = ['Placed', 'Not Placed']
            vals    = [prob_placed, prob_not]
            colors  = [PLACED_CLR, NOT_CLR]
            bars    = ax.barh(labels, vals, color=colors, height=0.45, zorder=3)
            for bar in bars:
                ax.text(min(bar.get_width() + 1, 98), bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width():.1f}%',
                        va='center', fontsize=12, fontweight='700', color='#1A1D2E')
            ax.set_xlim(0, 105)
            ax.xaxis.grid(True, color=GRID, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel('Probabilitas (%)', color='#555', fontsize=9)
            ax.set_title('Probabilitas Prediksi', fontweight='600', color='#1A1D2E', pad=8)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # ── Rekomendasi ───────────────────────────────────────────
        st.markdown("<div class='section-title'>Rekomendasi</div>", unsafe_allow_html=True)
        recs = []
        if study_hours < 4:
            recs.append("📚 Tingkatkan jam belajar — idealnya minimal **4–6 jam/hari**.")
        if attendance < 75:
            recs.append("📅 Tingkatkan kehadiran — target minimal **75%** untuk performa optimal.")
        if internet_usage > study_hours:
            recs.append("🌐 Kurangi penggunaan internet yang tidak produktif — saat ini melebihi jam belajar.")
        if sleep_quality == 0:
            recs.append("😴 Atur waktu tidur ke **7–9 jam/hari** untuk kualitas tidur optimal.")
        if assignments < 8:
            recs.append("✅ Perbanyak penyelesaian tugas — saat ini hanya **{} dari 20**.".format(assignments))
      

        if pred_label == 'Placed' and not recs:
            st.success("🎉 Profil mahasiswa sangat baik! Pertahankan konsistensi belajar dan kehadiran.")
        elif recs:
            for r in recs:
                st.markdown(f"- {r}")
        else:
            st.info("Terus pertahankan performa akademik yang baik.")
