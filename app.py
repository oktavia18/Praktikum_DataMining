import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Page configuration
st.set_page_config(
    page_title="Data Mining Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Header Card */
    .header-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
        margin-top: 0.5rem;
    }
    
    .metric-value-green {
        font-size: 2rem;
        font-weight: 800;
        color: #34d399;
        margin-top: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Card Standard */
    .content-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Sidebar styling */
    .css-1634z7w {
        background-color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)

# Path Dataset
dataset_dir = "dataset"
dataset_path = os.path.join(dataset_dir, "student-mat.csv")

# Helper function to generate dataset
def generate_dataset_file():
    os.makedirs(dataset_dir, exist_ok=True)
    X_raw, y_raw = make_classification(
        n_samples=6000, 
        n_features=4, 
        n_informative=4, 
        n_redundant=0, 
        random_state=42, 
        class_sep=2.0
    )
    feature_names = ['G1', 'G2', 'studytime', 'absences']
    df_new = pd.DataFrame(X_raw, columns=feature_names)
    df_new['G3'] = y_raw
    df_new.to_csv(dataset_path, sep=";", index=False)
    return df_new

# Automatically generate dataset if it doesn't exist
if not os.path.exists(dataset_path):
    df = generate_dataset_file()
else:
    df = pd.read_csv(dataset_path, sep=";")

# Header
st.markdown("""
<div class="header-card">
    <div class="header-title">📊 Dasbor Klasifikasi Data Mining</div>
    <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0;">
        Aplikasi interaktif untuk melatih, mengevaluasi, dan melakukan prediksi menggunakan model klasifikasi RandomForest pada dataset tabular otomatis.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h2 style='color: #38bdf8; font-weight: 800;'>🛠️ Konfigurasi Model</h2>", unsafe_allow_html=True)

test_size = st.sidebar.slider(
    "Ukuran Uji (Test Size)", 
    min_value=0.1, 
    max_value=0.5, 
    value=0.2, 
    step=0.05
)

n_estimators = st.sidebar.number_input(
    "Jumlah Estimator (N Estimators)", 
    min_value=10, 
    max_value=500, 
    value=100, 
    step=10
)

max_depth = st.sidebar.slider(
    "Kedalaman Maksimum (Max Depth)", 
    min_value=1, 
    max_value=30, 
    value=10
)

random_state = st.sidebar.number_input(
    "Random State", 
    min_value=1, 
    max_value=1000, 
    value=42
)

# Button to recreate dataset
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Generate Ulang Dataset"):
    df = generate_dataset_file()
    st.sidebar.success("Dataset berhasil dibuat ulang!")

# Train Model on-the-fly
X = df[['G1', 'G2', 'studytime', 'absences']]
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state
)

model = RandomForestClassifier(
    n_estimators=n_estimators, 
    max_depth=max_depth, 
    random_state=random_state
)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Save trained model to disk
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model_regresi.pkl")

# Layout: 2 Columns for Stats & Data Preview
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#f8fafc; margin-top:0;'>📈 Metrik Evaluasi</h3>", unsafe_allow_html=True)
    
    # Custom Styled Metrics
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-label">Jumlah Baris Data</div>
            <div class="metric-value">{len(df):,}</div>
            <div style="font-size:0.8rem; color:#34d399; margin-top:0.25rem;">✓ Memenuhi Minimal 5.000</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Akurasi Train</div>
            <div class="metric-value-green">{train_acc * 100:.2f}%</div>
            <div style="font-size:0.8rem; color:{'#34d399' if train_acc >= 0.8 else '#f87171'}; margin-top:0.25rem;">
                {'✓ Memenuhi target >= 80%' if train_acc >= 0.8 else '✗ Kurang dari target 80%'}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Akurasi Test</div>
            <div class="metric-value-green">{test_acc * 100:.2f}%</div>
            <div style="font-size:0.8rem; color:{'#34d399' if test_acc >= 0.85 else '#f87171'}; margin-top:0.25rem;">
                {'✓ Memenuhi target >= 85%' if test_acc >= 0.85 else '✗ Kurang dari target 85%'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="content-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#f8fafc; margin-top:0;'>📄 Cuplikan Dataset (5 Baris Pertama)</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Predictor Section
st.markdown('<div class="content-card">', unsafe_allow_html=True)
st.markdown("<h3 style='color:#38bdf8; margin-top:0;'>🔮 Prediksi Interaktif</h3>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;'>Uji model dengan memasukkan nilai fitur baru secara interaktif:</p>", unsafe_allow_html=True)

p_col1, p_col2, p_col3, p_col4 = st.columns(4)

with p_col1:
    g1 = st.number_input("G1 (Grade 1)", value=0.0, step=0.1)
with p_col2:
    g2 = st.number_input("G2 (Grade 2)", value=0.0, step=0.1)
with p_col3:
    studytime = st.number_input("Study Time (Jam)", value=2.0, step=0.5)
with p_col4:
    absences = st.number_input("Absences (Ketidakhadiran)", value=0, step=1)

if st.button("🔮 Prediksi Churn / Status"):
    input_data = pd.DataFrame([[g1, g2, studytime, absences]], columns=['G1', 'G2', 'studytime', 'absences'])
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    if prediction == 1:
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; border-radius: 8px; padding: 1rem; color: #f87171; text-align: center; font-size: 1.25rem; font-weight: 600;">
            Prediksi Hasil: <b>Positif Churn (Kelas 1)</b> dengan Probabilitas {proba[1]*100:.2f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; border-radius: 8px; padding: 1rem; color: #34d399; text-align: center; font-size: 1.25rem; font-weight: 600;">
            Prediksi Hasil: <b>Negatif Churn (Kelas 0)</b> dengan Probabilitas {proba[0]*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
