import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# ======================
# CONFIG (HARUS PALING ATAS UNTUK PERINTAH st.*)
# ======================
st.set_page_config(page_title="Prediksi TGM Nasional (ARIMA)", layout="wide")

DATA_PATH = "data/Datamlops.xlsx"
DB_PATH = "reading.db"
TARGET_COL = "TGM"
PROVINCE_COL = "Provinsi"  # sesuaikan dengan nama kolom provinsi di Excel

MODEL_DIR = "models"
ARIMA_PATH = os.path.join(MODEL_DIR, "arima_tgm.pkl")

# ======================
# HALAMAN LOGIN SEDERHANA (TANPA USERNAME/PASSWORD)
# ======================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.title("🔐 Login Aplikasi Prediksi TGM")
    st.write("Klik tombol di bawah untuk masuk ke aplikasi.")

    if st.button("Login"):
        st.session_state["logged_in"] = True
        st.rerun()

    st.stop()

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

# ======================
# LOAD DATA & HITUNG TGM
# ======================
@st.cache_data
def load_data_with_tgm() -> pd.DataFrame:
    try:
        df = pd.read_excel(DATA_PATH)
    except FileNotFoundError:
        st.error("File 'data/Datamlops.xlsx' tidak ditemukan. Pastikan path dan nama file sudah benar.")
        st.stop()

    df.rename(
        columns={
            "Reading Frequency per week": "RF",
            "Number of Readings per Quarter": "NR",
            "Daily Reading Duration (in minutes)": "DRD",
            "Internet Access Frequency per Week": "IAF",
            "Daily Internet Duration (in minutes)": "DID",
        },
        inplace=True,
    )

    required_cols = [PROVINCE_COL, "RF", "DRD", "NR", "IAF", "DID", "Year"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Kolom berikut belum ada di data: {missing}")
        st.stop()

    df[TARGET_COL] = (
        0.3 * df["RF"]
        + 0.3 * df["DRD"]
        + 0.3 * df["NR"]
        + 0.05 * df["IAF"]
        + 0.05 * df["DID"]
    )

    return df


df = load_data_with_tgm()
province_list = sorted(df[PROVINCE_COL].unique().tolist())

# ======================
# DATABASE SETUP
# ======================
def init_db():
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        c = conn.cursor()

        # tabel hasil forecast nasional
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS tgm_prediction (
                year INTEGER PRIMARY KEY,
                tgm  REAL
            )
            """
        )

        # tabel input user: struktur sesuai form user
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS reading (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                province TEXT,
                year     INTEGER,
                rf       REAL,
                drd      REAL,
                nr       REAL,
                iaf      REAL,
                did      REAL,
                tgm      REAL
            )
            """
        )

        conn.commit()


init_db()

# ======================
# SIMPAN HASIL FORECAST KE DB
# ======================
def save_forecasts_to_db(years, values):
    """Simpan hasil prediksi ARIMA ke tabel tgm_prediction."""
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        c = conn.cursor()
        for y, v in zip(years, values):
            c.execute(
                """
                INSERT OR REPLACE INTO tgm_prediction (year, tgm)
                VALUES (?, ?)
                """,
                (int(y), float(v)),
            )
        conn.commit()

# ======================
# MODEL HANDLING
# ======================
def fit_or_load_arima(series_values: np.ndarray):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(ARIMA_PATH):
        try:
            model_fit = joblib.load(ARIMA_PATH)
            return model_fit
        except Exception:
            pass

    model = ARIMA(series_values, order=(1, 1, 1))
    model_fit = model.fit()
    joblib.dump(model_fit, ARIMA_PATH)
    return model_fit


def maybe_retrain_model_with_user_data(trend_df: pd.Series):
    """Retrain ARIMA jika data user di tabel reading sudah ≥ 1000 baris."""
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM reading")
        count = c.fetchone()[0]

        if count < 1000:
            return

        c.execute("SELECT year, tgm FROM reading ORDER BY year")
        user_rows = c.fetchall()

    user_years = [r[0] for r in user_rows]
    user_tgm = [r[1] for r in user_rows]

    all_years = list(trend_df.index) + user_years
    all_tgm = list(trend_df.values) + user_tgm

    series_values = np.array(all_tgm, dtype=float)

    model = ARIMA(series_values, order=(1, 1, 1))
    model_fit = model.fit()
    joblib.dump(model_fit, ARIMA_PATH)

    st.info(f"Model ARIMA diretrain otomatis menggunakan {len(all_tgm)} titik (historis + {count} data user).")

# ======================
# UI: TAMPILAN UTAMA
# ======================
st.title("📚 Prediksi Tingkat Gemar Membaca Nasional (ARIMA)")
st.write(
    "Aplikasi ini menghitung Tingkat Gemar Membaca (TGM) menggunakan rumus berbobot "
    "dan memprediksi tren nasional ke depan dengan model time series ARIMA."
)

st.divider()

# ======================
# TREN NASIONAL TGM (HISTORIS)
# ======================
trend_df = df.groupby("Year")[TARGET_COL].mean().sort_index()

st.subheader("📈 Tren Historis TGM Nasional")

if trend_df.empty:
    st.warning("Data TGM historis kosong. Periksa kembali file Excel Anda.")
else:
    fig, ax = plt.subplots()
    ax.plot(trend_df.index, trend_df.values, marker="o")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Nilai TGM")
    ax.set_title("Tren Historis TGM Nasional")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig)

# ======================
# EVALUASI MODEL (TRAIN/TEST SPLIT)
# ======================
values = trend_df.values.astype(float)
years = trend_df.index.values

if len(values) >= 6:  # minimal titik data
    split_idx = int(0.8 * len(values))
    train_values = values[:split_idx]
    test_values = values[split_idx:]
    train_years = years[:split_idx]
    test_years = years[split_idx:]

    try:
        eval_model = ARIMA(train_values, order=(1, 1, 1)).fit()
        h = len(test_values)
        arima_forecast = eval_model.forecast(steps=h)

        naive_forecast_vals = np.repeat(train_values[-1], h)

        arima_mae = mean_absolute_error(test_values, arima_forecast)
        arima_mape = mean_absolute_percentage_error(test_values, arima_forecast)
        naive_mae = mean_absolute_error(test_values, naive_forecast_vals)
        naive_mape = mean_absolute_percentage_error(test_values, naive_forecast_vals)

        st.subheader("📊 Evaluasi Model (Train/Test)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ARIMA MAE", f"{arima_mae:.2f}")
            st.metric("ARIMA MAPE", f"{arima_mape * 100:.2f}%")
        with col2:
            st.metric("Naive MAE", f"{naive_mae:.2f}")
            st.metric("Naive MAPE", f"{naive_mape * 100:.2f}%")

    except Exception as e:
        st.warning(f"Gagal menghitung evaluasi model: {e}")

maybe_retrain_model_with_user_data(trend_df)

# ======================
# MODEL ARIMA & FORECAST
# ======================
st.subheader("🔮 Prediksi TGM Nasional (ARIMA)")

col_left, col_right = st.columns(2)
with col_left:
    default_steps = 2  # misal: 2 tahun ke depan
    steps = st.number_input(
        "Jumlah tahun ke depan yang akan diprediksi:",
        min_value=1,
        max_value=10,
        value=default_steps,
        step=1,
    )

with col_right:
    st.info(
        "Model ARIMA menggunakan parameter tetap (1, 1, 1). "
        "Dengan data historis, model akan memprediksi tahun-tahun berikutnya."
    )

retrain = st.checkbox("🔄 Paksa retrain model ARIMA dari data historis", value=False)

try:
    series_values = trend_df.values.astype(float)
    if retrain and os.path.exists(ARIMA_PATH):
        os.remove(ARIMA_PATH)
    model_fit = fit_or_load_arima(series_values)
except Exception as e:
    st.error(f"Terjadi error saat fitting / load ARIMA: {e}")
else:
    forecast = model_fit.forecast(steps=int(steps))

    last_year = int(trend_df.index.max())
    future_years = np.arange(last_year + 1, last_year + 1 + int(steps))

    fig2, ax2 = plt.subplots()
    ax2.plot(trend_df.index, trend_df.values, marker="o", label="Historis")
    ax2.plot(
        future_years,
        forecast,
        marker="o",
        linestyle="--",
        color="orange",
        label="Prediksi ARIMA",
    )
    ax2.set_xlabel("Tahun")
    ax2.set_ylabel("Nilai TGM")
    ax2.set_title("Prediksi TGM Nasional dengan ARIMA")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig2)

    st.markdown("### 📌 Hasil Prediksi TGM per Tahun")
    for y, v in zip(future_years, forecast):
        st.success(f"Prediksi TGM Nasional Tahun {int(y)}: **{v:.2f}**")

    if st.button("💾 Simpan Hasil Prediksi ke Database"):
        save_forecasts_to_db(future_years, forecast)
        st.info("Hasil prediksi berhasil disimpan ke tabel tgm_prediction di SQLite.")

# ======================
# FORM INPUT DATA USER
# ======================
st.divider()
st.subheader("📝 Input Data Tingkat Gemar Membaca (User)")

with st.form("user_input_form"):
    prov_in = st.selectbox("Provinsi", options=province_list)
    year_in = st.number_input("Tahun", min_value=2000, max_value=2100, value=2025, step=1)
    rf_in = st.number_input("RF - Reading Frequency per week", min_value=0.0, value=1.0)
    drd_in = st.number_input("DRD - Daily Reading Duration (minutes)", min_value=0.0, value=10.0)
    nr_in = st.number_input("NR - Number of Readings per Quarter", min_value=0.0, value=1.0)
    iaf_in = st.number_input("IAF - Internet Access Frequency per Week", min_value=0.0, value=1.0)
    did_in = st.number_input("DID - Daily Internet Duration (minutes)", min_value=0.0, value=10.0)

    submitted = st.form_submit_button("Simpan data user ke database")

if submitted:
    tgm_in = (
        0.3 * rf_in
        + 0.3 * drd_in
        + 0.3 * nr_in
        + 0.05 * iaf_in
        + 0.05 * did_in
    )

    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO reading (province, year, rf, drd, nr, iaf, did, tgm)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prov_in,
                int(year_in),
                float(rf_in),
                float(drd_in),
                float(nr_in),
                float(iaf_in),
                float(did_in),
                float(tgm_in),
            ),
        )
        conn.commit()

    st.success(f"Data untuk provinsi {prov_in} tersimpan. TGM = {tgm_in:.2f}")

# ======================
# OPSIONAL: LIHAT DATA MENTAH
# ======================
st.divider()
with st.expander("📄 Lihat Data TGM Nasional (Historis)"):
    st.dataframe(
        trend_df.reset_index().rename(columns={"Year": "Tahun", TARGET_COL: "Rata-rata TGM"})
    )

with st.expander("📄 Lihat Data Mentah (Per Baris)"):
    st.dataframe(df.head(100))
