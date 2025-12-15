import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import joblib

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Prediksi TGM Nasional (ARIMA)", layout="wide")

DATA_PATH = "data/Datamlops.xlsx"
DB_PATH = "reading.db"
TARGET_COL = "TGM"

MODEL_DIR = "models"
ARIMA_PATH = os.path.join(MODEL_DIR, "arima_tgm.pkl")

# ======================
# LOAD DATA & HITUNG TGM SESUAI RUMUS
# ======================
@st.cache_data
def load_data_with_tgm() -> pd.DataFrame:
    """Load data dari Excel dan hitung TGM dengan rumus resmi."""
    try:
        df = pd.read_excel(DATA_PATH)
    except FileNotFoundError:
        st.error("File 'data/Datamlops.xlsx' tidak ditemukan. Pastikan path dan nama file sudah benar.")
        st.stop()

    # Pastikan nama kolom konsisten
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

    # Cek kolom wajib
    required_cols = ["RF", "DRD", "NR", "IAF", "DID", "Year"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Kolom berikut belum ada di data: {missing}")
        st.stop()

    # Hitung TGM SESUAI RUMUS RESMI (bobot bisa disesuaikan dengan jurnal)
    df[TARGET_COL] = (
        0.3 * df["RF"]
        + 0.3 * df["DRD"]
        + 0.3 * df["NR"]
        + 0.05 * df["IAF"]
        + 0.05 * df["DID"]
    )

    return df


df = load_data_with_tgm()

# ======================
# DATABASE SETUP
# ======================
def init_db():
    """Inisialisasi tabel SQLite untuk menyimpan hasil prediksi TGM."""
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tgm_prediction (
                year INTEGER PRIMARY KEY,
                tgm REAL
            )
            """
        )
        conn.commit()


def save_forecasts_to_db(years: np.ndarray, values: np.ndarray) -> None:
    """Simpan hasil forecast ke tabel tgm_prediction (upsert by year)."""
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        cursor = conn.cursor()
        for y, v in zip(years, values):
            cursor.execute(
                """
                INSERT INTO tgm_prediction (year, tgm)
                VALUES (?, ?)
                ON CONFLICT(year) DO UPDATE SET tgm = excluded.tgm
                """,
                (int(y), float(v)),
            )
        conn.commit()


init_db()

# ======================
# MODEL HANDLING
# ======================
def fit_or_load_arima(series_values: np.ndarray):
    """Load model ARIMA dari disk jika ada, kalau belum fit dan simpan."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(ARIMA_PATH):
        try:
            model_fit = joblib.load(ARIMA_PATH)
            return model_fit
        except Exception:
            # Kalau file korup, retrain
            pass

    model = ARIMA(series_values, order=(1, 1, 1))
    model_fit = model.fit()
    joblib.dump(model_fit, ARIMA_PATH)
    return model_fit

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
# MODEL ARIMA & FORECAST
# ======================
st.subheader("🔮 Prediksi TGM Nasional (ARIMA)")

min_points = 5  # jumlah minimum titik data untuk ARIMA
if len(trend_df) < min_points:
    st.warning(
        f"Data historis TGM baru {len(trend_df)} titik. "
        f"Dibutuhkan minimal {min_points} tahun untuk pemodelan ARIMA yang lebih stabil."
    )
else:
    # Input steps prediksi dari user
    col_left, col_right = st.columns(2)
    with col_left:
        default_steps = 2
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
            "Untuk paper/jurnal, bisa ditambah analisis pemilihan orde (AIC/BIC)."
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

        # Plot historis + forecast
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

        # Tampilkan hasil numerik
        st.markdown("### 📌 Hasil Prediksi TGM per Tahun")
        for y, v in zip(future_years, forecast):
            st.success(f"Prediksi TGM Nasional Tahun {int(y)}: **{v:.2f}**")

        # Tombol simpan ke DB
        if st.button("💾 Simpan Hasil Prediksi ke Database"):
            save_forecasts_to_db(future_years, forecast)
            st.info("Hasil prediksi berhasil disimpan ke tabel `tgm_prediction` di SQLite.")

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
