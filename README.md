# Kelompok-6-MlOps-RB-2025

# 📚 Prediksi Tingkat Gemar Membaca (TGM) 

## Project Overview

Proyek ini merupakan implementasi **pipeline Machine Learning Operations (MLOps)** secara end-to-end untuk melakukan **prediksi Tingkat Gemar Membaca (TGM) Nasional** menggunakan pendekatan **time series forecasting (ARIMA)**.

Aplikasi dikembangkan berbasis **Streamlit** dan mendukung proses lengkap mulai dari pengelolaan data, training model, deployment aplikasi, hingga *continuous retraining* berdasarkan data baru dari pengguna.

Pipeline ini dirancang sederhana namun representatif untuk studi kasus akademik MLOps, dengan fokus pada *reproducibility*, *automation*, dan *feedback loop* antara data dan model.

---

## Objectives

Tujuan dari proyek ini adalah:

* Menghitung nilai Tingkat Gemar Membaca (TGM) berdasarkan indikator membaca dan penggunaan internet.
* Mengembangkan model time series forecasting untuk memprediksi TGM nasional.
* Mengimplementasikan pipeline MLOps end-to-end berbasis aplikasi.
* Menyimpan data dan hasil prediksi secara terstruktur menggunakan database.
* Menerapkan konsep continuous training (retraining) berbasis data user.
* Menyediakan aplikasi menggunakan Docker.

---

## Dataset Description

Dataset utama berasal dari Kaggle yang berisi indikator literasi dan kebiasaan membaca masyarakat per provinsi dan per tahun.

### Fitur Utama:

* Reading Frequency per Week (RF)
* Daily Reading Duration (DRD)
* Number of Readings per Quarter (NR)
* Internet Access Frequency per Week (IAF)
* Daily Internet Duration (DID)
* Provinsi
* Tahun

### Target:

* **Tingkat Gemar Membaca (TGM)**

Nilai TGM dihitung menggunakan rumus berbobot dan kemudian diagregasi menjadi **rata-rata nasional per tahun** untuk keperluan pemodelan time series.

 🧮 Rumus Tingkat Gemar Membaca (TGM)

TGM dihitung menggunakan rumus berbobot berikut:

```
TGM = 0.3 × RF + 0.3 × DRD + 0.3 × NR + 0.05 × IAF + 0.05 × DID
```

Keterangan:

* **RF**  : Reading Frequency per Week
* **DRD** : Daily Reading Duration (menit)
* **NR**  : Number of Readings per Quarter
* **IAF** : Internet Access Frequency per Week
* **DID** : Daily Internet Duration (menit)

Selain data historis, sistem juga menerima **data input dari user**, yang disimpan ke database dan dapat digunakan untuk retraining model.

---

## MLOps Workflow

Pipeline MLOps pada proyek ini mencakup tahapan berikut:

### 1️⃣ Data Ingestion

* Data historis dibaca dari file Excel (`Datamlops.xlsx`).
* Data baru diterima melalui form input pengguna pada aplikasi Streamlit.
* Semua data user disimpan ke database **SQLite**.

---

### 2️⃣ Data Processing & Feature Engineering

* Validasi dan standarisasi kolom data.
* Perhitungan nilai TGM menggunakan rumus berbobot.
* Agregasi data menjadi tren TGM nasional per tahun.

---

### 3️⃣ Model Training

* Model yang digunakan adalah **ARIMA (1,1,1)**.
* Training awal dilakukan menggunakan data historis nasional.
* Model disimpan sebagai artefak (`arima_tgm.pkl`).

---

### 4️⃣ Continuous Retraining

* Sistem memonitor jumlah data pada tabel input user.
* Jika jumlah data user ≥ **1000 baris**:

  * Data historis dan data user digabung.
  * Model ARIMA diretrain secara otomatis.
  * Model lama digantikan oleh model baru.

---

### 5️⃣ Model Deployment

* Model dideploy secara *embedded* di dalam aplikasi Streamlit.
* Tidak memerlukan inference service terpisah.
* Model dimuat otomatis saat aplikasi dijalankan.

---

### 6️⃣ Model Inference

* User menentukan jumlah tahun prediksi.
* Model menghasilkan prediksi TGM nasional untuk tahun-tahun mendatang.
* Hasil ditampilkan dalam bentuk grafik dan nilai numerik.

---

### 7️⃣ Prediction Storage

* Hasil prediksi dapat disimpan ke database SQLite.
* Data ini dapat digunakan untuk evaluasi dan monitoring tren.

---

### 8️⃣ Application Deployment

Aplikasi dapat dijalankan melalui dua metode:

* **Local Deployment** menggunakan Streamlit
* **Docker-based Deployment** untuk lingkungan terisolasi dan portabel

---

### 9️⃣ Monitoring & Feedback Loop

* Monitoring dilakukan melalui:

  * Perubahan tren prediksi
  * Penambahan data user
* Feedback loop:

```
Data Baru → Retraining → Model Baru → Prediksi Baru
```

---

## Project Structure

```
Aplikasi-Model-Prediksi-Tingkat-Kegemaran-Membaca/
│
├── data/               # Dataset historis
├── models/             # Model ARIMA tersimpan
├── .streamlit/         # Konfigurasi Streamlit
├── app.py              # Aplikasi utama
├── reading.db          # Database SQLite
├── Dockerfile          # Konfigurasi Docker
├── requirements.txt    # Dependency proyek
└── README.md           # Dokumentasi
```

---

## Deployment

Aplikasi dideploy sebagai aplikasi web menggunakan **Streamlit** dan dapat dijalankan di dalam **Docker container**.

### Tahapan Singkat Deployment

1. Build Docker image
2. Jalankan container
3. Akses aplikasi melalui browser (port 8501)

---

## Tools & Technologies

* Python
* Streamlit
* Pandas & NumPy
* Statsmodels (ARIMA)
* SQLite
* Docker
* Matplotlib

---

## Expected Output

Output dari sistem berupa:

* Visualisasi tren historis TGM nasional
* Prediksi nilai TGM nasional untuk tahun mendatang
* Penyimpanan data input user dan hasil prediksi

---

## Tim
1. Tessa Kania Sagala (122450040)
2. Renta Siahaan (122450070)
3. Nabila Zakiyah Zahra (122450139)
