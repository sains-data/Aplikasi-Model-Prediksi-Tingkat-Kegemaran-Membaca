# Kelompok-6-MlOps-RB-2025

# 📚 Aplikasi Model Prediksi Tingkat Kegemaran Membaca (TGM) 

## Project Overview

Proyek ini merupakan implementasi **pipeline Machine Learning Operations (MLOps)** secara end-to-end untuk melakukan **prediksi Tingkat Gemar Membaca (TGM) Nasional** menggunakan pendekatan **time series forecasting (ARIMA)**.

Aplikasi dikembangkan berbasis **Streamlit** dan mendukung proses lengkap mulai dari pengelolaan data, training model, deployment aplikasi, hingga *continuous retraining* berdasarkan data baru dari pengguna.

Pipeline ini dirancang sederhana namun representatif untuk studi kasus akademik MLOps, dengan fokus pada *reproducibility*, *automation*, dan *feedback loop* antara data dan model.

---


## ML Canvas

**Background**
Tingkat Gemar Membaca (TGM) nasional perlu dimonitor untuk mendukung kebijakan peningkatan literasi di Indonesia, namun analisis masih banyak dilakukan secara manual dan belum terotomasi.
Data survei kebiasaan membaca dan penggunaan internet per provinsi serta per tahun memungkinkan pembangunan sistem prediksi TGM nasional yang terukur dan berulang.

**Value Proposition**
Aplikasi ini memberikan dashboard interaktif yang menghitung TGM berbasis beberapa indikator kebiasaan membaca dan penggunaan internet, lalu memprediksi tren nasional ke depan.
Stakeholder (peneliti, pemerintah, kampus) dapat melihat tren historis, melakukan simulasi input data baru, dan memanfaatkan hasil prediksi yang tersimpan di database untuk analisis lanjutan.

**Objective**
Membangun sistem MLOps end‑to‑end untuk memprediksi TGM nasional: mulai dari pengambilan data, pemrosesan, training model time series, hingga deployment dan monitoring hasil.
Secara khusus, sistem harus mampu melakukan online inference melalui web app dan mendukung retrain otomatis saat data pengguna bertambah.

**Solution**
Solusi berupa aplikasi Streamlit yang membaca data survei dari file Excel, menghitung skor TGM per baris, mengagregasikan TGM nasional per tahun, lalu melatih model ARIMA untuk memprediksi beberapa tahun ke depan.
Aplikasi dikemas dalam Docker container, menggunakan SQLite untuk menyimpan data input pengguna dan hasil prediksi, serta menyediakan antarmuka untuk melihat tren historis dan hasil forecast.

**Data**
Dataset utama berasal dari Kaggle yang berisi indikator literasi dan kebiasaan membaca masyarakat per provinsi dan per tahun. 
Feature engineering dilakukan dengan menghitung skor TGM sebagai kombinasi berbobot dari fitur‑fitur tersebut, yang kemudian digunakan sebagai target untuk analisis tren dan peramalan.

**Metrics**
Metrik utama di aplikasi adalah nilai TGM rata‑rata nasional per tahun dan nilai prediksi TGM untuk tahun‑tahun berikutnya.
Untuk evaluasi model, metrik error yang digunakan (atau direncanakan di notebook terpisah) adalah misalnya MAE/MAPE antara prediksi ARIMA dan data historis, serta perbandingan dengan baseline sederhana (naive atau moving average).

**Evaluation**
Model dievaluasi dengan membandingkan prediksi pada periode uji terhadap data TGM historis menggunakan metrik error dan visualisasi kurva tren historis vs prediksi.
Kualitas sistem juga dievaluasi dari kemudahan deployment dengan Docker, kemampuan menyimpan data dan prediksi di SQLite, serta mekanisme retrain otomatis ketika data user mencapai ambang tertentu.

**Modeling**
Model utama yang digunakan adalah ARIMA (1,1,1) pada deret waktu TGM nasional per tahun, dengan opsi retrain ketika data historis dan data input pengguna digabung.
Ke depan dapat ditambahkan model pembanding (naive, moving average, atau model time series lain) dan eksperimen tuning hyperparameter ARIMA untuk menurunkan error.

**Inference (offline/online)**
Inference online dilakukan langsung di aplikasi Streamlit: pengguna memilih horizon tahun ke depan, aplikasi menjalankan model ARIMA dan menampilkan grafik serta nilai prediksi.
Inference offline dimungkinkan dengan memanfaatkan model .pkl yang tersimpan di folder models/ dan data yang tersimpan di SQLite untuk analisis atau pelatihan ulang di luar aplikasi.​

**Tujuan dari proyek ini adalah:**
* Menghitung nilai Tingkat Gemar Membaca (TGM) berdasarkan indikator membaca dan penggunaan internet.
* Mengembangkan model time series forecasting untuk memprediksi TGM nasional.
* Mengimplementasikan pipeline MLOps end-to-end berbasis aplikasi.
* Menyimpan data dan hasil prediksi secara terstruktur menggunakan database.
* Menerapkan konsep continuous training (retraining) berbasis data user.
* Menyediakan aplikasi yang dapat dijalankan secara portabel menggunakan Docker.

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
├── notebooks/          # Model Evaluasi 
├── .streamlit/         # Konfigurasi Streamlit
├── app.py              # Aplikasi utama
├── reading.db          # Database SQLite
├── Dockerfile          # Konfigurasi Docker
├── requirements.txt    # Dependency proyek
└── README.md           # Dokumentasi
```

---

## Deployment

## Cara Menjalankan dengan Git & Docker (PowerShell)

### 1. Clone repository dari GitHub

Buka PowerShell, lalu jalankan perintah berikut:

```bash
# Pindah ke folder kerja (boleh diganti sesuai kebutuhan)
cd $HOME\Documents

# Clone repository
git clone https://github.com/sains-data/Aplikasi-Model-Prediksi-Tingkat-Kegemaran-Membaca.git

# Masuk ke folder project
cd Aplikasi-Model-Prediksi-Tingkat-Kegemaran-Membaca
```

Pastikan di dalam folder ini sudah ada file `Dockerfile`, `app.py`, `requirements.txt`, dan folder `data/`.

### 2. Build Docker image

Masih di folder project yang sama, jalankan:

```bash
# Build image dengan nama tgm-app
docker build -t tgm-app .
```

Perintah ini akan:
* Menggunakan `python:3.11-slim` sebagai base image
* Menginstall semua dependency dari `requirements.txt`
* Menyalin seluruh kode aplikasi ke dalam image dan menyiapkan Streamlit untuk dijalankan

### 3. Jalankan container

Setelah build selesai tanpa error:

```bash
docker run -p 8501:8501 tgm-app
```

**Penjelasan singkat:**
* `-p 8501:8501` memetakan port 8501 di container ke port 8501 di komputer kamu
* Container otomatis menjalankan perintah `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`

### 4. Akses aplikasi di browser

Buka browser dan kunjungi:

```
http://localhost:8501
```

Di halaman ini kamu bisa:
* Melihat tren historis TGM nasional
* Melihat evaluasi model (MAE/MAPE) vs baseline
* Mengatur horizon prediksi dan menyimpan hasil ke database
* Mengisi form input data user yang akan tersimpan di `reading.db`

### 5. (Opsional) Menghentikan container

Tekan `Ctrl + C` di PowerShell tempat container berjalan, atau cari container-nya lalu hentikan dengan:

```bash
docker ps        # melihat container yang sedang berjalan
docker stop <CONTAINER_ID>
```

---

## Troubleshooting

Jika mengalami masalah:
* Pastikan Docker Desktop sudah berjalan
* Pastikan port 8501 tidak digunakan oleh aplikasi lain
* Periksa log error saat build atau run container

## Kontribusi

Silakan buat issue atau pull request jika ingin berkontribusi pada project ini.

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
