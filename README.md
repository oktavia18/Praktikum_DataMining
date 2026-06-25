# Praktikum_DataMining

# 📊 Dasbor Klasifikasi Data Mining (RandomForest Classifier Dashboard)

Repositori ini berisi proyek **Data Mining** berupa dasbor interaktif berbasis web menggunakan **Streamlit** untuk melatih, mengevaluasi, dan melakukan prediksi klasifikasi menggunakan algoritma **RandomForestClassifier**. 

Dasbor ini dirancang dengan antarmuka premium bertema gelap modern (*dark mode*), memanfaatkan efek *glassmorphism* dan visualisasi metrik yang interaktif untuk memberikan pengalaman pengguna yang profesional dan responsif.

---

## 🚀 Fitur Utama

- **Antarmuka Premium (Premium UI/UX)**: Menggunakan tema gradasi gelap modern (`#0f172a` ke `#1e1b4b`), tipografi Google Fonts "Outfit", serta efek hover kartu interaktif.
- **Konfigurasi Hyperparameter Interaktif**: Sesuaikan parameter model RandomForest langsung dari sidebar:
  - *Test Size* (Ukuran Data Uji): `0.1` - `0.5`
  - *N Estimators* (Jumlah Tree): `10` - `500`
  - *Max Depth* (Kedalaman Maksimum): `1` - `30`
  - *Random State*
- **Evaluasi Akurasi Real-time**: Menampilkan metrik performa model untuk memantau pemenuhan target akurasi:
  - Akurasi Train (target kelayakan $\ge 80\%$)
  - Akurasi Test (target kelayakan $\ge 85\%$)
  - Jumlah total baris dataset ($\ge 5.000$ data)
- **Preview Dataset**: Menampilkan cuplikan 5 baris pertama dari dataset tabular yang digunakan secara dinamis.
- **Prediksi Interaktif Terintegrasi**: Lakukan uji coba prediksi terhadap data baru dengan memasukkan nilai fitur:
  - `G1` (Nilai periode pertama)
  - `G2` (Nilai periode kedua)
  - `studytime` (Waktu belajar mingguan)
  - `absences` (Jumlah ketidakhadiran)
- **Dataset Generator Otomatis**: Menghasilkan dataset sintetis secara otomatis (6.000 sampel) jika file CSV asli tidak ditemukan di folder `dataset`.

---

## 📂 Struktur Direktori

```text
├── dataset/
│   └── student-mat.csv          # File dataset tabular (sintetis/aktual)
├── model/
│   └── model_regresi.pkl        # File model RandomForest yang berhasil disimpan
├── app.py                       # Source code utama dasbor interaktif Streamlit
├── train_model.py               # Script Python untuk pelatihan model secara offline
├── requirements.txt             # Daftar dependensi library Python yang dibutuhkan
├── student.txt                  # Dokumentasi informasi fitur & dataset Student Performance
├── student-merge.R              # Script R pendukung untuk penggabungan dataset asli
└── README.md                    # Dokumentasi penjelasan proyek ini
```

---

## 📊 Detail Dataset & Model

### 1. Fitur dan Target Klasifikasi
Model dilatih menggunakan fitur akademis siswa untuk memprediksi status klasifikasi (misal: *churn* atau status kelulusan):
- **Fitur Input**:
  - `G1` (Nilai Akademik 1)
  - `G2` (Nilai Akademik 2)
  - `studytime` (Durasi belajar mingguan)
  - `absences` (Jumlah absensi kelas)
- **Target Output (`G3`)**:
  - `0` (Negatif / Status Normal)
  - `1` (Positif / Status Churn)

### 2. Algoritma Pembelajaran Mesin
Proyek ini mengimplementasikan algoritma **RandomForestClassifier** dari library `scikit-learn` untuk menangani klasifikasi data dengan struktur decision tree ensemble guna meminimalkan overfitting dan meningkatkan akurasi.

---

## 🛠️ Panduan Instalasi dan Menjalankan

Ikuti langkah-langkah di bawah ini untuk menjalankan proyek di perangkat lokal Anda:

### 1. Kloning Repositori
```bash
git clone https://github.com/oktavia18/Praktikum_DataMining.git
cd Praktikum_DataMining
```

### 2. Instalasi Dependensi
Pastikan Anda telah menginstal Python (versi 3.8+ direkomendasikan). Install seluruh pustaka yang diperlukan melalui `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Melatih Model Offline (Opsional)
Anda dapat melatih model secara terpisah di terminal untuk melihat evaluasi akurasi awal:
```bash
python train_model.py
```

### 4. Menjalankan Dasbor Streamlit
Jalankan aplikasi web interaktif menggunakan perintah berikut:
```bash
streamlit run app.py
```
Aplikasi secara otomatis akan terbuka pada peramban default Anda di alamat `http://localhost:8501`.

---

## 🎨 Cuplikan Visual Aplikasi
Aplikasi dilengkapi dengan desain web modern:
- **Header Card**: Memuat judul dasbor dengan efek gradien warna *cyan* dan *purple* bersinar.
- **Metric Cards**: Panel ringkasan dengan indikator warna hijau/merah dinamis untuk validasi target akurasi.
- **Predictor Box**: Formulir ramah pengguna dengan tampilan kartu probabilitas prediksi dinamis.
