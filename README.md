# REKLE 
Notebook ini membangun ulang eksperimen klasifikasi sampah menggunakan **MobileNetV2** vs **ResNet50** dengan lima dataset sumber asli dari Kaggle. Proyek ini menggabungkan lima sumber data yang berbeda, membersihkan data, mengatasi duplikasi, dan melatih dua arsitektur model untuk performa optimal dalam klasifikasi sampah.

## 📋 Daftar Isi

- [Ringkasan Eksperimen](#ringkasan-eksperimen)
- [Dataset Sumber](#dataset-sumber)
- [Struktur Notebook](#struktur-notebook)
- [Konfigurasi Eksperimen](#konfigurasi-eksperimen)
- [Persyaratan & Instalasi](#persyaratan--instalasi)
- [Cara Menggunakan](#cara-menggunakan)
- [Label Kelas](#label-kelas)
- [Hasil & Output](#hasil--output)
- [Catatan Penting](#catatan-penting)

## 🎯 Ringkasan Eksperimen

Proyek **REKLE** adalah rekonstruksi penuh dari eksperimen klasifikasi sampah dengan fitur-fitur utama:

- **Alur:** unduh sumber → inventaris dan pemetaan label → audit kualitas/hash/konflik → group-aware split → training dua tahap → evaluasi statistik dan efisiensi → robustness dan ekspor artefak
- **Jumlah Data:** ~88,309 file gambar dari lima sumber
- **Arsitektur:** MobileNetV2 (mobile-first) vs ResNet50 (akurasi tinggi)
- **Training:** Dua tahap — Stage 1 (training penuh) + Stage 2 (fine-tuning)
- **Validasi:** Statistik bootstrap, confidence intervals, dan multiple testing corrections

**Catatan Penting:** Jumlah 19.185 citra dari eksperimen terdahulu bukan target yang dipaksakan. Jumlah aktual dihitung kembali dari file yang berhasil diunduh, valid, dan lolos kurasi.

## 📦 Dataset Sumber

Lima dataset Kaggle yang diintegrasikan:

| No. | Dataset | File | URL |
|-----|---------|------|-----|
| 1 | Sampah Daur Ulang | 7,014 | [Link](https://www.kaggle.com/datasets/fathurrahmanalfarizy/sampah-daur-ulang) |
| 2 | Medical Waste Dataset Annotated | 5,667 | [Link](https://www.kaggle.com/datasets/dsaivarun/medical-waste-dataset-annotated) |
| 3 | Garbage Classification | 15,515 | [Link](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) |
| 4 | Dataset Sampah Organik dan Anorganik | 50,154 | [Link](https://www.kaggle.com/datasets/eldadvikorian/dataset-sampah-organik-dan-anorganik) |
| 5 | Random Images | 9,959 | [Link](https://www.kaggle.com/datasets/ezzzio/random-images) |
| | **Total** | **~88,309** | |

**Proses:**
- Setiap dataset diunduh dari Kaggle secara otomatis
- File inventaris dengan pemetaan label dibuat
- Deteksi duplikasi menggunakan dhash (threshold: 4)
- Quality check: validitas gambar, korupsi, metadata

## 📖 Struktur Notebook

### Bagian A: Persiapan Eksperimen
- **CELL 1:** Instalasi dependensi (scikit-learn, statsmodels, seaborn, scipy)
- **CELL 2:** Import library, konfigurasi eksperimen, random seed
- **CELL 3:** Informasi hardware dan software environment
- **CELL 4:** Mount Google Drive dan autentikasi Kaggle
- **CELL 5:** Download dan inventaris lima dataset

### Bagian B: Dataset, Pembersihan, EDA, dan Pencegahan Data Leakage
- Pembersihan gambar rusak/corrupt
- Pemetaan label antar dataset (standardisasi)
- Exploratory Data Analysis (EDA)
- Deteksi dan penghapusan duplikasi
- Group-aware stratified split (train/val/test)
- Prevent data leakage dengan group-based splitting

### Bagian C: Model Architecture & Training Pipeline
- Dua arsitektur: MobileNetV2 + ResNet50
- Transfer learning dengan pre-trained weights
- Custom head dengan dropout dan regularization
- Stage 1: Full training dengan learning rate tinggi
- Stage 2: Fine-tuning dengan learning rate rendah

### Bagian D: Evaluation & Statistical Validation
- Confusion matrix per model
- Classification report (precision, recall, F1)
- Balanced accuracy dan macro/micro metrics
- Bootstrap confidence intervals
- Multiple testing corrections (Bonferroni/Benjamini-Hochberg)
- Robustness testing

### Bagian E: Model Export & Deployment
- TensorFlow SavedModel format
- TensorFlow Lite (TFLite) conversion
- Model compression dan optimization
- Deployment requirement validation

## ⚙️ Konfigurasi Eksperimen

```python
CFG = {
    "seed": 42,                          # Random seed untuk reproducibility
    "img_size": (224, 224),              # Ukuran gambar input
    "batch_size": 32,                    # Batch size training
    "dropout": 0.30,                     # Dropout rate
    
    # Stage 1 (Full Training)
    "stage1_lr": 1e-3,                   # Learning rate stage 1
    "stage1_max_epochs": 15,             # Max epochs stage 1
    "stage1_patience": 3,                # Early stopping patience
    
    # Stage 2 (Fine-tuning)
    "stage2_lr": 1e-5,                   # Learning rate stage 2
    "stage2_max_epochs": 30,             # Max epochs stage 2
    "stage2_patience": 5,                # Early stopping patience
    
    # Learning rate reduction
    "reduce_lr_patience": 3,             # ReduceLROnPlateau patience
    "reduce_lr_factor": 0.5,             # ReduceLROnPlateau factor
    
    # Fine-tuning
    "fine_tune_fraction": 0.20,          # Fraksi layer yang di-fine-tune
    
    # Data quality
    "dhash_threshold": 4,                # Threshold dhash untuk duplikasi
    "split_ratio": (0.70, 0.20, 0.10),  # Train/Val/Test split
    
    # Statistical testing
    "alpha": 0.05,                       # Significance level
    "practical_margin_pp": 1.0,          # Practical margin (%)
    "bootstrap_iterations": 2000,        # Bootstrap iterations
    
    # Deployment requirements
    "deployment_requirements": {
        "max_tflite_size_mb": 50.0,      # Max TFLite model size
        "max_latency_ms": 100.0,         # Max inference latency
        "min_macro_f1": 0.90,            # Min macro F1 score
        "min_class_recall": 0.85         # Min recall per class
    }
}
```

## 🔧 Persyaratan & Instalasi

### Prerequisites

- Python 3.10+
- TensorFlow 2.10+
- Akun Kaggle dengan API key
- Google Drive (untuk menyimpan raw dataset)
- Google Colab (rekomendasi) atau environment lokal dengan GPU

### Instalasi Dependensi

```bash
pip install tensorflow>=2.10.0
pip install scikit-learn statsmodels seaborn scipy
pip install pandas numpy matplotlib pillow
pip install kaggle
```

### Setup Kaggle Authentication

1. Buka [Kaggle Settings](https://www.kaggle.com/account)
2. Scroll ke "API" → "Create New API Token"
3. File `kaggle.json` akan terdownload
4. Di **Google Colab Secrets** (🔑 icon):
   - Tambah `KAGGLE_USERNAME` = username Kaggle Anda
   - Tambah `KAGGLE_KEY` = API key Anda
   - Aktifkan "Notebook access" untuk keduanya

## 🚀 Cara Menggunakan

### Opsi 1: Google Colab (Rekomendasi)

1. Buka notebook di Google Colab
2. Pastikan GPU runtime aktif (Runtime → Change Runtime Type → GPU)
3. Jalankan sel-sel secara berurutan dari atas ke bawah
4. Notebook akan otomatis:
   - Mount Google Drive
   - Autentikasi dengan Kaggle
   - Download lima dataset
   - Membersihkan dan memproses data
   - Train dua model
   - Generate evaluasi dan statistik

### Opsi 2: Environment Lokal

```bash
# Clone/download notebook ke folder lokal
jupyter notebook Revisi_JAIC_final.ipynb

# Jalankan sel-sel secara bertahap
# Modifikasi path dataset sesuai kebutuhan lokal
```

**Catatan:** Untuk environment lokal, ubah path di CELL 4:
```python
DRIVE_DIR = "/path/to/your/local/data"  # Ganti sesuai path Anda
RAW_DIR = os.path.join(DRIVE_DIR, "raw")
DATA_DIR = "/path/to/dataset_rekle_mapped"
SPLIT_DIR = "/path/to/rekle_split"
```

## 🏷️ Label Kelas

Dataset distandarisasi ke label utama berikut:

| Label | Deskripsi | Sumber Dataset |
|-------|-----------|-----------------|
| Plastik | Kemasan plastik, botol, tas | Semua |
| Kertas | Kardus, kertas, tissue | Garbage Classification, Sampah Daur Ulang |
| Logam | Kaleng, metal, logam | Garbage Classification, Sampah Daur Ulang |
| Kaca | Botol kaca, pecahan kaca | Garbage Classification, Sampah Daur Ulang |
| Kardus | Kotak kardus, packaging | Garbage Classification, Sampah Daur Ulang |
| Organik | Sampah makanan, organik | Dataset Sampah Organik dan Anorganik, Garbage Classification |
| Medical | APD medis, masker, sarung tangan | Medical Waste Dataset Annotated |
| Residu | Sampah lainnya, tidak terklasifikasi | Sampah Daur Ulang, Random Images |

**Pemetaan Label:**
- Label ambigu dari dataset sumber direvisi manual
- Konsistensi pemetaan dijaga dalam inventory file (`label_mapping.csv`)
- Label yang tidak terpetakan ditandai untuk review ulang

## 📊 Hasil & Output

Notebook menghasilkan artifact berikut:

### 1. Data Processing Artifacts
- `dataset_inventory.csv` — Inventaris lengkap semua gambar
- `label_mapping.csv` — Pemetaan label antar dataset
- `duplicate_analysis.json` — Laporan duplikasi dan dhash
- `quality_audit.json` — Audit kualitas gambar
- `rekle_split/` — Train/val/test split dalam folder terstruktur

### 2. Training Artifacts
- `mobilenet_v2_stage1.h5` — MobileNetV2 setelah stage 1
- `mobilenet_v2_stage2.h5` — MobileNetV2 setelah fine-tuning
- `resnet50_stage1.h5` — ResNet50 setelah stage 1
- `resnet50_stage2.h5` — ResNet50 setelah fine-tuning
- `training_history.json` — Training curves (loss, accuracy per epoch)

### 3. Evaluation Artifacts
- `mobilenet_metrics.json` — Metrics MobileNetV2
- `resnet_metrics.json` — Metrics ResNet50
- `confusion_matrices/` — Confusion matrix per model
- `statistical_validation.json` — Bootstrap CI, hypothesis tests
- `model_comparison.json` — Perbandingan kedua model

### 4. Export untuk Deployment
- `mobilenet_v2_final.pb` — TensorFlow SavedModel (MobileNetV2)
- `resnet50_final.pb` — TensorFlow SavedModel (ResNet50)
- `mobilenet_v2_final.tflite` — TensorFlow Lite (MobileNetV2)
- `resnet50_final.tflite` — TensorFlow Lite (ResNet50)

### 5. Hardware Environment
- `hardware_environment.json` — Versi TF, Python, spesifikasi GPU/CPU

## ⚠️ Catatan Penting

1. **Reproducibility:** Random seed (42) diatur di CELL 2. Ganti jika diperlukan eksperimen dengan seed berbeda.

2. **Deterministic Operations:** Notebook mencoba mengaktifkan `tf.config.experimental.enable_op_determinism()`. Pada beberapa versi TF, ini mungkin tidak fully supported.

3. **Data Leakage Prevention:**
   - Split dilakukan dengan `StratifiedGroupKFold`
   - Augmentasi hanya diterapkan pada training set
   - Test set tidak digunakan dalam hyperparameter tuning

4. **Label Ambigu:** Gambar dengan label yang tidak jelas atau ambigu harus ditinjau manual sebelum training dimulai. Daftar file ambigu disimpan dalam `ambiguous_labels.csv`.

5. **GPU Memory:** Batch size 32 memerlukan minimal ~8GB VRAM. Untuk GPU dengan VRAM lebih kecil, kurangi batch size di CFG.

6. **Training Duration:** 
   - Stage 1: ~1-2 jam (15 epochs)
   - Stage 2: ~2-3 jam (30 epochs)
   - Total: ~3-5 jam (dengan GPU)

7. **Validation Strategy:** 
   - Validation set digunakan untuk early stopping dan model selection
   - Test set digunakan hanya untuk final evaluation
   - Jangan ada overlap antara train/val/test

8. **Deployment Checklist:**
   - Verifikasi model memenuhi `deployment_requirements`
   - TFLite model size ≤ 50MB
   - Inference latency ≤ 100ms
   - Macro F1 ≥ 0.90
   - Per-class recall ≥ 0.85

## 📈 Key Metrics

### Performance Targets
- **Accuracy:** Target ≥ 92%
- **Macro F1:** Target ≥ 0.90
- **Balanced Accuracy:** Target ≥ 0.88
- **Per-class Recall:** Target ≥ 0.85

### Efficiency Targets (Deployment)
- **Model Size (TFLite):** ≤ 50MB
- **Inference Latency:** ≤ 100ms (mobile device)
- **Memory Footprint:** ≤ 200MB (at runtime)

## 📝 Lisensi & Attribution

- Dataset diunduh dari **Kaggle** — lihat link masing-masing dataset untuk terms of use
- TensorFlow & Keras — Apache 2.0 License
- Notebook ini — Gunakan sesuai kebutuhan penelitian/pembelajaran

## 🤝 Support & Troubleshooting

### Issue: "Kaggle API key not found"
**Solusi:** Pastikan `kaggle.json` berada di `~/.kaggle/` dengan permission 600, atau gunakan Colab Secrets.

### Issue: "Out of Memory (OOM) during training"
**Solusi:** Kurangi `batch_size` di CFG dari 32 menjadi 16 atau 8.

### Issue: "Some classes have low recall"
**Solusi:** Check data imbalance di CELL 5, pertimbangkan `compute_class_weight()` atau augmentasi class dengan sampel sedikit.

### Issue: "Training tidak converge"
**Solusi:** Sesuaikan learning rate di `stage1_lr` atau `stage2_lr`. Coba nilai 1e-4 atau 5e-4.

## 📞 Contact & Questions

Untuk pertanyaan atau feedback tentang eksperimen ini, silakan buka issue atau diskusi terkait.

