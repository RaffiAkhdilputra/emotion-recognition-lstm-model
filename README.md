# Emotion Recognition using LSTM

Aplikasi ini adalah sistem **Emotion Recognition** berbasis **LSTM (Long Short-Term Memory)** yang menganalisis emosi dalam teks menggunakan deep learning. Dokumentasi ini telah diperbarui agar sesuai dengan alur dan fungsi yang terdapat dalam file **`streamlit_app.py`**.

---

## 🚀 Fitur Utama

* Input teks bebas untuk dianalisis.
* Prediksi emosi berdasarkan model LSTM yang telah dilatih.
* Antarmuka interaktif menggunakan Streamlit.
* Contoh cepat (Quick Examples) untuk membantu pengguna memahami kategori emosi.
* Tampilan hasil analisis yang menarik dan intuitif.

---

## 📁 Struktur Proyek

```
emotion-recognition-lstm-model/
│
├── dataset/               # dataset
├── model/                 # Model LSTM tersimpan (.keras, tokenizer, config)
├── deployement.ipynb      # Prototype deployement
├── README.md              # Dokumentasi proyek
└── requirements.txt       # Semua dependency
├── streamlit_app.py       # Aplikasi Streamlit utama
├── train_model.ipynb      # Model training notebook
```

---

## ▶️ Menjalankan Aplikasi

### 1. Clone repository:

```bash
git clone https://github.com/RaffiAkhdilputra/emotion-recognition-lstm-model.git
cd emotion-recognition-lstm-model
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Jalankan Streamlit:

```bash
streamlit run streamlit_app.py
```

---

## 🧠 Cara Kerja Aplikasi (Mengikuti streamlit_app.py)

### 1. **Load Model & Preprocessing**

Aplikasi otomatis memuat:

* model LSTM (`model.keras`)
* tokenizer (`tokenizer.json`)
* label encoder (`label_encoder.json`)

### 2. **Input Teks**

Pengguna memasukkan teks dalam field yang disediakan.

### 3. **Preprocessing**

* Lowercase
* Clean special characters
* Tokenization
* Sequence padding

### 4. **Prediksi Emosi**

Model mengeluarkan output probabilitas → dikonversi menjadi label emosi.

### 5. **Tampilan Hasil**

Menggunakan komponen Streamlit seperti:

* `st.success()`
* `st.error()`
* `st.info()`
* Card-style preview

---

## 📝 Kategori Emosi

Berdasarkan implementasi di `streamlit_app.py`, kategori emosi yang tersedia:

* 😔 Depressed
* 😢 Sad
* 😐 Neutral
* 🙂 Good
* 😄 Happy
* 🤩 Excited

---

## 🧪 Quick Example

Di sidebar terdapat contoh cepat untuk pengguna mencoba berbagai emosi:

```python
examples = {
    "Depressed": "I feel so hopeless today...",
    "Sad": "I miss my friends so much.",
    "Neutral": "I just woke up and ate breakfast.",
    "Good": "Today was pretty nice!",
    "Happy": "I passed my exam!",
    "Excited": "I can’t wait for the concert tonight!"
}
```

---

## 📦 Deployment

Streamlit Cloud: https://emotion-recognition-lstm-model.streamlit.app/

---

## 📄 Lisensi

Proyek ini menggunakan lisensi **UNLICENSED LICENSE**.
