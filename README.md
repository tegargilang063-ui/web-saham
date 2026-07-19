# IDX Toolkit — SUPERKETAT Scanner + Analisa Valuasi Saham

Web app gabungan dari 2 notebook:
- `Superketat.ipynb` → tab **SUPERKETAT Scanner**
- `Analisa_Valuasi_Saham.ipynb` → tab **Analisa Valuasi**

Dibangun dengan Streamlit. Semua logika perhitungan (indikator, scoring,
persentil historis, composite score) sama persis dengan notebook asli —
hanya dibungkus jadi web interaktif.

## Isi folder
```
app.py            # UI utama Streamlit (2 tab)
scanner.py         # logika SUPERKETAT scanner
valuasi.py          # logika analisa valuasi persentil historis
stocks_list.py      # daftar ±800 saham IDX (universe default)
requirements.txt
```

## Cara deploy GRATIS — Streamlit Community Cloud (paling praktis)

1. **Buat repo GitHub baru** (public, gratis) dan upload semua file di folder ini
   (`app.py`, `scanner.py`, `valuasi.py`, `stocks_list.py`, `requirements.txt`).
   Contoh: buat repo `idx-toolkit`, upload lewat web GitHub (drag & drop) atau `git push`.

2. Buka **https://share.streamlit.io** → login pakai akun GitHub.

3. Klik **"New app"** → pilih repo `idx-toolkit` → branch `main` → main file path `app.py`.

4. Klik **Deploy**. Tunggu 1-2 menit, app langsung dapat URL publik
   (misal `https://idx-toolkit.streamlit.app`) — gratis selamanya, tidak perlu kartu kredit.

Setiap kali kamu push perubahan ke GitHub, app di Streamlit Cloud otomatis update.

## Alternatif gratis lain
- **Hugging Face Spaces** (pilih SDK "Streamlit") — sama gratis, cocok kalau app jarang
  dibuka (auto-sleep saat idle, bangun lagi otomatis saat diakses).
- **Render.com free tier** — bisa juga, tapi setup lebih manual dan free tier auto-sleep
  setelah 15 menit idle.

Streamlit Community Cloud paling direkomendasikan karena minim setup dan langsung baca dari GitHub.

## Catatan penting soal performa
- Scan **semua ±800 saham** butuh waktu (real, bukan bug) — bisa 5-20 menit tergantung
  jumlah thread & rate-limit Yahoo Finance. Untuk pemakaian sehari-hari, gunakan mode
  **Custom / subset** dan isi kode saham yang biasa kamu pantau saja — jauh lebih cepat.
- Yahoo Finance kadang membatasi request beruntun (rate-limit). Kalau banyak saham gagal,
  kurangi jumlah thread paralel atau coba lagi beberapa saat kemudian.
- Data Foreign Net Flow / Haka-Haki tetap tidak tersedia gratis (sama seperti notebook asli) —
  itu perlu Stockbit/RTI Business.
