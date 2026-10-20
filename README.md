# Bukan Hanya di Mana, Tapi Kenapa: Mengenali Penyebab Deforestasi dari Citra Satelit

Studi kasus **CRISP-DM** berdasarkan kuliah praktisi *Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI* (Bart Slagter, Wageningen University & Research).

Proyek ini membuat versi mini dari sistem yang dijelaskan pemateri: mengklasifikasikan **penyebab langsung** gangguan hutan (pertanian skala kecil dan besar, pertambangan, penebangan selektif, kebakaran, gangguan alami) dari kombinasi citra **radar (Sentinel-1)** dan **optik (Sentinel-2)**, lalu membandingkan kinerja optik saja, radar saja, dan gabungan, terutama saat langit berawan.

🔗 **[Lihat laporan HTML (live)](https://aurelclara.github.io/studi-kasus-penyebab-deforestasi/)**

> **Penting:** semua data adalah **simulasi** dan modelnya adalah *Random Forest* dengan ciri buatan tangan, bukan CNN pada citra asli. Angka akurasi tidak boleh dibaca sebagai kinerja sistem yang sebenarnya. Tujuannya memahami alur dan logikanya.

## Kerangka CRISP-DM

| Tahap | Penjelasan |
| --- | --- |
| **Business Understanding** | Peringatan deforestasi hanya memberi tahu *di mana*. Penegakan hukum dan kebijakan butuh tahu *kenapa*. |
| **Data Understanding** | Petak simulasi 32x32 piksel (10 m/piksel): NDVI (optik), VV dan VH (radar), lapisan sungai dan jalan, titik api, dan awan. |
| **Data Preparation** | Selisih sebelum-sesudah, penghalusan untuk meredam *speckle*, ambang perubahan, lalu ekstraksi ciri bentuk dan konteks. |
| **Modeling** | *Random Forest* (200 pohon) untuk tiga skenario sensor. |
| **Evaluation** | Akurasi, F1 makro, akurasi cerah vs berawan, matriks kebingungan, dan pola antarwilayah. |
| **Deployment** | Laporan HTML dengan penjelajah petak citra interaktif. |

## Struktur Proyek

```
studi-kasus-penyebab-deforestasi/
├── simulasi_citra.py     # Pembangkit petak citra simulasi + ekstraksi ciri
├── latih_model.py        # Pelatihan dan evaluasi 3 skenario (hasil di terminal)
├── generate_report.py    # Menyusun template + hasil menjadi index.html
├── template.html         # Kerangka laporan
├── warna.py              # Palet pastel + penghitung kontras WCAG
├── index.html            # Laporan (buka di browser)
├── fitur_simulasi.csv    # Tabel ciri hasil simulasi
├── hasil_evaluasi.csv    # Ringkasan kinerja tiap skenario
├── requirements.txt
├── .gitignore
└── README.md
```

## Cara Menjalankan

```bash
git clone https://github.com/USERNAME/studi-kasus-penyebab-deforestasi.git
cd studi-kasus-penyebab-deforestasi

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python latih_model.py            # latih dan lihat hasil di terminal
python generate_report.py        # buat ulang index.html
```

## Laporan HTML

- **Membuka lokal:** double-click `index.html` (data sudah tertanam, tidak perlu internet).
- **Lewat link:** aktifkan GitHub Pages di **Settings → Pages**, pilih branch `main` dan folder `/ (root)`.

## Keterbatasan

- Data simulasi; bentuk gangguan disederhanakan.
- Proporsi penyebab per wilayah diatur mengikuti pola kualitatif di materi, bukan statistik asli.
- Pengembangan lanjutan: ganti ciri manual dengan CNN, atau pakai citra Sentinel asli (mis. lewat Google Earth Engine).

## Lisensi
kuliah praktisi Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI, oleh Bart Slagter, Wageningen University & Research.
