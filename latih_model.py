"""
Melatih dan membandingkan tiga skenario klasifikasi penyebab gangguan hutan:
optik saja, radar saja, dan gabungan radar + optik + titik api.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from simulasi_citra import (
    FITUR_OPTIK, FITUR_RADAR, FITUR_TAMBAHAN, KELAS, LABEL,
    buat_adegan, buat_dataset, ekstrak_ciri, tambah_awan,
)

SKENARIO = {
    "Optik saja (Sentinel-2)": FITUR_OPTIK,
    "Radar saja (Sentinel-1)": FITUR_RADAR,
    "Gabungan radar + optik": FITUR_OPTIK + FITUR_RADAR + FITUR_TAMBAHAN,
}
GABUNGAN = "Gabungan radar + optik"

# Proporsi penyebab per wilayah. Angka ini DIATUR untuk meniru pola kualitatif di
# materi (bukan statistik asli): Amerika Selatan didominasi pertanian besar dan
# kebakaran, Afrika Tengah pertanian kecil, Asia Tenggara campuran.
PROPORSI_WILAYAH = {
    "Amerika Selatan": [0.10, 0.40, 0.05, 0.10, 0.30, 0.05],
    "Afrika Tengah":   [0.55, 0.05, 0.07, 0.20, 0.08, 0.05],
    "Asia Tenggara":   [0.25, 0.28, 0.17, 0.18, 0.06, 0.06],
}


def latih_dan_evaluasi(df, seed=11):
    latih, uji = train_test_split(df, test_size=0.3, stratify=df["kelas"], random_state=seed)
    model, baris = {}, []
    for nama, fitur in SKENARIO.items():
        rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        rf.fit(latih[fitur], latih["kelas"])
        prediksi = rf.predict(uji[fitur])
        cerah, berawan = (uji["berawan"] == 0).to_numpy(), (uji["berawan"] == 1).to_numpy()
        model[nama] = rf
        baris.append({
            "skenario": nama,
            "akurasi": accuracy_score(uji["kelas"], prediksi),
            "f1_makro": f1_score(uji["kelas"], prediksi, average="macro"),
            "akurasi_cerah": accuracy_score(uji["kelas"][cerah], prediksi[cerah]),
            "akurasi_berawan": accuracy_score(uji["kelas"][berawan], prediksi[berawan]),
        })
    pred_gab = model[GABUNGAN].predict(uji[SKENARIO[GABUNGAN]])
    matriks = confusion_matrix(uji["kelas"], pred_gab, labels=KELAS)
    penting = pd.Series(model[GABUNGAN].feature_importances_, index=SKENARIO[GABUNGAN]).sort_values(ascending=False)
    return {"model": model, "uji": uji, "hasil": pd.DataFrame(baris), "matriks": matriks, "penting": penting}


def simulasi_wilayah(model, per_wilayah=200, seed=99):
    """Klasifikasikan 'peringatan deforestasi' bulanan di tiga wilayah (simulasi)."""
    rng = np.random.default_rng(seed)
    fitur = SKENARIO[GABUNGAN]
    baris = []
    for wilayah, proporsi in PROPORSI_WILAYAH.items():
        sebenarnya = rng.choice(KELAS, size=per_wilayah, p=proporsi)
        ciri = []
        for kelas in sebenarnya:
            a = buat_adegan(kelas, rng)
            if rng.random() < 0.4:
                a = tambah_awan(a, rng.uniform(0.3, 0.9), rng)
            ciri.append(ekstrak_ciri(a))
        tebakan = model[GABUNGAN].predict(pd.DataFrame(ciri)[fitur])
        for kelas in KELAS:
            baris.append({"wilayah": wilayah, "kelas": kelas,
                          "sebenarnya": float((sebenarnya == kelas).mean()),
                          "prediksi": float((tebakan == kelas).mean())})
    return pd.DataFrame(baris)


def main():
    df = buat_dataset()
    h = latih_dan_evaluasi(df)
    print(f"Data: {len(df)} petak ({int(df['berawan'].mean() * 100)}% berawan)\n")
    print(h["hasil"].round(3).to_string(index=False))
    print("\nMatriks kebingungan (model gabungan, baris = sebenarnya):")
    print(pd.DataFrame(h["matriks"], index=[LABEL[k] for k in KELAS], columns=[k[:6] for k in KELAS]))
    print("\nCiri terpenting:\n", h["penting"].head(6).round(3).to_string())
    df.to_csv("fitur_simulasi.csv", index=False)
    h["hasil"].to_csv("hasil_evaluasi.csv", index=False)


if __name__ == "__main__":
    main()
