"""
Membuat laporan HTML (index.html) dari template.html.
Seluruh hasil (tabel, matriks, grafik batang, data penjelajah) dihitung ulang
oleh skrip ini, jadi angka di laporan selalu sama dengan hasil model.
"""
import json
from datetime import date

import numpy as np
import pandas as pd

from latih_model import GABUNGAN, SKENARIO, latih_dan_evaluasi, simulasi_wilayah
from simulasi_citra import (
    KELAS, LABEL, buat_adegan, buat_dataset, ekstrak_ciri, mask_optik, mask_radar, tambah_awan,
)
from warna import PALET as P, WARNA_KELAS, css_variabel, rasio_kontras

BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
         "Agustus", "September", "Oktober", "November", "Desember"]
SINGKAT = {"pertanian_kecil": "Tani kecil", "pertanian_besar": "Tani besar", "pertambangan": "Tambang",
           "penebangan": "Tebang", "kebakaran": "Api", "gangguan_alami": "Banjir"}
NAMA_CIRI = {
    "r_luas": "luas area terdeteksi radar", "r_memanjang": "kememanjangan area (radar)",
    "r_kompak": "kekompakan bentuk (radar)", "r_d_vv": "perubahan sinyal VV", "r_d_vh": "perubahan sinyal VH",
    "r_d_vv_std": "keragaman perubahan VV", "r_komponen": "jumlah potongan area (radar)",
    "r_jarak_sungai": "jarak ke sungai (radar)", "r_jarak_jalan": "jarak ke jalan (radar)",
    "o_luas": "luas area terdeteksi optik", "o_kompak": "kekompakan bentuk (optik)",
    "o_memanjang": "kememanjangan area (optik)", "o_d_ndvi": "penurunan NDVI",
    "o_d_ndvi_std": "keragaman penurunan NDVI", "o_komponen": "jumlah potongan area (optik)",
    "o_jarak_sungai": "jarak ke sungai (optik)", "o_jarak_jalan": "jarak ke jalan (optik)",
    "titik_api": "titik api",
}


def pct(v, d=1):
    return f"{v * 100:.{d}f}".replace(".", ",") + "%"


def campur(hex_a, hex_b, t):
    a = [int(hex_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(hex_b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(x + (y - x) * t)):02x}" for x, y in zip(a, b))


# ---------------------------------------------------------------- bagian-bagian HTML
def html_hasil(h):
    r = h["hasil"].set_index("skenario")
    o, ra, g = r.loc["Optik saja (Sentinel-2)"], r.loc["Radar saja (Sentinel-1)"], r.loc[GABUNGAN]
    teks = (f"<p>Model diuji pada {len(h['uji'])} petak yang tidak dipakai saat pelatihan. Model optik saja sangat baik "
            f"saat langit cerah ({pct(o.akurasi_cerah)}), tetapi turun menjadi {pct(o.akurasi_berawan)} saat berawan. "
            f"Radar stabil di kedua kondisi ({pct(ra.akurasi_cerah)} cerah, {pct(ra.akurasi_berawan)} berawan), tetapi "
            f"kalah dari optik saat langit cerah karena citranya lebih berisik. Model gabungan mengambil kelebihan "
            f"keduanya dan mencapai akurasi keseluruhan {pct(g.akurasi)}.</p>")
    baris = ""
    for nama, x in r.iterrows():
        tebal = " style='font-weight:700'" if nama == GABUNGAN else ""
        baris += (f"<tr{tebal}><td>{nama}</td><td>{pct(x.akurasi)}</td><td>{pct(x.f1_makro)}</td>"
                  f"<td>{pct(x.akurasi_cerah)}</td><td>{pct(x.akurasi_berawan)}</td></tr>\n")
    batang = ""
    for nama, x in r.iterrows():
        batang += f"<h3>{nama}</h3>"
        for jenis, kelas_css, nilai in (("Cerah", "cerah", x.akurasi_cerah), ("Berawan", "berawan", x.akurasi_berawan)):
            batang += (f"<div class='bb'><span>{jenis}</span><div class='rel {kelas_css}' role='img' "
                       f"aria-label='{nama}, {jenis.lower()}: {pct(nilai)}'><div style='width:{nilai * 100:.1f}%'></div></div>"
                       f"<span class='nilai'>{pct(nilai, 0)}</span></div>")
    return teks, baris, batang


def html_matriks(h):
    m = h["matriks"]
    total = m.sum(axis=1, keepdims=True)
    kepala = "<tr><th>Sebenarnya \\ Tebakan</th>" + "".join(
        f"<th title='{LABEL[k]}'>{SINGKAT[k]}</th>" for k in KELAS) + "</tr>"
    isi = ""
    for i, k in enumerate(KELAS):
        isi += f"<tr><td><strong>{LABEL[k]}</strong></td>"
        for j in range(len(KELAS)):
            f = m[i, j] / total[i, 0]
            warna = campur(P["kertas"], P["langit"] if i == j else P["mawar"], min(1, f * (1.0 if i == j else 2.2)))
            isi += f"<td style='background:{warna}' title='{m[i, j]} petak ({pct(f, 0)} dari baris)'>{m[i, j]}</td>"
        isi += "</tr>"
    sel = [(m[i, j], KELAS[i], KELAS[j]) for i in range(len(KELAS)) for j in range(len(KELAS)) if i != j]
    sel.sort(reverse=True)
    top = "; ".join(f"{LABEL[a].lower()} dikira {LABEL[b].lower()} ({n} petak)" for n, a, b in sel[:3])
    teks = f"<p>Tiga kekeliruan terbanyak pada model gabungan: {top}.</p>"
    return teks, kepala + isi


def html_penting(h):
    atas = h["penting"].head(4)
    daftar = ", ".join(f"{NAMA_CIRI[k]} ({pct(v, 0)})" for k, v in atas.items())
    return (f"Ciri yang paling berpengaruh pada model gabungan: {daftar}. Ukuran dan bentuk area gangguan ternyata "
            f"lebih menentukan daripada nilai sinyal mentah, sejalan dengan penjelasan pemateri bahwa klasifikasi "
            f"mempertimbangkan ukuran, bentuk, dan konteks sekitar.")


def html_wilayah(w):
    blok, ringkas = "", []
    for wil in w["wilayah"].unique():
        d = w[w["wilayah"] == wil].set_index("kelas")
        blok += f"<h3>{wil}</h3>"
        for judul, kolom in (("Sebenarnya", "sebenarnya"), ("Prediksi", "prediksi")):
            seg = ""
            for k in KELAS:
                v = d.loc[k, kolom]
                if v <= 0:
                    continue
                nama = k.replace("_", "-")
                teks = f"{SINGKAT[k]} {v * 100:.0f}%" if v >= 0.13 else ""
                seg += (f"<span style='width:{v * 100:.1f}%;background:var(--k-{nama})' "
                        f"title='{LABEL[k]}: {pct(v, 0)}'>{teks}</span>")
            blok += f"<div class='stack-baris'><span>{judul}</span><div class='stack'>{seg}</div></div>"
        urut = d["prediksi"].sort_values(ascending=False)
        ringkas.append((wil, urut.index[0], urut.iloc[0], urut.index[1], urut.iloc[1]))
    selisih = (w["prediksi"] - w["sebenarnya"]).abs().mean() * 100
    kal = "; ".join(f"di {wil}, penyebab terbanyak adalah {LABEL[a].lower()} ({pct(va, 0)}) disusul {LABEL[b].lower()} ({pct(vb, 0)})"
                    for wil, a, va, b, vb in ringkas)
    selisih_txt = f"{selisih:.1f}".replace(".", ",")
    teks = (f"Hasilnya: {kal}. Rata-rata selisih antara komposisi prediksi dan komposisi sebenarnya hanya "
            f"{selisih_txt} poin persentase, jadi pola antarwilayah tetap terbaca meskipun "
            "klasifikasi per petak tidak sempurna. Di Asia Tenggara, yang paling relevan bagi kita, komposisinya "
            "paling beragam: pertanian besar dan kecil, penebangan, dan pertambangan semuanya punya porsi nyata.")
    legenda = "".join(f"<span><i style='background:var(--k-{k.replace('_', '-')});border-color:var(--kt-{k.replace('_', '-')})'></i>{LABEL[k]}</span>" for k in KELAS)
    return blok, teks, legenda


# ---------------------------------------------------------------- data penjelajah
def data_penjelajah(model, seed=2024):
    rng = np.random.default_rng(seed)
    nama_model = [("optik", "Optik saja (Sentinel-2)"), ("radar", "Radar saja (Sentinel-1)"), ("gabungan", GABUNGAN)]
    keluar = {}
    for kelas in KELAS:
        a = buat_adegan(kelas, rng)
        keluar[kelas] = {}
        for kondisi, fraksi in (("cerah", 0.0), ("berawan", 0.6)):
            v = tambah_awan(a, fraksi, rng) if fraksi else a
            ciri = pd.DataFrame([ekstrak_ciri(v)])
            pred = {}
            for kunci, nama in nama_model:
                rf = model[nama]
                p = dict(zip(rf.classes_, rf.predict_proba(ciri[SKENARIO[nama]])[0]))
                pred[kunci] = [round(float(p[k]), 3) for k in KELAS]
            biner = lambda m: "".join("1" if x else "0" for x in m.ravel())
            keluar[kelas][kondisi] = {
                "ndvi": [round(float(x), 2) for x in v["ndvi_sesudah"].ravel()],
                "vv": [round(float(x), 1) for x in v["vv_sesudah"].ravel()],
                "m_opt": biner(mask_optik(v)), "m_sar": biner(mask_radar(v)), "awan": biner(v["awan"]),
                "pred": pred,
            }
    return keluar, [{"kunci": k, "nama": n} for k, n in nama_model]


def main():
    df = buat_dataset()
    h = latih_dan_evaluasi(df)
    w = simulasi_wilayah(h["model"])
    demo, daftar_model = data_penjelajah(h["model"])

    teks_hasil, tabel_hasil, batang = html_hasil(h)
    teks_matriks, matriks = html_matriks(h)
    blok_wil, teks_wil, legenda_wil = html_wilayah(w)
    kelas_json = [{"id": k, "label": LABEL[k], "isi": P[WARNA_KELAS[k][0]], "tepi": P[WARNA_KELAS[k][1]]} for k in KELAS]
    kontras_min = min(rasio_kontras(P["tinta"], P[WARNA_KELAS[k][0]]) for k in KELAS)

    html = open("template.html", encoding="utf-8").read()
    hari = date.today()
    pengganti = {
        "/*CSS_WARNA*/": css_variabel(),
        "__TANGGAL__": f"{hari.day} {BULAN[hari.month - 1]} {hari.year}",
        "__HASIL_TEKS__": teks_hasil, "__TABEL_HASIL__": tabel_hasil, "__BATANG__": batang,
        "__MATRIKS_TEKS__": teks_matriks, "__MATRIKS__": matriks, "__PENTING__": html_penting(h),
        "__WILAYAH__": blok_wil, "__LEGENDA_WILAYAH__": legenda_wil, "__WILAYAH_TEKS__": teks_wil,
        "__KELAS_JSON__": json.dumps(kelas_json, ensure_ascii=False),
        "__DATA_JSON__": json.dumps(demo, separators=(",", ":")),
        "__MODEL_JSON__": json.dumps(daftar_model, ensure_ascii=False),
        "__WARNA_MAWAR__": P["mawar"], "__WARNA_LAVENDER_TUA__": P["lavender-tua"], "__WARNA_KERTAS__": P["kertas"],
        "Semua data pada laporan ini adalah simulasi.":
            "Semua data pada laporan ini adalah simulasi. Rasio kontras teks gelap terhadap warna kelas minimal "
            + f"{kontras_min:.1f}".replace(".", ",") + ":1 (WCAG AA menuntut 4,5:1).",
    }
    for kunci, nilai in pengganti.items():
        html = html.replace(kunci, nilai)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    df.to_csv("fitur_simulasi.csv", index=False)
    h["hasil"].to_csv("hasil_evaluasi.csv", index=False)
    print(f"index.html dibuat ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
