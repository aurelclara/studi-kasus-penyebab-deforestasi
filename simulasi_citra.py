"""
Simulasi petak citra satelit (Sentinel-2 optik dan Sentinel-1 radar) sebelum dan
sesudah gangguan hutan, lengkap dengan awan, sungai, jalan, dan titik api.

Ini BUKAN citra sungguhan. Tujuannya meniru logika yang dijelaskan pemateri:
  - optik memberi informasi kehijauan (NDVI) tetapi terhalang awan,
  - radar menembus awan tetapi lebih sulit dibaca (bintik acak / speckle),
  - ukuran, bentuk, dan konteks sekitar membantu membedakan penyebab gangguan.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

UKURAN = 32  # 32 x 32 piksel; 1 piksel = 10 m, jadi 1 petak = 320 m x 320 m
KELAS = ["pertanian_kecil", "pertanian_besar", "pertambangan",
         "penebangan", "kebakaran", "gangguan_alami"]
LABEL = {
    "pertanian_kecil": "Pertanian skala kecil",
    "pertanian_besar": "Pertanian skala besar",
    "pertambangan": "Pertambangan",
    "penebangan": "Penebangan selektif",
    "kebakaran": "Kebakaran hutan",
    "gangguan_alami": "Gangguan alami (banjir)",
}

# NDVI sesudah gangguan, perubahan VV (dB), perubahan VH (dB) untuk tiap kelas
PARAM = {
    "pertanian_kecil": (0.38, -2.8, -2.4),
    "pertanian_besar": (0.30, -3.4, -3.0),
    "pertambangan": (0.07, -4.2, -4.4),
    "penebangan": (0.58, -2.4, -2.0),
    "kebakaran": (0.18, -3.2, -3.8),
    "gangguan_alami": (0.45, -3.6, -3.9),
}
SPECKLE = 2.0  # simpangan baku bintik acak radar (dB); radar jauh lebih "berisik" daripada optik

YY, XX = np.mgrid[0:UKURAN, 0:UKURAN]
BENTUK = (UKURAN, UKURAN)


# ------------------------------------------------------------ alat bantu geometri
def kabur(a, n=3):
    """Rata-rata bergerak n x n (dipakai untuk meredam speckle dan membuat tekstur)."""
    return ndi.uniform_filter(a, size=n, mode="reflect")


def garis_sungai(rng):
    x = np.linspace(0, UKURAN - 1, 80)
    y = rng.uniform(6, 26) + rng.uniform(2, 6) * np.sin(rng.uniform(0.15, 0.3) * x + rng.uniform(0, 6.28))
    titik = np.column_stack([x, y])
    return titik[:, ::-1] if rng.random() < 0.5 else titik  # kolom: (x, y)


def garis_jalan(rng):
    pusat = rng.uniform(8, 24, 2)
    sudut = rng.uniform(0, np.pi)
    t = np.linspace(-30, 30, 120)
    titik = pusat + np.column_stack([np.cos(sudut) * t, np.sin(sudut) * t])
    dalam = (titik >= 0).all(1) & (titik <= UKURAN - 1).all(1)
    return titik[dalam]


def peta_jarak(garis):
    """Jarak setiap piksel ke garis (sungai/jalan) terdekat."""
    return np.hypot(XX[..., None] - garis[:, 0], YY[..., None] - garis[:, 1]).min(axis=2)


def jarak_ke_garis(titik, garis):
    if garis is None or len(garis) == 0:
        return 40.0
    return float(np.hypot(garis[:, 0] - titik[0], garis[:, 1] - titik[1]).min())


def persegi(pusat, tinggi, lebar):
    return (np.abs(XX - pusat[0]) <= lebar / 2) & (np.abs(YY - pusat[1]) <= tinggi / 2)


def blob(pusat, luas, kekasaran, rng, panjang=1.0, sudut=0.0):
    """Area tidak beraturan seluas kira-kira `luas` piksel di sekitar `pusat`."""
    dx, dy = XX - pusat[0], YY - pusat[1]
    u = (dx * np.cos(sudut) + dy * np.sin(sudut)) / panjang
    v = -dx * np.sin(sudut) + dy * np.cos(sudut)
    derau = kabur(rng.normal(size=BENTUK), 5)
    derau = (derau - derau.mean()) / (derau.std() + 1e-9)
    skor = -np.hypot(u, v) + kekasaran * derau
    return skor >= np.sort(skor.ravel())[-int(luas)]


# ------------------------------------------------------------ pembuat adegan
def buat_adegan(kelas, rng):
    sungai = garis_sungai(rng) if (kelas == "gangguan_alami" or rng.random() < 0.7) else None
    jalan = garis_jalan(rng) if (kelas == "penebangan" or rng.random() < 0.6) else None
    acak = rng.uniform(9, 23, 2)

    if kelas == "pertanian_kecil":
        pusat = acak if jalan is None else np.clip(jalan[rng.integers(len(jalan))] + rng.normal(0, 4, 2), 4, 27)
        mask = persegi(pusat, rng.uniform(4, 9), rng.uniform(4, 9))
    elif kelas == "pertanian_besar":
        mask = persegi(rng.uniform(12, 20, 2), rng.uniform(14, 24), rng.uniform(14, 24))
    elif kelas == "pertambangan":
        mask = blob(acak, rng.integers(180, 450), 0.8, rng)
    elif kelas == "penebangan":
        titik = jalan[rng.integers(len(jalan))]
        dekat = np.hypot(XX - titik[0], YY - titik[1]) <= rng.uniform(8, 12)
        mask = (peta_jarak(jalan) <= rng.uniform(0.8, 1.5)) & dekat
        for _ in range(rng.integers(3, 7)):
            mask |= persegi(titik + rng.normal(0, 5, 2), rng.integers(2, 4), rng.integers(2, 4))
    elif kelas == "kebakaran":
        pusat = acak
        if sungai is not None and rng.random() < 0.6:  # kebakaran di tepi sungai bisa mirip banjir
            pusat = sungai[rng.integers(10, len(sungai) - 10)]
        mask = blob(pusat, rng.integers(120, 340), 0.6, rng, panjang=rng.uniform(1, 2.5), sudut=rng.uniform(0, np.pi))
    else:  # gangguan_alami: pita di sepanjang sungai
        titik = sungai[rng.integers(10, len(sungai) - 10)]
        dekat = np.hypot(XX - titik[0], YY - titik[1]) <= rng.uniform(9, 14)
        mask = (peta_jarak(sungai) <= rng.uniform(2.0, 3.5)) & dekat

    ndvi_ssd, dvv, dvh = PARAM[kelas]
    g_ndvi, g_sar = rng.normal(0, 0.07), rng.normal(0, 1.2)  # variasi antar adegan
    hutan_ndvi = lambda: rng.normal(0.82, 0.03, BENTUK)

    ndvi_sebelum = hutan_ndvi()
    ndvi_sesudah = np.where(mask, ndvi_ssd + g_ndvi + rng.normal(0, 0.05, BENTUK), hutan_ndvi())
    vv_sebelum = rng.normal(-7.0, SPECKLE, BENTUK)
    vh_sebelum = rng.normal(-13.0, SPECKLE, BENTUK)
    vv_sesudah = np.where(mask, -7.0 + dvv + g_sar + rng.normal(0, SPECKLE, BENTUK), rng.normal(-7.0, SPECKLE, BENTUK))
    vh_sesudah = np.where(mask, -13.0 + dvh + g_sar + rng.normal(0, SPECKLE, BENTUK), rng.normal(-13.0, SPECKLE, BENTUK))
    peluang_api = 0.55 if kelas == "kebakaran" else 0.08  # sensor titik api tidak selalu menangkap

    return {
        "kelas": kelas, "mask_benar": mask, "sungai": sungai, "jalan": jalan,
        "ndvi_sebelum": ndvi_sebelum, "ndvi_sesudah": ndvi_sesudah,
        "vv_sebelum": vv_sebelum, "vv_sesudah": vv_sesudah,
        "vh_sebelum": vh_sebelum, "vh_sesudah": vh_sesudah,
        "hotspot": int(rng.random() < peluang_api),
        "awan": np.zeros(BENTUK, dtype=bool),
    }


def tambah_awan(adegan, fraksi, rng):
    """Tutupi citra optik 'sesudah' dengan awan; citra radar tidak terpengaruh."""
    derau = kabur(rng.normal(size=BENTUK), 6)
    awan = derau >= np.quantile(derau, 1 - fraksi)
    baru = dict(adegan)
    baru["awan"] = awan
    baru["ndvi_sesudah"] = np.where(awan, 0.10 + rng.normal(0, 0.04, BENTUK), adegan["ndvi_sesudah"])
    return baru


# ------------------------------------------------------------ ekstraksi ciri
def mask_optik(a):
    return ndi.binary_opening(kabur(a["ndvi_sesudah"] - a["ndvi_sebelum"], 3) < -0.25)


def mask_radar(a):
    return ndi.binary_opening(kabur(a["vv_sesudah"] - a["vv_sebelum"], 5) < -1.6)


def ciri_bentuk(mask, a, awalan):
    """Luas, kekompakan, kememanjangan, jumlah komponen, dan jarak konteks."""
    kosong = {f"{awalan}_luas": 0, f"{awalan}_kompak": 0.0, f"{awalan}_memanjang": 1.0,
              f"{awalan}_komponen": 0, f"{awalan}_jarak_sungai": 40.0, f"{awalan}_jarak_jalan": 40.0}
    luas = int(mask.sum())
    if luas < 3:
        return kosong
    keliling = int((mask & ~ndi.binary_erosion(mask)).sum())
    koordinat = np.column_stack(np.where(mask)).astype(float)
    nilai_eigen = np.sort(np.linalg.eigvalsh(np.cov(koordinat.T) + 1e-6 * np.eye(2)))
    pusat = koordinat.mean(axis=0)[::-1]  # (x, y)
    return {
        f"{awalan}_luas": luas,
        f"{awalan}_kompak": 4 * np.pi * luas / (keliling ** 2),
        f"{awalan}_memanjang": float(np.sqrt(nilai_eigen[1] / nilai_eigen[0])),
        f"{awalan}_komponen": int(ndi.label(mask)[1]),
        f"{awalan}_jarak_sungai": jarak_ke_garis(pusat, a["sungai"]),
        f"{awalan}_jarak_jalan": jarak_ke_garis(pusat, a["jalan"]),
    }


def ekstrak_ciri(a):
    mo, mr = mask_optik(a), mask_radar(a)
    d_ndvi = a["ndvi_sesudah"] - a["ndvi_sebelum"]
    d_vv = a["vv_sesudah"] - a["vv_sebelum"]
    d_vh = a["vh_sesudah"] - a["vh_sebelum"]
    ciri = {}
    ciri.update(ciri_bentuk(mo, a, "o"))
    ciri["o_d_ndvi"] = float(d_ndvi[mo].mean()) if mo.any() else 0.0
    ciri["o_d_ndvi_std"] = float(d_ndvi[mo].std()) if mo.any() else 0.0
    ciri.update(ciri_bentuk(mr, a, "r"))
    ciri["r_d_vv"] = float(d_vv[mr].mean()) if mr.any() else 0.0
    ciri["r_d_vh"] = float(d_vh[mr].mean()) if mr.any() else 0.0
    ciri["r_d_vv_std"] = float(d_vv[mr].std()) if mr.any() else 0.0
    ciri["titik_api"] = a["hotspot"]
    return ciri


FITUR_OPTIK = ["o_luas", "o_kompak", "o_memanjang", "o_komponen", "o_jarak_sungai",
               "o_jarak_jalan", "o_d_ndvi", "o_d_ndvi_std"]
FITUR_RADAR = ["r_luas", "r_kompak", "r_memanjang", "r_komponen", "r_jarak_sungai",
               "r_jarak_jalan", "r_d_vv", "r_d_vh", "r_d_vv_std"]
FITUR_TAMBAHAN = ["titik_api"]


def buat_dataset(per_kelas=300, seed=7, peluang_awan=0.4):
    rng = np.random.default_rng(seed)
    baris = []
    for kelas in KELAS:
        for _ in range(per_kelas):
            a = buat_adegan(kelas, rng)
            berawan = rng.random() < peluang_awan
            if berawan:
                a = tambah_awan(a, rng.uniform(0.3, 0.9), rng)
            baris.append({**ekstrak_ciri(a), "kelas": kelas, "berawan": int(berawan)})
    return pd.DataFrame(baris)
