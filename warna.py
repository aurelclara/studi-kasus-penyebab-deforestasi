"""
Palet pastel + warna tiap kelas penyebab deforestasi + rasio kontras WCAG.
Satu sumber warna dipakai oleh CSS laporan dan data yang dikirim ke JavaScript.
"""

PALET = {
    "latar": "#f2f6f1",
    "kertas": "#fffefa",
    "tinta": "#34404a",
    "tinta-lembut": "#5d6b76",
    "garis": "#d5dfd6",
    "hutan": "#cfe5d3",
    "hutan-tua": "#4a7c59",
    "langit": "#d4e6f1",
    "langit-tua": "#3f6f8f",
    "pasir": "#f4e8c8",
    "pasir-tua": "#8a6d2f",
    "persik": "#f7d6c5",
    "persik-tua": "#a4573a",
    "lavender": "#e0d9f2",
    "lavender-tua": "#6a5ea3",
    "mawar": "#f0b3bd",
    "mawar-tua": "#9c4358",
    "mentega": "#f8f0c4",
}

# kelas -> (warna isi, warna tepi)
WARNA_KELAS = {
    "pertanian_kecil": ("mentega", "pasir-tua"),
    "pertanian_besar": ("persik", "persik-tua"),
    "pertambangan": ("lavender", "lavender-tua"),
    "penebangan": ("hutan", "hutan-tua"),
    "kebakaran": ("mawar", "mawar-tua"),
    "gangguan_alami": ("langit", "langit-tua"),
}


def _kanal(c):
    c = c / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminans(hex_warna):
    h = hex_warna.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _kanal(r) + 0.7152 * _kanal(g) + 0.0722 * _kanal(b)


def rasio_kontras(depan, belakang):
    """Rasio kontras WCAG 2.x antara dua warna hex (1 sampai 21)."""
    a, b = luminans(depan), luminans(belakang)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def css_variabel():
    isi = "".join(f"--{n}:{k};" for n, k in PALET.items())
    for kelas, (isi_w, tepi_w) in WARNA_KELAS.items():
        nama = kelas.replace("_", "-")
        isi += f"--k-{nama}:{PALET[isi_w]};--kt-{nama}:{PALET[tepi_w]};"
    return ":root{" + isi + "}"
