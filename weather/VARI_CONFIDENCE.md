# Confidence VARI dari data cuaca

Setiap scan drone mendapat **confidence**: angka 0–1, label (*tinggi*, *sedang*, *rendah*) dan alasannya. Angka ini
menyatakan seberapa jauh VARI scan itu bisa dipercaya sebagai tanda kondisi hara.

VARI = (G − R) / (G + R − B) juga berubah karena hal yang bukan hara:

- **Pemotretan.** Matahari rendah membuat bayangan panjang di tajuk. Awan sebagian membuat terang berganti dari foto ke
  foto. Daun yang masih basah setelah hujan tampak lebih gelap.
- **Air.** Sawit yang kekurangan air bisa pucat seperti kekurangan hara (aturan L5: P30 di bawah 60 mm).

Kode: `l6_vari_confidence.py`. Contoh pada data asli: notebook `04_L6_vari_confidence.ipynb`.
Status: **prototipe.** Semua faktor masih tebakan, kecuali aturan L5 dan ambang hujan lebat 25 mm yang sudah
diputuskan.

## Proses: setiap scan, setelah HP sinkron

1. **HP mengunggah manifest scan**: id scan, id kebun, waktu terbang mulai dan selesai (UTC), dan bila ada ringkasan
   VARI. Cara mengunggahnya belum dibangun (keputusan tim Swift/infra).
2. **Backend mencari lokasi kebun** dari daftar kebun (`plots.json`).
3. **Backend mengambil cuaca untuk waktu dan tempat itu:**
   - **Matahari:** elevasi di awal, tengah dan akhir terbang (`solar.py`); yang dipakai elevasi terendah.
   - **Awan saat terbang:** rata-rata tutupan awan dari jam-jam terbang.
   - **Hujan sebelum terbang:** 6 jam dan 24 jam sebelum lepas landas. Dijumlah hanya jika setiap jamnya ada;
     satu jam kosong membuatnya "tidak diketahui", tidak pernah 0 mm.
   - **P30 pada tanggal scan:** aturan L5 yang sudah diputuskan, tanpa perubahan. WeatherKit dulu, NASA POWER yang
     lolos cek bulanan sebagai cadangan.
4. **Setiap input menjadi faktor 0–1** (tabel di bawah). Input yang tidak diketahui tidak pernah mendapat 1.
5. **Digabung:**
   - *pengukuran* = matahari × awan × daun basah
   - *air* = faktor P30
   - **confidence = pengukuran × air**, lalu label dan alasan per faktor yang di bawah 1.
6. **Disimpan di backend:** `scans/<plot_key>/<scan_id>.confidence.json`. Cara hasilnya kembali ke HP belum diputuskan.
7. **Hitung ulang** jika `final` masih `false` (ada input yang belum tersedia), misalnya ketika NASA POWER sudah
   menyusul.
8. **Kalibrasi (L9):** semua hasil disimpan sebagai log, supaya angka tebakan bisa diganti angka terukur.

## Sumber data dan cadangannya

Satu sumber per input; sumber berikutnya hanya dipakai jika sebelumnya tidak ada.

| Input | 1 | 2 | 3 |
|---|---|---|---|
| Awan | WeatherKit per jam | WeatherKit harian, bagian siang (07.00–19.00 WIB) | NASA POWER: ALLSKY ÷ langit cerah FAO-56, sehari penuh |
| Hujan sebelum terbang | WeatherKit per jam (6 dan 24 jam) | WeatherKit harian: siang + malam sebelumnya (24 jam saja) | — |
| P30 | WeatherKit harian (area ~11 km) | NASA POWER yang lolos cek | — |
| Matahari | `solar.py` (dihitung, tidak butuh jaringan) | — | — |

WeatherKit memberi riwayat per jam sampai 240 jam per permintaan. Satu scan butuh 1 permintaan per jam, 5–8 permintaan
riwayat harian untuk P30 (dan cek bulanannya), dan 1 permintaan NASA POWER.

## Faktor

| Faktor | Kondisi | Nilai | Asal |
|---|---|---|---|
| Matahari | elevasi terendah ≥ 45° · 30–45° · 15–30° · < 15° · tidak diketahui | 1,0 · 0,85 · 0,6 · 0,3 · 0,6 | tebakan |
| Awan | cerah (≤ 0,2) · sebagian · mendung (≥ 0,8) · tidak diketahui | 1,0 · 0,7 · 0,85 · 0,8 | tebakan |
| Awan dari NASA | kecerahan ≥ 0,7 cerah · ≤ 0,35 mendung · di antara sebagian | seperti di atas | tebakan |
| Daun basah | ≥ 0,5 mm dalam 6 jam · > 25 mm dalam 24 jam · kering · tidak diketahui | 0,7 · 0,85 · 1,0 · 0,85 | tebakan; **25 mm sudah diputuskan** (ambang hujan lebat L8) |
| Air (L5) | tidak stres, WeatherKit · tidak stres, cadangan NASA · `water_stress` · tidak diketahui | 1,0 · 0,9 · 0,6 · 0,75 | aturan P30 **sudah diputuskan**; nilai faktornya tebakan |
| Label | ≥ 0,8 · ≥ 0,5 · di bawahnya | tinggi · sedang · rendah | tebakan |

Semua angka ini adalah konstanta bernama di `l6_vari_confidence.py`, ditandai `# GUESS – no source yet`.

## Manifest scan (dari HP)

```json
{
 "scan_id": "scan-0001",
 "plot_id": "plot-alpha",
 "flight_start": "2023-06-01T02:40:00Z",
 "flight_end": "2023-06-01T03:05:00Z",
 "vari": {"crowns": 42, "median": 0.118, "iqr": 0.041}
}
```

Waktu terbang harus dalam UTC. Foto yang sudah melewati Roboflow atau WebODM kehilangan EXIF, jadi waktu harus datang
dari aplikasi atau dari foto DJI aslinya. `vari` boleh tidak ada; backend hanya meneruskannya.

## Hasil (di backend)

Hasil asli untuk manifest di atas di kebun uji, yaitu pagi hujan 1 Jun 2023 (dipendekkan):

```json
{
 "schema": 1,
 "scan_id": "scan-0001",
 "confidence": 0.595,
 "label": "sedang",
 "final": true,
 "components": {"pengukuran": 0.595, "air": 1.0},
 "factors": {
  "sun":   {"value": 1.0,  "class": "high",        "elevation_deg": 50.0},
  "cloud": {"value": 0.85, "class": "overcast",    "cover": 0.955, "source": "weatherkit-hourly"},
  "wet":   {"value": 0.7,  "class": "rain_before", "last_6h_mm": 11.74, "last_24h_mm": 12.73, "source": "weatherkit-hourly"},
  "water": {"value": 1.0,  "class": "ok",          "p30": {"p30_mm": 195.8, "p30_source": "weatherkit-history", "...": "..."}}
 },
 "reasons": [
  {"code": "cloud_overcast",  "factor": 0.85, "text": "Langit mendung (96%) saat terbang: cahaya rata tapi redup."},
  {"code": "wet_rain_before", "factor": 0.7,  "text": "Hujan 11,7 mm dalam 6 jam sebelum terbang: daun mungkin basah."}
 ]
}
```

Hasil tidak memuat koordinat. `final: false` berarti ada input yang belum diketahui; hasilnya bisa dihitung ulang.

## Yang sudah terlihat (notebook 04, kebun uji Tanjung Pinang)

- **Matahari:** terbang pukul **10.00–14.00 WIB** membuat faktor matahari 1 sepanjang tahun. Ini satu-satunya faktor
  yang bisa diatur petani.
- **Backcast** penerbangan pukul 10.00 setiap hari 2022–2025: 10% tinggi, 62% sedang, 28% rendah.
- **Garis awan dan daun basah belum tepat untuk lokasi ini.**
  - Tutupan awan WeatherKit pukul 10.00 bermedian 0,77, sehingga garis "cerah" 0,2 hampir tidak pernah tercapai.
    Faktor awan menurunkan 99% penerbangan.
  - Hujan ≥ 0,5 mm dalam 6 jam terjadi di 61% pagi.
  - Notebook memuat tabel sensitivitas untuk garis lain.
- **Air** jarang berpengaruh di sini: `water_stress` hanya pada 1% hari.

**Saran:** pakai confidence sebagai **log (L9)** dan untuk mengurutkan scan mana yang paling bisa dipercaya. Jangan
dulu dipakai L6 untuk mengubah rekomendasi, sampai garis awan dan daun basah dikalibrasi.

## Belum diputuskan

- **Garis awan dan daun basah.** Kalibrasi dengan terbang ulang di blok yang sama dalam beberapa hari dengan langit
  berbeda, lalu bandingkan perubahan VARI per tajuk dengan perubahan confidence.
- **Seberapa jauh L6 menurunkan kepercayaan saat `water_stress`** (faktor 0,6 masih tebakan).
- **Cara HP mengunggah manifest dan menerima hasilnya** (tim Swift/infra).
- **Seberapa dekat awan dan hujan per jam WeatherKit dengan kondisi di kebun.** Data stasiun BMKG per jam, atau
  data eksposur dari drone sendiri, bisa menjawabnya.
- **Langkah berikutnya yang diusulkan:** cek dari foto itu sendiri, yaitu porsi piksel jenuh atau gelap di dalam tajuk.

## Menjalankan

```bash
cd weather
python l6_vari_confidence.py --scan scan.json --plots plots.json --weather-dir /srv/krops/weather --out hasil.json
python -m unittest discover -s tests
```

`--weather-dir` adalah folder `--out` milik job harian; dari sana diambil `climate.json` sel tersebut. Tanpanya, normal
1991–2020 diambil langsung dari NASA POWER. Kredensial WeatherKit dibaca dari `.env`, seperti job harian.
