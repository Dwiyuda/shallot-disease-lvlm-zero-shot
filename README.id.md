# Identifikasi Penyakit Daun Bawang Merah dengan LVLM Zero-Shot

Eksperimen *prompt engineering* pada Large Vision-Language Model (LVLM) untuk
mengklasifikasi penyakit daun bawang merah (*Allium cepa* var. *aggregatum*)
**tanpa fine-tuning**. Model dipanggil apa adanya; yang divariasikan hanya
susunan prompt dan pilihan model.

**Kelas (5):** Purple Blotch · Botrytis Leaf Blight · Downy Mildew · Rust · Healthy Leaf
**Data uji:** 110 gambar unik, **22 per kelas** (seimbang)
**Pembagian:** dev 30 (pemilih prompt) · test 80 (angka final)

---

## Hasil — perbandingan 4 model

Angka yang dilaporkan adalah **split test (held-out)**. Dev tidak dipakai untuk
angka final karena sudah terpakai memilih prompt dan model.

| Model | Prompt terbaik | Akurasi | Macro-F1 | 95% CI |
|---|---|---|---|---|
| **`google/gemini-3.1-flash-lite`** | **enhanced_v2** | **82,5%** | 82,6% | 72,7–89,3% |
| `qwen/qwen3-vl-235b-a22b-instruct` | enhanced_v2 | 76,2% | 76,4% | 65,9–84,2% |
| `meta-llama/llama-4-maverick` | enhanced | 72,5% | 72,5% | 61,9–81,1% |
| `qwen/qwen3-vl-8b-instruct` *(awal)* | enhanced | 67,5% | 66,5% | 56,6–76,8% |

Uji McNemar eksak + Holm pada test: Gemini **secara statistik lebih baik**
dari model awal (p=0,0452). Selisih terhadap model lain arahnya konsisten
tetapi belum terbukti pada n=80.

Grafik lengkap: `results/perbandingan/` — akurasi antar model, heatmap 19
kombinasi model x prompt, sensitivitas prompt, dan recall per kelas.

### Temuan utama: sensitivitas prompt bergantung kekuatan model

| Model | Rentang akurasi antar 5 prompt |
|---|---|
| `gemini-3.1-flash-lite` | 5,0 poin |
| `llama-4-maverick` | 5,0 poin |
| `qwen3-vl-235b` | 16,2 poin |
| `qwen3-vl-8b` | 18,8 poin |

Prompt engineering berfungsi sebagai **kompensasi untuk model lemah**, bukan
pengganda untuk model kuat.

### Downy Mildew — kelemahan utama, teratasi dengan ganti model

| Model | Recall DM | Daun sakit divonis sehat |
|---|---|---|
| `qwen3-vl-235b` | 81,2% | 4 dari 64 |
| **`gemini-3.1-flash-lite`** | 75,0% | **0 dari 64** |
| `llama-4-maverick` | 56,2% | 4 dari 64 |
| `qwen3-vl-8b` | 37,5% | 10 dari 64 |

### Latensi

| Model | Rata-rata | Median |
|---|---|---|
| `gemini-3.1-flash-lite` | 1,72 dtk | 1,40 dtk |
| `qwen3-vl-235b` | 1,79 dtk | 1,50 dtk |
| `llama-4-maverick` | 2,01 dtk | 1,80 dtk |
| `qwen3-vl-8b` | 0,70 dtk | 0,60 dtk |

---

## Rincian model awal `qwen/qwen3-vl-8b-instruct`

Angka yang dilaporkan adalah **split test (held-out)**. Dev tidak pernah
dipakai untuk angka final karena sudah terpakai memilih prompt.

| Prompt | dev (n=30) | **test (n=80)** | 95% CI test | Macro-F1 test |
|---|---|---|---|---|
| **enhanced** | 80,0% | **67,5%** | 56,6% – 76,8% | 66,5% |
| structured | 73,3% | 60,0% | 49,0% – 70,0% | 55,3% |
| cot | 70,0% | 53,8% | 42,9% – 64,3% | 51,3% |
| direct | 70,0% | 48,8% | 38,1% – 59,5% | 44,3% |

> Selisih dev 80,0% vs test 67,5% adalah **bias pemilihan** — prompt terbaik
> tampak lebih baik pada data yang dipakai memilihnya. Urutan peringkat
> bertahan di test, jadi pemilihan `enhanced` valid; hanya angkanya yang
> sebelumnya optimistis.

Uji McNemar eksak dengan koreksi Holm, dijalankan pada split test:

| Pasangan | p (Holm) | Kesimpulan |
|---|---|---|
| enhanced vs direct | 0,0044 | **beda nyata** |
| enhanced vs cot | 0,0586 | tidak nyata |
| enhanced vs structured | 0,3750 | tidak nyata |

Rincian per kelas untuk `enhanced` pada test (16 gambar per kelas):

| Kelas | Precision | Recall |
|---|---|---|
| Healthy Leaf | 60,0% | 93,8% |
| Botrytis Leaf Blight | 56,5% | 81,2% |
| Rust | 91,7% | 68,8% |
| Purple Blotch | 75,0% | 56,2% |
| **Downy Mildew** | 75,0% | **37,5%** |

---

## Prompt yang diuji

| Nama | Teknik | Status |
|---|---|---|
| `direct` | zero-shot + persona + daftar kelas | diuji |
| `cot` | zero-shot chain-of-thought (3 langkah) | diuji |
| `structured` | constraint/format prompting | diuji |
| `enhanced` | persona + deskripsi visual rinci + negative prompting | diuji |
| `enhanced_v2` | enhanced + urutan pemeriksaan difus + gerbang sehat + dugaan kedua | **terbaik untuk Gemini & Qwen-235B** |

`enhanced_v2` dirancang dari analisis kesalahan `enhanced`: model terlalu mudah
menjawab "Healthy Leaf" (precision 60,0%) dan buta terhadap gejala difus Downy
Mildew (recall 37,5%). Field `CONFIDENCE` diganti `SECOND_GUESS` karena yang
lama tidak informatif — model awal menjawab "High" untuk 121 dari 124 gambar.

---

## Cara menjalankan

```bash
pip install -r requirements.txt
```

Salin `.env.example` menjadi `.env`, lalu isi `OPENROUTER_API_KEY`
(ambil di https://openrouter.ai/keys).

| Perintah | Fungsi | Panggilan API |
|---|---|---|
| `python siapkan_dataset.py` | Dedup, seimbangkan kelas, bagi dev/test | 0 |
| `python run_openrouter.py` | Eksperimen penuh semua prompt | 110 × jumlah prompt |
| `python rerun_prompt.py cot` | Ulangi satu prompt, gabungkan ke hasil lama | 110 |
| `python dedup_report.py` | Hitung metrik dev/test dari CSV | 0 |
| `python analisis_statistik.py` | McNemar, latensi, stabilitas provider | 0 |
| `python uji_model.py` | Saring kandidat model di split dev | 30 x model x prompt |
| `python grafik_perbandingan.py` | Grafik antar model untuk Bab IV | 0 |
| `python -m src.evaluation.visualizer` | Regenerasi confusion matrix & grafik | 0 |

Mengganti model — semua skrip mengikuti variabel yang sama, dan hasilnya
otomatis masuk folder terpisah:

```bash
OPENROUTER_MODEL=qwen/qwen3-vl-32b-instruct python run_openrouter.py
```

Self-check modul — tanpa API, tanpa framework tes:

```bash
python -m src.utils.parser
python -m src.preprocessing.image_processor
python -m src.evaluation.metrics
python -m src.model.inference_api
python uji_model.py --self-check
python -c "import analisis_statistik as a; a._self_check()"
python -c "import grafik_perbandingan as g; g._self_check()"
```

---

## Struktur

```
siapkan_dataset.py        dedup + seimbangkan kelas + bagi dev/test (sekali jalan)
run_openrouter.py         eksperimen penuh (satu-satunya yang memetakan folder -> label)
rerun_prompt.py           ulangi satu prompt, gabungkan ke hasil lama
dedup_report.py           hitung metrik dev/test (satu pintu perhitungan metrik)
analisis_statistik.py     McNemar + Holm, latensi, stabilitas provider
uji_model.py              saring kandidat model di split dev
grafik_perbandingan.py    grafik antar model (akurasi, matriks, sensitivitas)

src/
  model/inference_api.py           panggilan OpenRouter + jejak provider/finish_reason
  prompts/templates.py             5 varian prompt + daftar prompt berformat wajib
  utils/parser.py                  teks bebas model -> label kelas baku
  preprocessing/image_processor.py load gambar + dedup berbasis hash isi file
  evaluation/metrics.py            accuracy/precision/recall/F1/confusion matrix
  evaluation/visualizer.py         PNG untuk Bab IV
  disease_info.py                  info penyakit statis
  paths.py                         path terpusat + dir_hasil(model)

data/
  samples/          110 gambar, 22 per kelas
  duplikat/         16 salinan byte-identik yang disingkirkan
  cadangan/         14 kelebihan agar tiap kelas sama banyak
  split.csv         penetapan dev/test per gambar (benih 42)
  duplikat.csv      daftar grup duplikat (bukti/lampiran)

results/
  perbandingan/                    grafik & tabel antar model (Bab IV)
  perbandingan_model.json          ringkasan penyaringan dev
  <nama-model>/                    satu folder per model yang diuji
    predictions.csv                bukti mentah: prediksi, raw, latensi, provider
    metrics.json                   metrik dev, test, dan gabungan
    analisis_statistik.json        McNemar, latensi, stabilitas
    figures/                       confusion matrix + grafik perbandingan
    arsip/                         hasil run sebelumnya
```

Perhitungan metrik hanya ada di `dedup_report.py`; skrip lain memanggilnya.
Dengan begitu angka di CSV, JSON, dan figur pasti berasal dari definisi yang sama.

---

## Keterbatasan

**Dataset.** Dari 140 berkas awal, 16 duplikat byte-identik disingkirkan dan 14
sisanya dipindah ke cadangan agar tiap kelas berisi 22 gambar. Seluruh gambar
berukuran 640×640 — indikasi dataset sudah melalui praproses/augmentasi,
sehingga sebagian gambar mungkin varian dari foto yang sama.

> **Harus dilengkapi sebelum evaluasi:** sumber dataset (tautan/penerbit),
> lisensi penggunaan, cara pengambilan sampel, dan siapa yang memverifikasi
> label ground truth. Empat hal ini belum terdokumentasi di mana pun.

**Ukuran sampel.** Dengan test n=80, interval kepercayaan akurasi selebar
sekitar 20 poin. Untuk mempersempit menjadi ±5 poin dibutuhkan sekitar 500
gambar. Angka per kelas (16 per kelas) hanya indikatif.

**Reproducibility.** `temperature=0` hanya deterministik di dalam satu provider.
OpenRouter merutekan permintaan ke provider berbeda tanpa pemberitahuan — pada
run yang dilaporkan, 404 panggilan dilayani Alibaba dan 92 oleh Parasail.
Menjalankan ulang eksperimen yang sama menghasilkan prediksi identik hanya pada
91,9%–98,4% gambar. Kolom `provider_*` dicatat per panggilan sebagai bukti.

**Downy Mildew tetap kelas tersulit.** Pada model awal recall-nya hanya 37,5%
dan sebagian besar kesalahannya diklasifikasi sebagai *Healthy Leaf*. Ganti
model menaikkannya ke 75,0% (Gemini) dan 81,2% (Qwen-235B), tetapi kelas ini
tetap yang terendah — konsisten dengan sifat embun bulu yang difus dan tidak
membentuk bercak tegas.

**Confidence model tidak terkalibrasi.** Model awal menjawab `CONFIDENCE: High`
untuk 121 dari 124 gambar pada eksperimen sebelum data diseimbangkan. Nilai ini tidak boleh
dipakai sebagai ukuran keandalan; `enhanced_v2` menggantinya dengan dugaan kedua.

**Field KEY FEATURE bukan bukti penalaran.** Kemiripan teksnya dengan deskripsi
kelas di dalam prompt praktis sama besar untuk prediksi benar (35%) maupun
salah (37%) — model menyusun pembenaran mengikuti templat, bukan melaporkan
yang benar-benar dilihat.

**Perbedaan antar model belum semuanya terbukti.** Dari 6 pasangan model yang
diuji McNemar, hanya Gemini vs model awal yang lolos uji signifikansi setelah
koreksi Holm. Selisih terhadap Qwen-235B dan Llama arahnya konsisten tetapi
belum terbukti pada n=80.

**Bukan alat diagnosis.** Dengan akurasi terbaik 82,5% (CI 72,7–89,3%) dan
Downy Mildew 75,0%, keluaran sistem ini belum layak menjadi dasar tunggal
keputusan pemakaian fungisida. Teks pada
`src/disease_info.py` adalah rangkuman umum dan **belum diverifikasi ke sumber
agronomi resmi**.

---

## Catatan keamanan

`.env` berisi API key asli dan sudah masuk `.gitignore`, tetapi itu **tidak**
melindungi saat folder di-zip dan dikirim ke ahli domain atau diunggah ke
repositori kampus. Kosongkan nilainya sebelum membagikan folder, dan cabut
(*revoke*) key di dasbor OpenRouter bila pernah ikut terkirim.
