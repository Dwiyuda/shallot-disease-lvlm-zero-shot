"""
Rapikan data uji sekali jalan: buang duplikat, seimbangkan jumlah per kelas,
lalu bagi menjadi dev (pemilih prompt) dan test (angka final).

Kenapa dibagi: prompt terbaik dipilih berdasarkan data uji. Kalau angka final
juga dilaporkan dari data yang sama, angkanya optimistis — prompt itu menang
sebagian karena benar-benar lebih baik, sebagian karena kebetulan cocok dengan
gambar-gambar tersebut. Dev dipakai memilih, test hanya dibuka di akhir.

Tidak ada gambar yang dihapus: duplikat dan kelebihan dipindah ke folder lain
agar bisa dikembalikan kapan saja.

Jalankan: python siapkan_dataset.py
          python siapkan_dataset.py --dry-run    # lihat rencana tanpa memindah

Output:
    data/samples/       tinggal 22 gambar per kelas (110 total, seimbang)
    data/duplikat/      salinan byte-identik yang disingkirkan
    data/cadangan/      kelebihan gambar agar tiap kelas sama banyak
    data/split.csv      penetapan dev/test per gambar
"""
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.preprocessing.image_processor import get_image_paths, find_duplicate_groups
from src.paths import DATA_DIR, SAMPLES_DIR, SPLIT_CSV

PER_KELAS = 22          # dibatasi kelas terkecil (purple_blotch)
N_DEV = 6               # cukup untuk memeringkat prompt dan menyaring model
N_TEST = PER_KELAS - N_DEV
SEED = 42               # tetap, agar pembagian bisa direproduksi persis

DUPLIKAT_DIR = DATA_DIR / "duplikat"
CADANGAN_DIR = DATA_DIR / "cadangan"


def pindah(src: Path, tujuan_dir: Path, dry: bool) -> None:
    tujuan = tujuan_dir / src.parent.name / src.name
    print(f"    {src.name:<30} -> {tujuan.relative_to(DATA_DIR)}")
    if dry:
        return
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(tujuan))


def main(dry: bool = False) -> None:
    if dry:
        print("[DRY RUN] tidak ada berkas yang benar-benar dipindah\n")

    # ── 1. Singkirkan duplikat byte-identik ──────────────────────────────────
    grup = find_duplicate_groups(str(SAMPLES_DIR))
    n_dup = sum(len(g) - 1 for g in grup)
    print(f"1. Duplikat byte-identik: {len(grup)} grup, {n_dup} salinan disingkirkan")
    for g in grup:
        for salinan in g[1:]:                       # g[0] dipertahankan
            pindah(salinan, DUPLIKAT_DIR, dry)
    if not grup:
        print("    (tidak ada — mungkin sudah pernah dijalankan)")

    # ── 2. Seimbangkan jumlah per kelas ──────────────────────────────────────
    # Setelah duplikat pindah, baca ulang isi folder yang sebenarnya.
    tersisa = defaultdict(list)
    for p in get_image_paths(str(SAMPLES_DIR), unique=False):
        if dry and any(p.name == s.name for g in grup for s in g[1:]):
            continue                                 # anggap sudah pindah
        tersisa[p.parent.name].append(p)

    kurang = {k: len(v) for k, v in tersisa.items() if len(v) < PER_KELAS}
    if kurang:
        sys.exit(f"[ERROR] Kelas berikut punya < {PER_KELAS} gambar unik: {kurang}")

    print(f"\n2. Menyeimbangkan ke {PER_KELAS} gambar per kelas")
    terpilih = {}
    for kelas in sorted(tersisa):
        urut = sorted(tersisa[kelas], key=lambda p: p.name)
        # Acak dengan benih tetap agar pemilihan tidak bergantung urutan nama
        # (penomoran berkas tidak selalu netral terhadap isi gambar).
        rng = random.Random(f"{SEED}-{kelas}")
        rng.shuffle(urut)
        terpilih[kelas] = urut[:PER_KELAS]
        buang = urut[PER_KELAS:]
        print(f"  {kelas}: {len(urut)} -> {PER_KELAS}  ({len(buang)} ke cadangan)")
        for p in buang:
            pindah(p, CADANGAN_DIR, dry)

    # ── 3. Bagi dev / test, seimbang di kedua sisi ───────────────────────────
    baris = []
    for kelas, paths in terpilih.items():
        for i, p in enumerate(paths):
            baris.append({
                "file": p.name,
                "kelas": kelas,
                "split": "dev" if i < N_DEV else "test",
            })

    print(f"\n3. Pembagian dev/test (benih={SEED})")
    ringkas = Counter((b["kelas"], b["split"]) for b in baris)
    print(f"  {'kelas':<16}{'dev':>5}{'test':>6}{'total':>7}")
    for kelas in sorted(terpilih):
        d, t = ringkas[(kelas, "dev")], ringkas[(kelas, "test")]
        print(f"  {kelas:<16}{d:>5}{t:>6}{d+t:>7}")
    total_dev = sum(1 for b in baris if b["split"] == "dev")
    print(f"  {'TOTAL':<16}{total_dev:>5}{len(baris)-total_dev:>6}{len(baris):>7}")

    if dry:
        print(f"\n[DRY RUN] {SPLIT_CSV.name} tidak ditulis.")
        return

    with open(SPLIT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "kelas", "split"])
        w.writeheader()
        w.writerows(sorted(baris, key=lambda b: (b["kelas"], b["split"], b["file"])))
    print(f"\n[INFO] Pembagian disimpan: {SPLIT_CSV}")
    print("[INFO] Hitung ulang metrik: python dedup_report.py")


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
