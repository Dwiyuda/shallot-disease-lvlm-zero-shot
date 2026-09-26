"""
Hitung metrik satu model dari CSV prediksinya. TIDAK memanggil API dan TIDAK
menghapus berkas gambar apa pun.

Metrik dilaporkan terpisah untuk dev (dasar pemilihan prompt) dan test (angka
final), mengikuti data/split.csv. Duplikat tetap disaring sebagai pengaman,
meski folder sudah dibersihkan oleh siapkan_dataset.py.

Jalankan: python dedup_report.py
          OPENROUTER_MODEL=qwen/qwen3-vl-32b-instruct python dedup_report.py

Output (mengikuti model aktif):
    results/<nama-model>/metrics.json   — metrik dev, test, dan gabungan
    data/duplikat.csv                   — daftar grup gambar byte-identik
"""
import csv
import json
import os
import shutil
from pathlib import Path
from collections import Counter

from src.preprocessing.image_processor import get_image_paths, find_duplicate_groups
from src.prompts.templates import ALL_PROMPTS
from src.evaluation.metrics import compute_metrics, print_report
from src.model.inference_api import DEFAULT_MODEL
from src.paths import SAMPLES_DIR, SPLIT_CSV, DUPLIKAT_CSV, dir_hasil

MODEL = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
HASIL = dir_hasil(MODEL)
PRED_CSV = HASIL / "predictions.csv"
METRICS_JSON = HASIL / "metrics.json"
DUP_CSV = DUPLIKAT_CSV


def tulis_laporan_duplikat(groups: list[list[Path]]) -> None:
    """Simpan daftar duplikat sebagai bukti — satu baris per salinan yang dibuang."""
    DUP_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(DUP_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grup", "kelas", "dipertahankan", "duplikat"])
        for i, grup in enumerate(groups, 1):
            kept = grup[0]
            for dup in grup[1:]:
                w.writerow([i, kept.parent.name, kept.name, dup.name])
    print(f"[INFO] Laporan duplikat: {DUP_CSV}")


def ringkas_dataset(groups: list[list[Path]], semua: list[Path], unik: list[Path]) -> None:
    per_kelas = Counter(g[0].parent.name for g in groups for _ in g[1:])
    print(f"\n{'='*58}")
    print("DUPLIKAT PADA DATA UJI")
    print(f"{'='*58}")
    print(f"  Total gambar        : {len(semua)}")
    print(f"  Grup duplikat       : {len(groups)}")
    print(f"  Salinan dibuang     : {len(semua) - len(unik)}")
    print(f"  Data uji efektif    : {len(unik)}\n")
    print(f"  {'Kelas':<16}{'semula':>8}{'unik':>7}{'duplikat':>10}")
    print(f"  {'-'*41}")
    semula = Counter(p.parent.name for p in semua)
    bersih = Counter(p.parent.name for p in unik)
    for kelas in sorted(semula):
        print(f"  {kelas:<16}{semula[kelas]:>8}{bersih[kelas]:>7}{per_kelas.get(kelas, 0):>10}")
    print(f"{'='*58}")


def main() -> None:
    if not PRED_CSV.exists():
        raise SystemExit(f"{PRED_CSV} belum ada. Jalankan run_openrouter.py dulu.")

    semua = get_image_paths(str(SAMPLES_DIR), unique=False)
    unik = get_image_paths(str(SAMPLES_DIR))
    groups = find_duplicate_groups(str(SAMPLES_DIR))

    ringkas_dataset(groups, semua, unik)
    tulis_laporan_duplikat(groups)

    # Saring baris prediksi ke gambar unik saja (dicocokkan via nama file)
    rows = list(csv.DictReader(open(PRED_CSV, encoding="utf-8")))
    nama_unik = {p.name for p in unik}
    dedup = [r for r in rows if r["file"] in nama_unik]

    hilang = nama_unik - {r["file"] for r in rows}
    if hilang:
        print(f"[WARN] {len(hilang)} gambar unik belum ada prediksinya di CSV: {sorted(hilang)[:5]}")
    print(f"[INFO] Baris prediksi: {len(rows)} -> {len(dedup)} setelah dedup\n")

    # Cadangkan metrik sebelum ditimpa, sekali saja
    lama = HASIL / "arsip" / "metrics_sebelum_split.json"
    if METRICS_JSON.exists() and not lama.exists():
        lama.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(METRICS_JSON, lama)
        print(f"[INFO] Metrik lama dicadangkan: {lama}")

    model = MODEL

    # Pembagian dev/test dari siapkan_dataset.py. Dev dipakai MEMILIH prompt,
    # test hanya untuk angka final — memisahkannya mencegah angka yang
    # optimistis karena prompt dipilih dan dinilai pada gambar yang sama.
    bagian = {"all": dedup}
    if SPLIT_CSV.exists():
        peta = {r["file"]: r["split"] for r in csv.DictReader(open(SPLIT_CSV, encoding="utf-8"))}
        bagian["dev"] = [r for r in dedup if peta.get(r["file"]) == "dev"]
        bagian["test"] = [r for r in dedup if peta.get(r["file"]) == "test"]
        print(f"[INFO] Split terbaca: dev={len(bagian['dev'])}, test={len(bagian['test'])}\n")
    else:
        print(f"[WARN] {SPLIT_CSV.name} belum ada — jalankan siapkan_dataset.py dulu.\n")

    # Prompt yang belum pernah dijalankan pada model ini tidak punya kolomnya
    # di CSV — lewati, jangan bikin skrip mati.
    tersedia = [p for p in ALL_PROMPTS if f"pred_{p}" in dedup[0]]
    belum = [p for p in ALL_PROMPTS if p not in tersedia]
    if belum:
        print(f"[INFO] Prompt belum diuji pada model ini, dilewati: {belum}\n")

    hasil_split = {}
    for nama_split, baris in bagian.items():
        if not baris:
            continue
        metrics_all = {}
        for nama_prompt in tersedia:
            trues = [r["true_label"] for r in baris]
            preds = [r[f"pred_{nama_prompt}"] for r in baris]
            m = compute_metrics(trues, preds)
            if nama_split != "all":                 # cetak yang dipakai penelitian saja
                print_report(m, f"[{nama_split}] {nama_prompt} — {model} (n={len(baris)})")
            metrics_all[nama_prompt] = m
        hasil_split[nama_split] = {"n_samples": len(baris), "per_prompt": metrics_all}

    METRICS_JSON.write_text(
        json.dumps(
            {
                "model": model,
                "n_samples": len(dedup),
                "dedup": {
                    "n_raw": len(rows),
                    "n_duplicate_removed": len(rows) - len(dedup),
                    "method": "md5 isi file; salinan pertama (urutan nama) dipertahankan",
                },
                "catatan_split": "dev = dasar pemilihan prompt; test = angka final yang dilaporkan",
                "split": hasil_split,
                "per_prompt": hasil_split["all"]["per_prompt"],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[INFO] Metrik disimpan: {METRICS_JSON}")
    print("[INFO] Regenerasi figur: python -m src.evaluation.visualizer")


if __name__ == "__main__":
    main()
