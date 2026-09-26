"""
Re-run SATU prompt saja lalu merge ke hasil yang sudah ada — hemat biaya API.
Dipakai mis. setelah perbaikan parser: python rerun_prompt.py cot

Efek:
  - update kolom pred_<name>/raw_<name>/time_<name> di predictions.csv
    (baris dicocokkan via nama file); kolom prompt lain tidak disentuh
  - hitung ulang metrics.json lewat dedup_report (dev/test terpisah)
"""
import os
import sys
import csv
import time

from src.model.inference_api import predict, DEFAULT_MODEL
from src.preprocessing.image_processor import load_image, get_image_paths
from src.prompts.templates import ALL_PROMPTS, STRICT_PROMPTS
from src.utils.parser import parse_output
from src.paths import SAMPLES_DIR, dir_hasil

MODEL = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
PRED_CSV = dir_hasil(MODEL) / "predictions.csv"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "cot"
    if name not in ALL_PROMPTS:
        sys.exit(f"Prompt '{name}' tidak dikenal. Pilihan: {list(ALL_PROMPTS)}")
    if not PRED_CSV.exists():
        sys.exit(f"{PRED_CSV} belum ada. Jalankan run_openrouter.py dulu.")

    prompt_text = ALL_PROMPTS[name]
    rows = list(csv.DictReader(open(PRED_CSV, encoding="utf-8")))
    by_file = {r["file"]: r for r in rows}

    paths = get_image_paths(str(SAMPLES_DIR))
    print(f"[INFO] Re-run '{name}' pada {len(paths)} gambar via {MODEL}\n")

    for path in paths:
        row = by_file.get(path.name)
        if row is None:
            print(f"  [SKIP] {path.name} tidak ada di CSV")
            continue
        image = load_image(str(path))
        t0 = time.time()
        hasil = predict(image, prompt_text, model=MODEL)
        dt = time.time() - t0
        pred = parse_output(hasil.text, strict=name in STRICT_PROMPTS)

        row[f"pred_{name}"] = pred
        row[f"raw_{name}"] = hasil.text.replace("\n", " ")
        row[f"time_{name}"] = round(dt, 1)
        row[f"finish_{name}"] = hasil.finish_reason
        row[f"provider_{name}"] = hasil.provider

        status = "OK " if pred == row["true_label"] else "X  "
        print(f"  {status}{path.name:<28} -> {pred} ({dt:.1f}s)")

    # Tulis ulang CSV (kolom prompt lain tetap seperti semula). Kolom diambil
    # dari gabungan semua baris: baris yang di-rerun bisa punya kolom baru
    # (finish_/provider_) yang belum ada di baris lain.
    kolom = list(dict.fromkeys(k for r in rows for k in r))
    with open(PRED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=kolom)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] CSV diperbarui: {PRED_CSV}")

    # Metrik dihitung ulang lewat dedup_report, bukan di sini — kalau dihitung
    # dari seluruh baris CSV, gambar duplikat ikut terhitung dan n kembali 140.
    import dedup_report
    dedup_report.main()


if __name__ == "__main__":
    main()
