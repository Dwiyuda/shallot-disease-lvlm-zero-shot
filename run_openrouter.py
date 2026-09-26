"""
Eksperimen zero-shot penuh untuk SATU model via OpenRouter.
Model ditentukan variabel OPENROUTER_MODEL; hasilnya masuk folder terpisah
per model sehingga beberapa model bisa dibandingkan tanpa saling menimpa.

Setup: salin .env.example → .env, isi OPENROUTER_API_KEY.
Jalankan:
    python run_openrouter.py                       # data/samples, semua prompt
    python run_openrouter.py data/samples           # sama
    # ganti model: set OPENROUTER_MODEL di .env (mis. qwen/qwen3-vl-8b-thinking)

Output:
    results/<nama-model>/predictions.csv   — prediksi, raw, latensi, provider
    results/<nama-model>/metrics.json      — metrik dev/test (via dedup_report)
"""
import os
import sys
import csv
import time
import shutil
from pathlib import Path
from collections import Counter

from src.model.inference_api import predict, DEFAULT_MODEL
from src.preprocessing.image_processor import load_image, get_image_paths
from src.prompts.templates import ALL_PROMPTS, STRICT_PROMPTS
from src.utils.parser import parse_output
from src.paths import SAMPLES_DIR, dir_hasil

# Mapping nama subfolder → label kelas baku. Satu-satunya tempat ground truth
# diturunkan dari struktur folder; skrip lain membacanya dari kolom true_label
# di CSV, jadi tabel ini tidak perlu diduplikasi.
FOLDER_TO_CLASS = {
    "purple_blotch": "Purple Blotch",
    "botrytis":      "Botrytis Leaf Blight",
    "downy_mildew":  "Downy Mildew",
    "rust":          "Rust",
    "healthy":       "Healthy Leaf",
}

MODEL = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
HASIL = dir_hasil(MODEL)


def folder_ke_label(folder_name: str) -> str:
    return FOLDER_TO_CLASS.get(folder_name.lower(), folder_name)


def jalankan(folder: str) -> list[dict]:
    paths = get_image_paths(folder)
    if not paths:
        print(f"[ERROR] Tidak ada gambar di: {folder}")
        sys.exit(1)

    n_call = len(paths) * len(ALL_PROMPTS)
    print(f"[INFO] Model : {MODEL}")
    print(f"[INFO] {len(paths)} gambar × {len(ALL_PROMPTS)} prompt = {n_call} panggilan API\n")

    hasil = []
    for path in paths:
        image = load_image(str(path))
        true_label = folder_ke_label(path.parent.name)
        row = {"file": path.name, "true_label": true_label}

        for nama_prompt, prompt_text in ALL_PROMPTS.items():
            t0 = time.time()
            keluaran = predict(image, prompt_text, model=MODEL)
            elapsed = time.time() - t0
            prediksi = parse_output(keluaran.text, strict=nama_prompt in STRICT_PROMPTS)

            row[f"pred_{nama_prompt}"] = prediksi
            row[f"raw_{nama_prompt}"] = keluaran.text.replace("\n", " ")  # penuh, untuk bukti & re-parse
            row[f"time_{nama_prompt}"] = round(elapsed, 1)
            row[f"finish_{nama_prompt}"] = keluaran.finish_reason   # 'length' = output terpotong
            row[f"provider_{nama_prompt}"] = keluaran.provider      # jejak routing OpenRouter

            status = "OK " if prediksi == true_label else "X  "
            print(f"  {status}[{nama_prompt:10s}] {path.name:<25} -> {prediksi} ({elapsed:.1f}s)")

        hasil.append(row)
        print()

    return hasil


def simpan_csv(hasil: list[dict], output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # CSV prediksi adalah bukti mentah eksperimen — jangan ditimpa tanpa salinan.
    if out.exists():
        cadangan = out.with_name(f"{out.stem}_sebelumnya{out.suffix}")
        shutil.copy2(out, cadangan)
        print(f"[INFO] CSV lama dicadangkan: {cadangan}")

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=hasil[0].keys())
        writer.writeheader()
        writer.writerows(hasil)
    print(f"[INFO] Prediksi disimpan: {out}")


def ringkas_provider(hasil: list[dict]) -> None:
    """Tampilkan provider mana saja yang melayani run ini — jejak reproducibility."""
    dipakai = Counter(
        r[f"provider_{p}"] for r in hasil for p in ALL_PROMPTS if r.get(f"provider_{p}")
    )
    terpotong = sum(1 for r in hasil for p in ALL_PROMPTS if r.get(f"finish_{p}") == "length")
    print(f"\n[INFO] Provider OpenRouter: {dict(dipakai) or '(tidak dilaporkan API)'}")
    print(f"[INFO] Output terpotong plafon token: {terpotong}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else str(SAMPLES_DIR)
    hasil = jalankan(folder)
    simpan_csv(hasil, HASIL / "predictions.csv")
    ringkas_provider(hasil)

    # Metrik dihitung lewat dedup_report supaya definisinya satu pintu
    # (dedup + blok metadata) — jangan diduplikasi di sini.
    import dedup_report
    dedup_report.main()
