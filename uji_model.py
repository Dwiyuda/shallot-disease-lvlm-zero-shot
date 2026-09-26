"""
Penyaringan kandidat model LVLM pada split DEV.

Semua kandidat diuji pada gambar yang SAMA PERSIS dan prompt yang sama, supaya
selisih akurasinya benar-benar berasal dari modelnya. Dev dipakai untuk memilih
— test tidak disentuh di sini agar angka finalnya tetap jujur.

Alur:
    1. python uji_model.py                    <- saring kandidat di 30 gambar dev
    2. lihat peringkatnya
    3. OPENROUTER_MODEL=<pemenang> python run_openrouter.py    <- run penuh 110 gambar

Contoh:
    python uji_model.py --dry-run             lihat rencana + estimasi panggilan
    python uji_model.py                       jalankan semua kandidat
    python uji_model.py --models qwen/qwen3-vl-235b-a22b-instruct
    python uji_model.py --prompts enhanced

Kunci API: satu OPENROUTER_API_KEY di .env berlaku untuk SEMUA model. Tidak
perlu kunci terpisah per model.

Output:
    results/<nama-model>/screening_dev.csv    prediksi mentah per gambar per prompt
    results/perbandingan_model.json           ringkasan akurasi semua kandidat
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict

from src.model.inference_api import predict, DEFAULT_MODEL
from src.preprocessing.image_processor import load_image, get_image_paths
from src.prompts.templates import ALL_PROMPTS, STRICT_PROMPTS
from src.utils.parser import parse_output
from src.paths import SAMPLES_DIR, SPLIT_CSV, RESULTS_DIR, dir_hasil

# Kandidat pilihan: satu tangga ukuran dalam keluarga Qwen, sisanya keluarga
# berbeda untuk menguji apakah temuan berlaku umum atau khusus satu vendor.
# Varian 'thinking' sengaja tidak masuk — prompt cot sudah membuktikan penalaran
# bertahap merugikan pada tugas ini (lebih lambat, akurasi lebih rendah).
MODEL_KANDIDAT = [
    "qwen/qwen3-vl-235b-a22b-instruct",
    "google/gemini-3.1-flash-lite",
    "meta-llama/llama-4-maverick",
    "google/gemini-3.1-pro-preview",
]

PROMPT_UJI = ["enhanced", "enhanced_v2"]

FOLDER_TO_CLASS = {
    "purple_blotch": "Purple Blotch",
    "botrytis":      "Botrytis Leaf Blight",
    "downy_mildew":  "Downy Mildew",
    "rust":          "Rust",
    "healthy":       "Healthy Leaf",
}

KOLOM = ["file", "true_label", "prompt", "pred", "benar", "detik",
         "finish_reason", "provider", "raw"]


def gambar_dev() -> list:
    """Path gambar split dev, terurut agar tiap model menerima urutan sama."""
    if not SPLIT_CSV.exists():
        sys.exit(f"{SPLIT_CSV} belum ada. Jalankan siapkan_dataset.py dulu.")
    dev = {r["file"] for r in csv.DictReader(open(SPLIT_CSV, encoding="utf-8"))
           if r["split"] == "dev"}
    paths = [p for p in get_image_paths(str(SAMPLES_DIR)) if p.name in dev]
    if len(paths) != len(dev):
        hilang = dev - {p.name for p in paths}
        print(f"[WARN] {len(hilang)} gambar dev tidak ditemukan di folder: {sorted(hilang)[:5]}")
    return paths


def sudah_ada(csv_path) -> set:
    """Pasangan (file, prompt) yang sudah pernah dijalankan — supaya bisa dilanjutkan."""
    if not csv_path.exists():
        return set()
    return {(r["file"], r["prompt"])
            for r in csv.DictReader(open(csv_path, encoding="utf-8"))}


def jalankan_model(model: str, paths: list, prompts: list[str], max_tokens: int) -> list[dict]:
    """
    Uji satu model pada seluruh gambar dev. Baris yang sudah ada dilewati,
    jadi skrip bisa dijalankan ulang setelah putus di tengah tanpa membayar
    dua kali.

    max_tokens harus longgar dan SAMA untuk semua kandidat. Sebagian model
    (mis. gemini-3.1-pro) memakai token penalaran internal sebelum menuliskan
    jawaban; kalau plafonnya sempit, jawabannya terpotong dan model itu
    dihukum karena alasan yang tidak ada hubungannya dengan kemampuan
    mengenali penyakit.
    """
    out_csv = dir_hasil(model) / "screening_dev.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lama = list(csv.DictReader(open(out_csv, encoding="utf-8"))) if out_csv.exists() else []
    selesai = sudah_ada(out_csv)

    baris = []
    for path in paths:
        true_label = FOLDER_TO_CLASS.get(path.parent.name.lower(), path.parent.name)
        image = None
        for nama_prompt in prompts:
            if (path.name, nama_prompt) in selesai:
                continue
            if image is None:
                image = load_image(str(path))          # baru dibuka bila perlu

            t0 = time.time()
            keluaran = predict(image, ALL_PROMPTS[nama_prompt], model=model,
                               max_tokens=max_tokens)
            dt = time.time() - t0
            pred = parse_output(keluaran.text, strict=nama_prompt in STRICT_PROMPTS)

            baris.append({
                "file": path.name,
                "true_label": true_label,
                "prompt": nama_prompt,
                "pred": pred,
                "benar": int(pred == true_label),
                "detik": round(dt, 2),
                "finish_reason": keluaran.finish_reason,
                "provider": keluaran.provider,
                "raw": keluaran.text.replace("\n", " "),
            })
            tanda = "OK " if pred == true_label else "X  "
            print(f"    {tanda}[{nama_prompt:<12}] {path.name:<28} -> {pred} ({dt:.1f}s)")

    semua = lama + baris
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KOLOM)
        w.writeheader()
        w.writerows(semua)
    print(f"    [INFO] {len(baris)} baris baru, total {len(semua)} -> {out_csv}")
    return semua


def ringkas(semua_hasil: dict, n_gambar: int) -> dict:
    """Peringkat kandidat: akurasi per (model, prompt) plus latensi rata-rata."""
    ringkasan = {}
    for model, baris in semua_hasil.items():
        per_prompt = defaultdict(lambda: {"benar": 0, "n": 0, "detik": 0.0, "unknown": 0, "terpotong": 0})
        for r in baris:
            s = per_prompt[r["prompt"]]
            s["n"] += 1
            s["benar"] += int(r["benar"])
            s["detik"] += float(r["detik"])
            s["unknown"] += int(r["pred"] == "UNKNOWN")
            s["terpotong"] += int(r.get("finish_reason") == "length")
        ringkasan[model] = {
            p: {
                "n": s["n"],
                "akurasi": round(s["benar"] / s["n"], 4) if s["n"] else 0.0,
                "detik_rata2": round(s["detik"] / s["n"], 2) if s["n"] else 0.0,
                "unknown": s["unknown"],
                "terpotong": s["terpotong"],
            }
            for p, s in per_prompt.items()
        }

    print(f"\n{'='*78}")
    print(f"PERINGKAT KANDIDAT — split dev, {n_gambar} gambar, data identik untuk semua model")
    print(f"{'='*78}")
    print(f"  {'model':<38}{'prompt':<14}{'akurasi':>9}{'detik':>8}{'UNK':>5}{'potong':>8}")
    print(f"  {'-'*82}")

    urut = sorted(
        ((m, p, v) for m, pp in ringkasan.items() for p, v in pp.items()),
        key=lambda x: -x[2]["akurasi"],
    )
    for m, p, v in urut:
        print(f"  {m:<38}{p:<14}{v['akurasi']*100:>8.1f}%{v['detik_rata2']:>7.2f}s"
              f"{v['unknown']:>5}{v['terpotong']:>8}")

    dipotong = [(m, p, v["terpotong"]) for m, p, v in urut if v["terpotong"]]
    if dipotong:
        print("\n  [PERINGATAN] Output terpotong plafon token — akurasinya tidak adil:")
        for m, p, n in dipotong:
            print(f"    {m} + {p}: {n} gambar. Ulangi dengan --max-tokens lebih besar.")

    if urut:
        m, p, v = urut[0]
        print(f"\n  Teratas: {m} + prompt '{p}' = {v['akurasi']*100:.1f}%")
        print(f"  Lanjutkan run penuh:  OPENROUTER_MODEL={m} python run_openrouter.py")
    return ringkasan


def _self_check():
    """Agregasi harus menghitung benar, UNKNOWN, dan rata-rata latensi dengan tepat."""
    palsu = {"model-x": [
        {"prompt": "enhanced", "benar": 1, "detik": "1.0", "pred": "Rust"},
        {"prompt": "enhanced", "benar": 0, "detik": "3.0", "pred": "UNKNOWN"},
        {"prompt": "enhanced", "benar": 1, "detik": "2.0", "pred": "Rust"},
    ]}
    r = ringkas(palsu, 3)["model-x"]["enhanced"]
    assert r["n"] == 3, r
    assert r["akurasi"] == round(2 / 3, 4), r
    assert r["detik_rata2"] == 2.0, r
    assert r["unknown"] == 1, r
    print("Self-check agregasi lulus.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Saring kandidat model LVLM di split dev.")
    ap.add_argument("--models", default=",".join(MODEL_KANDIDAT),
                    help="daftar model dipisah koma")
    ap.add_argument("--prompts", default=",".join(PROMPT_UJI),
                    help="daftar prompt dipisah koma")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="plafon token output, sama untuk semua kandidat (default 2048)")
    ap.add_argument("--dry-run", action="store_true", help="tampilkan rencana saja")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    tak_dikenal = [p for p in prompts if p not in ALL_PROMPTS]
    if tak_dikenal:
        sys.exit(f"Prompt tidak dikenal: {tak_dikenal}. Pilihan: {list(ALL_PROMPTS)}")

    paths = gambar_dev()
    total = len(paths) * len(prompts) * len(models)
    print(f"[INFO] {len(models)} model x {len(prompts)} prompt x {len(paths)} gambar dev"
          f" = {total} panggilan API")
    print(f"[INFO] Prompt diuji : {prompts}")
    print(f"[INFO] Model diuji  : {models}\n")
    if args.dry_run:
        print("[DRY RUN] tidak ada panggilan API dilakukan.")
        return

    semua_hasil, gagal = {}, {}
    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model}")
        try:
            semua_hasil[model] = jalankan_model(model, paths, prompts, args.max_tokens)
        except Exception as e:
            # Satu model bermasalah (tidak tersedia, saldo kurang, rate limit)
            # tidak boleh menggagalkan kandidat lain.
            gagal[model] = str(e)
            print(f"    [GAGAL] {e}\n")

    if not semua_hasil:
        sys.exit("Tidak ada model yang berhasil diuji.")

    ringkasan = ringkas(semua_hasil, len(paths))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "perbandingan_model.json"
    out.write_text(json.dumps({
        "split": "dev",
        "n_gambar": len(paths),
        "prompt_diuji": prompts,
        "max_tokens": args.max_tokens,
        "hasil": ringkasan,
        "gagal": gagal,
        "catatan": "Penyaringan pada dev. Angka final harus dari split test "
                   "lewat run_openrouter.py + dedup_report.py.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] Ringkasan disimpan: {out}")
    if gagal:
        print(f"[WARN] Model gagal diuji: {list(gagal)}")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
