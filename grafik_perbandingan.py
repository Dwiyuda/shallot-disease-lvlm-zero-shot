"""
Grafik perbandingan antar model untuk Bab IV. TIDAK memanggil API — semua
angka dibaca dari results/<nama-model>/metrics.json yang sudah ada.

Semua grafik memakai split TEST (held-out). Dev sengaja tidak dipakai karena
sudah terpakai untuk memilih prompt, sehingga angkanya optimistis.

Jalankan: python grafik_perbandingan.py

Output (results/perbandingan/):
    akurasi_model.png       akurasi + macro-F1 tiap model pada prompt terbaiknya
    matriks_model_prompt.png    heatmap akurasi seluruh kombinasi model x prompt
    sensitivitas_prompt.png     rentang akurasi antar prompt per model
    recall_per_kelas.png        recall tiap kelas per model
    ringkasan.csv               tabel yang sama dalam bentuk teks
"""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # tanpa GUI, langsung simpan berkas
import matplotlib.pyplot as plt

from src.utils.parser import VALID_CLASSES
from src.paths import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "perbandingan"
N_TEST = 80                     # ukuran split test, dipakai untuk interval kepercayaan
BIRU, JINGGA = "#4C72B0", "#DD8452"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Interval kepercayaan 95% untuk proporsi (metode Wilson).

    Dipakai, bukan rumus normal biasa, karena n=80 tergolong kecil dan akurasi
    mendekati 0,8 — di situ rumus normal memberi batas yang terlalu sempit.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    pusat = (p + z * z / (2 * n)) / d
    lebar = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, pusat - lebar), min(1.0, pusat + lebar)


def baca_semua() -> dict:
    """Kumpulkan metrik split test tiap model. Model tanpa split test dilewati."""
    hasil = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        f = d / "metrics.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        test = data.get("split", {}).get("test")
        if test:
            hasil[d.name] = test["per_prompt"]
    return hasil


def prompt_terbaik(per_prompt: dict) -> str:
    return max(per_prompt, key=lambda p: per_prompt[p]["accuracy"])


def grafik_akurasi(hasil: dict) -> None:
    """Batang akurasi + macro-F1 pada prompt terbaik tiap model, dengan galat CI."""
    model = sorted(hasil, key=lambda m: -hasil[m][prompt_terbaik(hasil[m])]["accuracy"])
    akurasi, f1, galat, label = [], [], [[], []], []
    for m in model:
        p = prompt_terbaik(hasil[m])
        a = hasil[m][p]["accuracy"]
        lo, hi = wilson(round(a * N_TEST), N_TEST)
        akurasi.append(a * 100)
        f1.append(hasil[m][p]["macro_f1"] * 100)
        galat[0].append((a - lo) * 100)
        galat[1].append((hi - a) * 100)
        label.append(f"{m}\n({p})")

    x = range(len(model))
    w = 0.38
    plt.figure(figsize=(10, 5.5))
    plt.bar([i - w / 2 for i in x], akurasi, w, label="Accuracy", color=BIRU,
            yerr=galat, capsize=4, ecolor="#555555")
    plt.bar([i + w / 2 for i in x], f1, w, label="Macro-F1", color=JINGGA)
    for i, (a, f) in enumerate(zip(akurasi, f1)):
        plt.text(i - w / 2, a + 3.5, f"{a:.1f}", ha="center", fontsize=9)
        plt.text(i + w / 2, f + 1, f"{f:.1f}", ha="center", fontsize=9)

    plt.xticks(list(x), label, fontsize=8)
    plt.ylabel("Persen (%)")
    plt.ylim(0, 100)
    plt.title(f"Perbandingan Model — prompt terbaik tiap model, split test (n={N_TEST})\n"
              f"garis galat = interval kepercayaan 95%", fontsize=11)
    plt.legend()
    plt.tight_layout()
    simpan("akurasi_model.png")


def grafik_matriks(hasil: dict) -> None:
    """Heatmap akurasi seluruh kombinasi model x prompt."""
    model = sorted(hasil, key=lambda m: -hasil[m][prompt_terbaik(hasil[m])]["accuracy"])
    prompts = sorted({p for m in hasil for p in hasil[m]})
    matriks = [[hasil[m].get(p, {}).get("accuracy", float("nan")) * 100 for p in prompts]
               for m in model]

    plt.figure(figsize=(9, 4.5))
    plt.imshow(matriks, cmap="Blues", aspect="auto", vmin=40, vmax=90)
    for i, baris in enumerate(matriks):
        for j, v in enumerate(baris):
            if not math.isnan(v):
                # Teks putih di sel gelap agar tetap terbaca
                plt.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                         color="white" if v > 72 else "#12345B")
    plt.xticks(range(len(prompts)), prompts, fontsize=9)
    plt.yticks(range(len(model)), model, fontsize=8)
    plt.title(f"Akurasi (%) tiap kombinasi model x prompt — split test (n={N_TEST})",
              fontsize=11)
    plt.colorbar(label="Akurasi (%)")
    plt.tight_layout()
    simpan("matriks_model_prompt.png")


def grafik_sensitivitas(hasil: dict) -> None:
    """
    Rentang akurasi antar prompt per model.

    Temuan utamanya: semakin kuat modelnya, semakin pendek rentangnya — prompt
    engineering berfungsi sebagai kompensasi untuk model lemah, bukan pengganda
    untuk model kuat.
    """
    model = sorted(hasil, key=lambda m: -hasil[m][prompt_terbaik(hasil[m])]["accuracy"])
    plt.figure(figsize=(9, 4.5))
    for i, m in enumerate(model):
        nilai = [v["accuracy"] * 100 for v in hasil[m].values()]
        lo, hi = min(nilai), max(nilai)
        plt.plot([lo, hi], [i, i], color="#999999", lw=2, zorder=1)
        plt.scatter(nilai, [i] * len(nilai), color=BIRU, s=45, zorder=2)
        plt.text(hi + 1.2, i, f"rentang {hi - lo:.1f} poin", va="center", fontsize=9)

    plt.yticks(range(len(model)), model, fontsize=8)
    plt.xlabel("Akurasi (%)")
    plt.xlim(40, 100)
    plt.title(f"Sensitivitas terhadap prompt — tiap titik satu prompt, split test (n={N_TEST})",
              fontsize=11)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    simpan("sensitivitas_prompt.png")


def grafik_recall_kelas(hasil: dict) -> None:
    """Recall tiap kelas per model — menunjukkan kelas mana yang tertolong."""
    model = sorted(hasil, key=lambda m: -hasil[m][prompt_terbaik(hasil[m])]["accuracy"])
    n = len(model)
    x = range(len(VALID_CLASSES))
    w = 0.8 / n

    plt.figure(figsize=(10, 5))
    warna = plt.cm.viridis([i / max(1, n - 1) for i in range(n)])
    for k, m in enumerate(model):
        p = prompt_terbaik(hasil[m])
        nilai = [hasil[m][p]["per_class"][c]["recall"] * 100 for c in VALID_CLASSES]
        plt.bar([i + k * w - 0.4 + w / 2 for i in x], nilai, w,
                label=f"{m} ({p})", color=warna[k])

    plt.xticks(list(x), ["Purple\nBlotch", "Botrytis\nLeaf Blight", "Downy\nMildew",
                         "Rust", "Healthy\nLeaf"], fontsize=9)
    plt.ylabel("Recall (%)")
    plt.ylim(0, 105)
    plt.title(f"Recall per kelas — prompt terbaik tiap model, split test (16 gambar/kelas)",
              fontsize=11)
    plt.legend(fontsize=7.5, loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    simpan("recall_per_kelas.png")


def tulis_csv(hasil: dict) -> None:
    """Tabel yang sama dalam bentuk teks, supaya angkanya bisa disalin ke naskah."""
    baris = []
    for m, pp in hasil.items():
        for p, v in pp.items():
            lo, hi = wilson(round(v["accuracy"] * N_TEST), N_TEST)
            baris.append({
                "model": m,
                "prompt": p,
                "accuracy": round(v["accuracy"] * 100, 1),
                "macro_f1": round(v["macro_f1"] * 100, 1),
                "ci_bawah": round(lo * 100, 1),
                "ci_atas": round(hi * 100, 1),
                "unknown": v.get("n_unknown", 0),
                "recall_downy_mildew": round(v["per_class"]["Downy Mildew"]["recall"] * 100, 1),
            })
    baris.sort(key=lambda b: -b["accuracy"])
    out = OUT_DIR / "ringkasan.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0]))
        w.writeheader()
        w.writerows(baris)
    print(f"  tersimpan: {out}")


def simpan(nama: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / nama
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  tersimpan: {path}")


def _self_check() -> None:
    """Wilson harus simetris di p=0,5 dan mengurung nilai titiknya."""
    lo, hi = wilson(40, 80)
    assert lo < 0.5 < hi, (lo, hi)
    assert abs((0.5 - lo) - (hi - 0.5)) < 1e-9, "di p=0,5 interval harus simetris"
    lo, hi = wilson(66, 80)                       # 82,5%
    assert lo < 0.825 < hi, (lo, hi)
    assert wilson(80, 80)[1] == 1.0, "batas atas tidak boleh melebihi 1"
    assert wilson(0, 0) == (0.0, 0.0), "n=0 tidak boleh membagi nol"
    print("Self-check interval kepercayaan lulus.")


def main() -> None:
    hasil = baca_semua()
    if not hasil:
        raise SystemExit("Belum ada model dengan split test. Jalankan run_openrouter.py dulu.")
    print(f"[INFO] {len(hasil)} model dibaca: {list(hasil)}\n")

    grafik_akurasi(hasil)
    grafik_matriks(hasil)
    grafik_sensitivitas(hasil)
    grafik_recall_kelas(hasil)
    tulis_csv(hasil)
    print(f"\n[INFO] Semua keluaran di: {OUT_DIR}")


if __name__ == "__main__":
    _self_check()
    main()
