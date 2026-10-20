"""
Analisis statistik pendukung hasil, dihitung dari CSV prediksi yang sudah ada.
TIDAK memanggil API.

Isi:
  1. Uji McNemar antar pasangan prompt (+ koreksi Holm untuk 6 perbandingan)
  2. Latensi & panjang output per prompt
  3. Stabilitas: pengaruh provider OpenRouter dan perbandingan antar-run

Jalankan: python analisis_statistik.py

Output: results/<nama-model>/analisis_statistik.json + tabel di layar
"""
import csv
import json
import statistics as st
from math import comb
from collections import Counter, defaultdict

from src.prompts.templates import ALL_PROMPTS
from src.preprocessing.image_processor import get_image_paths
from src.model.inference_api import DEFAULT_MODEL
from src.paths import SAMPLES_DIR, SPLIT_CSV, dir_hasil

import os
MODEL = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
HASIL = dir_hasil(MODEL)
PRED_CSV = HASIL / "predictions.csv"
PREV_CSV = HASIL / "arsip" / "predictions_sebelumnya.csv"
OUT_JSON = HASIL / "analisis_statistik.json"



def mcnemar_exact(b: int, c: int) -> float:
    """
    Uji McNemar versi eksak (binomial dua sisi).

    b = jumlah gambar yang benar di prompt A tapi salah di B
    c = kebalikannya. Gambar yang sama-sama benar/salah tidak berpengaruh.

    Versi eksak dipakai, bukan aproksimasi chi-square, karena b+c di sini
    kecil (belasan) — aproksimasi tidak valid di rentang itu.
    """
    n = b + c
    if n == 0:
        return 1.0
    ekor = sum(comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * ekor / 2 ** n)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """
    Koreksi Holm-Bonferroni untuk banyak perbandingan sekaligus.
    Tanpa ini, menguji 6 pasangan pada alpha 5% membuat peluang salah
    menyatakan 'ada beda' naik menjadi sekitar 26%.
    """
    urut = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(urut)
    hasil, sebelumnya = {}, 0.0
    for i, (nama, p) in enumerate(urut):
        disesuaikan = max(sebelumnya, min(1.0, (m - i) * p))  # jaga urutan monoton
        hasil[nama] = disesuaikan
        sebelumnya = disesuaikan
    return hasil


def baca_baris() -> list[dict]:
    """
    Ambil baris prediksi untuk gambar TEST saja. Uji signifikansi harus
    dijalankan pada data yang tidak ikut memilih prompt; kalau dev disertakan,
    p-value-nya optimistis.
    """
    nama_unik = {p.name for p in get_image_paths(str(SAMPLES_DIR))}
    rows = [r for r in csv.DictReader(open(PRED_CSV, encoding="utf-8")) if r["file"] in nama_unik]

    if SPLIT_CSV.exists():
        peta = {r["file"]: r["split"] for r in csv.DictReader(open(SPLIT_CSV, encoding="utf-8"))}
        rows = [r for r in rows if peta.get(r["file"]) == "test"]
        print("[INFO] Analisis dijalankan pada split TEST (held-out)")

    if not rows:
        raise SystemExit(f"Tidak ada baris terpakai di {PRED_CSV}.")
    return rows


def uji_mcnemar(rows: list[dict], PROMPTS: list[str]) -> dict:
    benar = {p: [r[f"pred_{p}"] == r["true_label"] for r in rows] for p in PROMPTS}

    mentah, rincian = {}, {}
    for i, a in enumerate(PROMPTS):
        for b_ in PROMPTS[i + 1:]:
            hanya_a = sum(1 for x, y in zip(benar[a], benar[b_]) if x and not y)
            hanya_b = sum(1 for x, y in zip(benar[a], benar[b_]) if y and not x)
            p = mcnemar_exact(hanya_a, hanya_b)
            kunci = f"{a} vs {b_}"
            mentah[kunci] = p
            rincian[kunci] = {"hanya_menang_A": hanya_a, "hanya_menang_B": hanya_b, "p": round(p, 5)}

    disesuaikan = holm(mentah)
    for kunci in rincian:
        rincian[kunci]["p_holm"] = round(disesuaikan[kunci], 5)
        rincian[kunci]["signifikan_5persen"] = disesuaikan[kunci] < 0.05

    print(f"\n{'='*74}")
    print("1. UJI McNEMAR ANTAR PROMPT (eksak, dua sisi; Holm untuk 6 perbandingan)")
    print(f"{'='*74}")
    print(f"  {'pasangan':<26}{'A>B':>5}{'B>A':>5}{'p':>10}{'p (Holm)':>11}  kesimpulan")
    print(f"  {'-'*70}")
    for kunci, v in rincian.items():
        tanda = "BEDA NYATA" if v["signifikan_5persen"] else "tidak nyata"
        print(f"  {kunci:<26}{v['hanya_menang_A']:>5}{v['hanya_menang_B']:>5}"
              f"{v['p']:>10.4f}{v['p_holm']:>11.4f}  {tanda}")
    return rincian


def analisis_latensi(rows: list[dict], PROMPTS: list[str]) -> dict:
    print(f"\n{'='*74}")
    print("2. LATENSI & PANJANG OUTPUT PER PROMPT")
    print(f"{'='*74}")
    print(f"  {'prompt':<12}{'rata2':>9}{'median':>9}{'total':>10}{'panjang raw':>14}")
    print(f"  {'-'*54}")
    out = {}
    for p in PROMPTS:
        t = [float(r[f"time_{p}"]) for r in rows if r.get(f"time_{p}")]
        panjang = [len(r[f"raw_{p}"]) for r in rows]
        out[p] = {
            "detik_rata2": round(st.mean(t), 2),
            "detik_median": round(st.median(t), 2),
            "detik_total": round(sum(t), 1),
            "panjang_raw_rata2": round(st.mean(panjang)),
        }
        print(f"  {p:<12}{out[p]['detik_rata2']:>8.2f}s{out[p]['detik_median']:>8.2f}s"
              f"{out[p]['detik_total']:>9.0f}s{out[p]['panjang_raw_rata2']:>14}")
    return out


def analisis_stabilitas(rows: list[dict], PROMPTS: list[str]) -> dict:
    """Provider mana yang melayani, dan apakah hasilnya berbeda antar provider."""
    print(f"\n{'='*74}")
    print("3. STABILITAS: PROVIDER & PERBANDINGAN ANTAR-RUN")
    print(f"{'='*74}")

    per_prompt = {}
    print("  Akurasi per provider (deskriptif — routing tidak diacak, bukan eksperimen terkontrol):")
    for p in PROMPTS:
        kolom = f"provider_{p}"
        if not rows[0].get(kolom):
            continue
        ember = defaultdict(lambda: [0, 0])          # provider -> [benar, total]
        for r in rows:
            e = ember[r[kolom] or "(tidak dilaporkan)"]
            e[1] += 1
            e[0] += r[f"pred_{p}"] == r["true_label"]
        # Provider hanya sebanding kalau komposisi kelasnya sebanding. Routing
        # OpenRouter mengikuti urutan pemrosesan, jadi satu provider bisa sama
        # sekali tidak kebagian kelas tertentu — dan angka akurasinya jadi
        # tidak berarti. Kondisi itu ditandai eksplisit, jangan dibaca mentah.
        semua_kelas = {r["true_label"] for r in rows}
        kelas_per_prov = defaultdict(set)
        for r in rows:
            kelas_per_prov[r[kolom] or "(tidak dilaporkan)"].add(r["true_label"])

        per_prompt[p] = {}
        bagian = []
        for prov, (ben, tot) in sorted(ember.items()):
            hilang = sorted(semua_kelas - kelas_per_prov[prov])
            per_prompt[p][prov] = {
                "n": tot, "akurasi": round(ben / tot, 4), "kelas_tidak_terwakili": hilang,
            }
            tanda = f" [TIDAK SEBANDING: tanpa {', '.join(hilang)}]" if hilang else ""
            bagian.append(f"{prov}: {ben/tot*100:.1f}% (n={tot}){tanda}")
        print(f"    {p:<12}" + "\n                ".join(bagian))

    antar_run = {}
    if PREV_CSV.exists():
        lama = {r["file"]: r for r in csv.DictReader(open(PREV_CSV, encoding="utf-8"))}
        cocok = [r for r in rows if r["file"] in lama]
        print(f"\n  Perbandingan dengan run sebelumnya ({len(cocok)} gambar, temperature=0, model sama):")
        print(f"    {'prompt':<12}{'berubah':>9}{'stabil':>10}")
        for p in PROMPTS:
            beda = sum(1 for r in cocok if r[f"pred_{p}"] != lama[r["file"]][f"pred_{p}"])
            antar_run[p] = {"berubah": beda, "dari": len(cocok), "stabil": round(1 - beda / len(cocok), 4)}
            print(f"    {p:<12}{beda:>9}{(1-beda/len(cocok))*100:>9.1f}%")
        print("\n    Catatan: temperature=0 hanya deterministik dalam satu provider.")
        print("    Routing OpenRouter yang berpindah membuat hasil antar-run tidak identik.")
    else:
        print(f"\n  ({PREV_CSV.name} tidak ada — perbandingan antar-run dilewati)")

    return {"per_provider": per_prompt, "antar_run": antar_run}


def _self_check():
    # Nilai acuan dihitung tangan: b=10, c=2 -> n=12, 2*(C(12,0)+C(12,1)+C(12,2))/2^12
    assert abs(mcnemar_exact(10, 2) - 2 * (1 + 12 + 66) / 4096) < 1e-12
    assert mcnemar_exact(0, 0) == 1.0            # tidak ada beda sama sekali
    assert mcnemar_exact(5, 5) == 1.0            # seimbang sempurna
    assert mcnemar_exact(20, 0) < 0.001          # satu arah total -> sangat signifikan

    # Holm: p terkecil dikali m, dan hasilnya tidak boleh menurun setelah diurutkan
    h = holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert abs(h["a"] - 0.03) < 1e-12, h
    assert h["a"] <= h["b"] <= h["c"], h
    print("Self-check statistik lulus.")


def main():
    rows = baca_baris()
    prompts = [p for p in ALL_PROMPTS if f"pred_{p}" in rows[0]]
    print(f"[INFO] {len(rows)} gambar test, {len(prompts)} prompt: {prompts}")

    hasil = {
        "n_samples": len(rows),
        "mcnemar": uji_mcnemar(rows, prompts),
        "latensi": analisis_latensi(rows, prompts),
        "stabilitas": analisis_stabilitas(rows, prompts),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] Analisis disimpan: {OUT_JSON}")


if __name__ == "__main__":
    _self_check()
    main()
