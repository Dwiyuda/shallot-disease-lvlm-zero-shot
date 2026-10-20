"""
Visualisasi hasil evaluasi.
Baca results/<nama-model>/metrics.json → hasilkan PNG:
  - confusion matrix per prompt
  - bar chart perbandingan accuracy & macro-F1 antar prompt

Jalankan: python -m src.evaluation.visualizer
          OPENROUTER_MODEL=qwen/qwen3-vl-32b-instruct python -m src.evaluation.visualizer

Figur disimpan ke results/<nama-model>/figures/ dan memakai angka split TEST.
"""
import os
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # tanpa GUI, langsung simpan file
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.parser import VALID_CLASSES
from src.model.inference_api import DEFAULT_MODEL
from src.paths import dir_hasil

SHORT = ["PB", "BLB", "DM", "Rust", "HL"]  # label ringkas sesuai urutan VALID_CLASSES


def plot_confusion(cm_dict: dict, title: str, out_path: Path):
    """Gambar confusion matrix (baris=aktual, kolom=prediksi) sebagai heatmap."""
    # Kolom UNKNOWN hanya ditampilkan bila memang ada gambar yang gagal
    # diklasifikasi — supaya tiap baris berjumlah sama dengan support.
    ada_unknown = any(row.get("UNKNOWN") for row in cm_dict.values())
    kolom = VALID_CLASSES + (["UNKNOWN"] if ada_unknown else [])
    kolom_pendek = SHORT + (["UNK"] if ada_unknown else [])
    matrix = [[cm_dict.get(t, {}).get(p, 0) for p in kolom] for t in VALID_CLASSES]

    plt.figure(figsize=(6.6 if ada_unknown else 6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=kolom_pendek, yticklabels=SHORT, square=True,
        linewidths=0.5, linecolor="lightgray",
    )
    plt.title(title)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  tersimpan: {out_path}")


def plot_accuracy_comparison(per_prompt: dict, out_path: Path, n_samples: int | None = None,
                             model: str = ""):
    """Bar chart accuracy & macro-F1 antar prompt."""
    names = list(per_prompt.keys())
    acc = [per_prompt[n]["accuracy"] * 100 for n in names]
    f1 = [per_prompt[n]["macro_f1"] * 100 for n in names]

    x = range(len(names))
    w = 0.38
    plt.figure(figsize=(7, 4.5))
    plt.bar([i - w/2 for i in x], acc, w, label="Accuracy", color="#4C72B0")
    plt.bar([i + w/2 for i in x], f1, w, label="Macro-F1", color="#DD8452")
    for i, (a, f) in enumerate(zip(acc, f1)):
        plt.text(i - w/2, a + 1, f"{a:.1f}", ha="center", fontsize=8)
        plt.text(i + w/2, f + 1, f"{f:.1f}", ha="center", fontsize=8)

    plt.xticks(list(x), names)
    plt.ylabel("Persen (%)")
    plt.ylim(0, 100)
    nama_model = model.split("/")[-1] if model else "model"
    judul = f"Perbandingan Prompt — {nama_model} (zero-shot)"
    plt.title(f"{judul}, n={n_samples}" if n_samples else judul)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  tersimpan: {out_path}")


def main(metrics_json: str, figures_dir: Path):
    data = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    # Figur untuk penelitian memakai angka TEST (held-out), bukan gabungan —
    # dev sudah terpakai untuk memilih prompt sehingga angkanya optimistis.
    bagian = data.get("split", {}).get("test")
    per_prompt = bagian["per_prompt"] if bagian else data["per_prompt"]
    n = bagian["n_samples"] if bagian else data.get("n_samples")
    model = data.get("model", "")
    print(f"[INFO] Model: {model} | {len(per_prompt)} prompt | n={n}"
          f"{' (test/held-out)' if bagian else ''}\n")

    for name, m in per_prompt.items():
        acc = m["accuracy"] * 100
        plot_confusion(
            m["confusion_matrix"],
            f"Confusion Matrix — {name} (acc {acc:.1f}%)",
            figures_dir / f"cm_{name}.png",
        )
    plot_accuracy_comparison(per_prompt, figures_dir / "prompt_comparison.png", n, model)
    print(f"\n[INFO] Semua figur di: {figures_dir}")


if __name__ == "__main__":
    hasil = dir_hasil(os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    main(sys.argv[1] if len(sys.argv) > 1 else str(hasil / "metrics.json"),
         hasil / "figures")
