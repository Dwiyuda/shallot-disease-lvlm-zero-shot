"""
Hitung metrik evaluasi dari hasil prediksi model.
"""
from collections import defaultdict
from src.utils.parser import VALID_CLASSES


def compute_metrics(true_labels: list[str], pred_labels: list[str]) -> dict:
    """
    Hitung accuracy, precision, recall, F1 macro dari list label.
    Kembalikan dict berisi semua metrik + confusion matrix.
    """
    classes = VALID_CLASSES
    n = len(true_labels)

    # Confusion matrix: cm[true][pred]
    cm = defaultdict(lambda: defaultdict(int))
    for t, p in zip(true_labels, pred_labels):
        cm[t][p] += 1

    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / n if n else 0.0

    per_class = {}
    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in classes if other != cls)
        # fn = seluruh baris kelas ini dikurangi yang benar — termasuk prediksi
        # UNKNOWN. Kalau hanya menjumlah kelas valid, gambar yang gagal
        # diklasifikasi hilang dari hitungan dan recall jadi terlalu tinggi.
        fn = sum(cm[cls].values()) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    macro_precision = sum(v["precision"] for v in per_class.values()) / len(classes)
    macro_recall    = sum(v["recall"]    for v in per_class.values()) / len(classes)
    macro_f1        = sum(v["f1"]        for v in per_class.values()) / len(classes)

    return {
        "accuracy":        round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall":    round(macro_recall, 4),
        "macro_f1":        round(macro_f1, 4),
        "per_class":       {k: {m: round(v2, 4) for m, v2 in v.items()} for k, v in per_class.items()},
        "confusion_matrix": {t: dict(cm[t]) for t in classes},
        "n_samples":       n,
        "n_unknown":       sum(1 for p in pred_labels if p == "UNKNOWN"),
    }


def print_report(metrics: dict, prompt_name: str = ""):
    header = f"=== {prompt_name} ===" if prompt_name else "=== HASIL EVALUASI ==="
    print(f"\n{header}")
    print(f"Accuracy : {metrics['accuracy']*100:.1f}%")
    print(f"Macro F1 : {metrics['macro_f1']*100:.1f}%")
    print(f"Macro P  : {metrics['macro_precision']*100:.1f}%  |  Macro R: {metrics['macro_recall']*100:.1f}%")
    print(f"\n{'Kelas':<26} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}")
    print("-" * 50)
    for cls, v in metrics["per_class"].items():
        print(f"{cls:<26} {v['precision']*100:>5.1f}% {v['recall']*100:>5.1f}% {v['f1']*100:>5.1f}% {v['support']:>5}")
    print("-" * 50)

    n_unk = metrics.get("n_unknown", 0)
    if n_unk:
        print(f"Gagal diklasifikasi (UNKNOWN): {n_unk} dari {metrics['n_samples']}")

    print("\nConfusion Matrix (baris=true, kolom=pred):")
    classes = list(metrics["per_class"].keys())
    # Kolom UNK ikut ditampilkan supaya tiap baris berjumlah sama dengan support.
    kolom = classes + ["UNKNOWN"]
    short = ["PB", "BLB", "DM", "Rust", "HL", "UNK"]
    print(f"{'':>5}" + "".join(f"{s:>6}" for s in short))
    for cls, s in zip(classes, short):
        row = metrics["confusion_matrix"].get(cls, {})
        vals = "".join(f"{row.get(c, 0):>6}" for c in kolom)
        print(f"{s:>5}{vals}")


def _self_check():
    """Prediksi UNKNOWN harus dihitung sebagai kegagalan, bukan diabaikan."""
    trues = ["Rust"] * 4
    m = compute_metrics(trues, ["Rust", "Rust", "UNKNOWN", "UNKNOWN"])
    assert m["n_unknown"] == 2, m["n_unknown"]
    assert m["per_class"]["Rust"]["support"] == 4, "UNKNOWN hilang dari support"
    assert m["per_class"]["Rust"]["recall"] == 0.5, m["per_class"]["Rust"]["recall"]
    assert m["accuracy"] == 0.5, m["accuracy"]

    # Precision tidak boleh terpengaruh: UNKNOWN bukan false positive kelas mana pun
    assert m["per_class"]["Rust"]["precision"] == 1.0, m["per_class"]["Rust"]["precision"]
    print("Self-check metrics lulus.")


if __name__ == "__main__":
    _self_check()
