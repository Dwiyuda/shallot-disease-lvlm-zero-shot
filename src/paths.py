"""
Path terpusat proyek. Di-anchor ke root lewat __file__ (bukan cwd), supaya
skrip bisa dijalankan dari folder mana pun tanpa path relatif yang pecah.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # .../shallot-disease-lvlm
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
SPLIT_CSV = DATA_DIR / "split.csv"
DUPLIKAT_CSV = DATA_DIR / "duplikat.csv"
RESULTS_DIR = ROOT / "results"


def dir_hasil(model_id: str) -> Path:
    """
    Folder hasil satu model: results/<nama-model>/

    Tiap model diuji pada data yang sama, jadi hasilnya harus terpisah supaya
    bisa dibandingkan tanpa saling menimpa. Awalan vendor dibuang karena nama
    modelnya sudah unik ('qwen/qwen3-vl-8b-instruct' -> 'qwen3-vl-8b-instruct').
    """
    return RESULTS_DIR / model_id.split("/")[-1]
