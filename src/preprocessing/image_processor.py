"""
Load dan validasi gambar sebelum dikirim ke model.
Qwen2-VL processor menangani resize/normalisasi, modul ini hanya pastikan
gambar bisa dibaca dan dalam format RGB.
"""
import hashlib
from pathlib import Path
from PIL import Image


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(image_path: str) -> Image.Image:
    """
    Buka gambar dari path, konversi ke RGB.
    Raise ValueError jika format tidak didukung atau file rusak.
    """
    path = Path(image_path)

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Format tidak didukung: {path.suffix}. Gunakan: {SUPPORTED_FORMATS}")

    try:
        image = Image.open(path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Gagal membuka gambar '{image_path}': {e}")

    return image


def _file_hash(path: Path) -> str:
    """Hash isi file (MD5). Dipakai untuk mendeteksi gambar byte-identik."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def get_image_paths(folder: str, unique: bool = True) -> list[Path]:
    """
    Ambil semua path gambar dari sebuah folder (rekursif), terurut.

    unique=True (default) membuang gambar yang isinya byte-identik dan hanya
    menyisakan salinan pertama (urutan nama file) — data uji tidak boleh
    memuat gambar yang sama dua kali. Pakai unique=False untuk mengambil
    seluruh file apa adanya, mis. saat menyusun laporan duplikat.

    ponytail: deteksi berbasis hash isi file — hanya menangkap duplikat
    PERSIS. Duplikat hasil augmentasi (rotasi/flip/re-encode) tidak tertangkap;
    naikkan ke perceptual hash kalau dataset ternyata hasil augmentasi.
    """
    folder_path = Path(folder)
    paths = sorted(
        p for p in folder_path.rglob("*")
        if p.suffix.lower() in SUPPORTED_FORMATS
    )
    if not unique:
        return paths

    pertama_per_hash: dict[str, Path] = {}
    for p in paths:
        pertama_per_hash.setdefault(_file_hash(p), p)
    return sorted(pertama_per_hash.values())


def find_duplicate_groups(folder: str) -> list[list[Path]]:
    """
    Kelompokkan gambar yang isinya byte-identik.
    Kembalikan hanya grup beranggota >1; anggota pertama tiap grup adalah
    salinan yang dipertahankan oleh get_image_paths().
    """
    per_hash: dict[str, list[Path]] = {}
    for p in get_image_paths(folder, unique=False):
        per_hash.setdefault(_file_hash(p), []).append(p)
    return sorted((g for g in per_hash.values() if len(g) > 1), key=lambda g: g[0])


if __name__ == "__main__":
    # Self-check dedup: 3 file, dua di antaranya byte-identik.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        hijau = Image.new("RGB", (16, 16), (34, 139, 34))
        hijau.save(d / "a.jpg")
        (d / "b.jpg").write_bytes((d / "a.jpg").read_bytes())   # duplikat persis dari a
        Image.new("RGB", (16, 16), (128, 0, 128)).save(d / "c.jpg")
        (d / "catatan.txt").write_text("bukan gambar")

        semua = get_image_paths(str(d), unique=False)
        unik = get_image_paths(str(d))
        grup = find_duplicate_groups(str(d))

        assert [p.name for p in semua] == ["a.jpg", "b.jpg", "c.jpg"], semua
        assert [p.name for p in unik] == ["a.jpg", "c.jpg"], unik      # b.jpg dibuang
        assert len(grup) == 1 and [p.name for p in grup[0]] == ["a.jpg", "b.jpg"], grup
    print("Self-check dedup lulus.")
