"""
Parser output teks model → label kelas baku.
"""

# Kata kunci per kelas untuk fuzzy matching
CLASS_KEYWORDS = {
    "Purple Blotch":       ["purple blotch", "alternaria", "blotch", "purple"],
    "Botrytis Leaf Blight":["botrytis", "leaf blight", "blight", "squamosa"],
    "Downy Mildew":        ["downy mildew", "downy", "mildew", "peronospora"],
    "Rust":                ["rust", "puccinia", "pustule"],
    "Healthy Leaf":        ["healthy", "normal", "no disease", "no lesion", "no sign"],
}

VALID_CLASSES = list(CLASS_KEYWORDS.keys())


def _extract_class_field(text: str) -> str | None:
    """
    Ekstrak label dari baris jawaban eksplisit: 'CLASS:' (structured/enhanced)
    atau 'DIAGNOSIS:' (chain-of-thought). Ambil kemunculan TERAKHIR agar jawaban
    final menang atas kelas yang cuma disebut di teks reasoning.
    Contoh: 'CLASS: Rust' → 'Rust' ; '...DIAGNOSIS: Downy Mildew' → 'Downy Mildew'
    """
    import re
    matches = re.findall(r"(?:class|diagnosis)\s*:\s*(.+)", text, re.IGNORECASE)
    if not matches:
        return None
    candidate = matches[-1].strip().rstrip(".,;").strip()
    for label in VALID_CLASSES:
        if label.lower() in candidate.lower():
            return label
    return None


def parse_output(raw_text: str, strict: bool = False) -> str:
    """
    Ekstrak label kelas dari teks bebas model.

    Strategi (prioritas):
    1. Ekstrak dari field 'CLASS:'/'DIAGNOSIS:' jika ada (structured/enhanced/cot)
    2. Exact match label dalam teks
    3. Keyword scan (fallback untuk output parafrase)
    Kembalikan "UNKNOWN" jika tidak ada yang cocok.

    strict=True menghentikan proses di prioritas 1: dipakai untuk prompt yang
    mewajibkan field jawaban (lihat templates.STRICT_PROMPTS). Tanpa ini,
    output yang terpotong sebelum baris jawaban akan ditebak dari kata di teks
    reasoning — sumber 23 prediksi "benar" palsu pada eksperimen CoT pertama.
    """
    text = raw_text.lower().strip()

    # Prioritas 1: format structured (CLASS: ...)
    from_field = _extract_class_field(text)
    if from_field:
        return from_field
    if strict:
        return "UNKNOWN"

    # Prioritas 2: exact match label
    for label in VALID_CLASSES:
        if label.lower() in text:
            return label

    # Prioritas 3: keyword scan
    for label, keywords in CLASS_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return label

    return "UNKNOWN"


if __name__ == "__main__":
    # Self-check sederhana
    tests = [
        ("Purple Blotch\nThe leaf shows alternaria spots.", "Purple Blotch"),
        ("This appears to be Rust disease with pustules.", "Rust"),
        ("Healthy Leaf - no signs of disease.", "Healthy Leaf"),
        ("I cannot determine the disease.", "UNKNOWN"),
        # Format structured/enhanced
        ("CLASS: Rust\nCONFIDENCE: High\nKEY FEATURE: orange pustules", "Rust"),
        ("CLASS: Downy Mildew\nCONFIDENCE: Medium\nKEY FEATURE: gray fuzzy coating", "Downy Mildew"),
        ("CLASS: Healthy Leaf\nCONFIDENCE: High\nKEY FEATURE: uniform green", "Healthy Leaf"),
        # Format chain-of-thought: reasoning menyebut kelas lain, DIAGNOSIS harus menang
        ("Step 1 - The leaf shows small white spots, not purple blotches.\n"
         "Step 2 - Matches Botrytis.\nDIAGNOSIS: Botrytis Leaf Blight", "Botrytis Leaf Blight"),
        ("Step 1 - gray fuzzy coating, could resemble purple discoloration.\n"
         "DIAGNOSIS: Downy Mildew", "Downy Mildew"),
    ]
    # Mode strict: tanpa field jawaban → UNKNOWN, tidak boleh menebak.
    tests_strict = [
        ("CLASS: Rust\nCONFIDENCE: High", "Rust"),
        ("Step 1 - The leaf has purple spots with a yellow halo.\n"
         "Step 2 - Comparing against Purple Blotch and", "UNKNOWN"),   # kepotong, dulu → Purple Blotch
        ("The leaf shows orange pustules typical of rust.", "UNKNOWN"),  # tak ikut format
    ]

    all_pass = True
    for raw, expected in tests:
        result = parse_output(raw)
        status = "OK" if result == expected else "X "
        if result != expected:
            all_pass = False
        print(f"{status}  '{raw[:40]}...' -> {result} (expected: {expected})")
    for raw, expected in tests_strict:
        result = parse_output(raw, strict=True)
        status = "OK" if result == expected else "X "
        if result != expected:
            all_pass = False
        print(f"{status}  [strict] '{raw[:32]}...' -> {result} (expected: {expected})")
    print("Semua test lulus!" if all_pass else "Ada test gagal.")
