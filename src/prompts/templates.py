"""
Template prompt zero-shot untuk identifikasi penyakit daun bawang merah.
Lima varian untuk eksperimen prompt engineering. Empat pertama adalah
perbandingan utama; enhanced_v2 adalah revisi berbasis analisis kesalahan.
"""

_CLASSES = (
    "1. Purple Blotch - oval purple/brown spots with yellow halo at leaf edges\n"
    "2. Botrytis Leaf Blight - small white spots with brown edges, dead tissue\n"
    "3. Downy Mildew - gray/purple fungal growth on leaf surface, yellowing\n"
    "4. Rust - orange-red oval pustules with powdery appearance\n"
    "5. Healthy Leaf - uniform green color, no lesions or abnormal discoloration"
)

# ── Prompt 1: Direct (baseline) ──────────────────────────────────────────────
# Instruksi singkat, langsung minta label + justifikasi.
PROMPT_DIRECT = (
    "You are an expert plant pathologist specializing in shallot (Allium cepa) diseases.\n"
    "Analyze this shallot leaf image and identify the disease class.\n\n"
    "Available classes:\n"
    f"{_CLASSES}\n\n"
    "Respond with ONLY the class name from the list above, followed by a brief justification "
    "of the visual features that support your diagnosis."
)

# ── Prompt 2: Chain-of-Thought ───────────────────────────────────────────────
# Minta model deskripsikan visual dulu sebelum klasifikasi.
# Tujuan: mendorong reasoning lebih dalam sebelum memberi label.
PROMPT_COT = (
    "You are an expert plant pathologist specializing in shallot (Allium cepa) diseases.\n\n"
    "Analyze this shallot leaf image step by step:\n"
    "Step 1 - Describe the visual features you observe (color, texture, spots, patterns).\n"
    "Step 2 - Match those features to the disease classes below.\n"
    "Step 3 - State your final diagnosis.\n\n"
    "Available classes:\n"
    f"{_CLASSES}\n\n"
    "End your response with:\n"
    "DIAGNOSIS: [class name from the list above]"
)

# ── Prompt 3: Structured Output ──────────────────────────────────────────────
# Paksa format output yang ketat: label + confidence + alasan.
# Tujuan: memudahkan parsing dan mendapat estimasi keyakinan model.
PROMPT_STRUCTURED = (
    "You are an expert plant pathologist specializing in shallot (Allium cepa) diseases.\n"
    "Classify this shallot leaf image using EXACTLY this format:\n\n"
    "CLASS: [one of: Purple Blotch | Botrytis Leaf Blight | Downy Mildew | Rust | Healthy Leaf]\n"
    "CONFIDENCE: [High | Medium | Low]\n"
    "REASON: [one sentence describing the key visual evidence]\n\n"
    "Disease reference:\n"
    f"{_CLASSES}"
)

# ── Prompt 4: Enhanced Visual Detail ────────────────────────────────────────
# Deskripsi visual sangat spesifik per kelas + peringatan "jangan keliru".
# Fokus pada kelas yang sering gagal: Rust dan Downy Mildew.
_CLASSES_DETAIL = """1. Purple Blotch (Alternaria porri)
   - Oval to elongated spots with PURPLE or dark brown CENTER
   - Distinctive YELLOW or water-soaked HALO surrounding each spot
   - Spots may merge into large lesions; edges are well-defined

2. Botrytis Leaf Blight (Botrytis squamosa)
   - Many SMALL (2-4 mm) WHITE to pale spots scattered across leaf
   - Each spot has a TAN or BROWN border, clearly defined edges
   - Dead/necrotic tissue at spot centers; no fuzzy growth

3. Downy Mildew (Peronospora destructor)
   - DIFFUSE yellowing or pale green patches — NOT discrete spots
   - GRAY to VIOLET FUZZY coating on the leaf surface (like dust or mold)
   - Affected area feels damp; lesion boundaries are indistinct/blurry
   - Do NOT confuse with: Purple Blotch (no fuzzy coating, has sharp oval spots)

4. Rust (Puccinia allii)
   - RAISED, ROUND to oval PUSTULES in ORANGE, brick-red, or rust-brown color
   - Pustules appear as small bumps on leaf; look powdery or dusty if rubbed
   - Scattered distribution across leaf; surrounding tissue may turn yellow
   - Do NOT confuse with: Purple Blotch (purple color, flat spots, yellow halo)
   - Do NOT confuse with: Botrytis (white spots, not orange bumps)

5. Healthy Leaf
   - Uniform BRIGHT GREEN color throughout
   - Smooth surface; no spots, bumps, discoloration, or lesions of any kind"""

PROMPT_ENHANCED = (
    "You are an expert plant pathologist specializing in shallot (Allium cepa) diseases.\n"
    "Carefully examine this shallot leaf image and identify the disease.\n\n"
    "DISEASE REFERENCE GUIDE:\n"
    f"{_CLASSES_DETAIL}\n\n"
    "Instructions:\n"
    "- Look for the MOST DISTINCTIVE visual feature in the image\n"
    "- Pay special attention to: spot COLOR, spot SHAPE, surface TEXTURE, and lesion EDGES\n"
    "- Use the 'Do NOT confuse with' warnings above to eliminate similar-looking classes\n\n"
    "Respond using EXACTLY this format:\n"
    "CLASS: [Purple Blotch | Botrytis Leaf Blight | Downy Mildew | Rust | Healthy Leaf]\n"
    "CONFIDENCE: [High | Medium | Low]\n"
    "KEY FEATURE: [the single most distinctive visual feature you observed]"
)

# ── Prompt 5: Enhanced v2 — perbaikan berbasis analisis kesalahan ───────────
# Tiga kelemahan enhanced yang terlihat pada split test (n=80):
#   1. Model terlalu mudah menjawab "Healthy Leaf" (precision hanya 60,0%);
#      7 dari 16 gambar Downy Mildew divonis sehat.
#   2. Downy Mildew tidak terlihat oleh model (recall 37,5%) karena gejalanya
#      difus — tidak membentuk bercak tegas seperti kelas lain.
#   3. Field CONFIDENCE tidak informatif: 121 dari 124 jawaban "High".
#
# Tiga penangkalnya, berurutan: pemeriksaan difus DIDAHULUKAN sebelum mencari
# bercak, gerbang eksplisit sebelum boleh menjawab sehat, dan CONFIDENCE
# diganti dugaan kedua yang benar-benar membawa informasi.
#
# Nama field sengaja SECOND_GUESS, bukan "ALTERNATIVE CLASS" — parser mencari
# pola 'class:' sehingga field kedua tidak boleh memuat kata itu.
PROMPT_ENHANCED_V2 = (
    "You are an expert plant pathologist specializing in shallot (Allium cepa) diseases.\n\n"
    "EXAMINATION PROCEDURE - follow in this order:\n"
    "1. Scan the WHOLE leaf for DIFFUSE changes first: gradual yellowing, pale or\n"
    "   washed-out patches, a dull or dusty film over the surface. These are easy to\n"
    "   overlook and are the primary sign of Downy Mildew.\n"
    "2. Only after that, look for DISCRETE marks: spots, pustules, bordered lesions.\n"
    "3. Judge each mark by COLOR, SHAPE, EDGE sharpness, and surface TEXTURE\n"
    "   (flat or raised, dry or fuzzy).\n\n"
    "DISEASE REFERENCE GUIDE:\n"
    f"{_CLASSES_DETAIL}\n\n"
    "HEALTHY GATE - answer 'Healthy Leaf' only if ALL THREE are true:\n"
    "- No diffuse yellowing or pale gradient anywhere on the leaf\n"
    "- No fuzzy, dusty, or dull film on the leaf surface\n"
    "- No spots, pustules, or dead tissue, however small or faint\n"
    "If you are unsure about any one of them, choose the disease the leaf most\n"
    "resembles instead of Healthy Leaf.\n\n"
    "Respond using EXACTLY this format:\n"
    "CLASS: [Purple Blotch | Botrytis Leaf Blight | Downy Mildew | Rust | Healthy Leaf]\n"
    "SECOND_GUESS: [next most likely class from the list, or 'none']\n"
    "KEY FEATURE: [the single most distinctive feature you actually see in THIS image]"
)

# Daftar semua prompt untuk eksperimen perbandingan
ALL_PROMPTS = {
    "direct":      PROMPT_DIRECT,
    "cot":         PROMPT_COT,
    "structured":  PROMPT_STRUCTURED,
    "enhanced":    PROMPT_ENHANCED,
    "enhanced_v2": PROMPT_ENHANCED_V2,
}

# Prompt yang MEWAJIBKAN field jawaban eksplisit (DIAGNOSIS:/CLASS:). Untuk
# prompt ini parser dilarang menebak dari teks bebas — output tanpa field itu
# berarti model gagal mengikuti instruksi, dan harus dihitung sebagai UNKNOWN,
# bukan ditebak dari kata yang kebetulan muncul di teks reasoning.
# 'direct' tidak masuk daftar: formatnya memang jawaban bebas.
STRICT_PROMPTS = {"cot", "structured", "enhanced", "enhanced_v2"}
