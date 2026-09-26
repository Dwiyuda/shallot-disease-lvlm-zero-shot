"""
Inferensi zero-shot lewat OpenRouter (OpenAI-compatible API).
Default model: Gemini-3.1-Flash-Lite — terpilih dari perbandingan 4 model
(82,5% pada split test, tertinggi sekaligus tercepat). Dijalankan remote,
bukan lokal, dan tetap zero-shot tanpa fine-tuning.

Setup: salin .env.example → .env, isi OPENROUTER_API_KEY.
"""
import os
import io
import time
import base64
from typing import NamedTuple

from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # muat OPENROUTER_API_KEY / OPENROUTER_MODEL dari file .env

DEFAULT_MODEL = "google/gemini-3.1-flash-lite"


class Hasil(NamedTuple):
    """
    Output satu panggilan API beserta jejaknya.

    OpenRouter merutekan permintaan ke provider yang berbeda-beda, dan
    temperature=0 hanya menjamin determinisme di dalam satu provider. Karena
    itu provider dan model yang benar-benar melayani tiap panggilan ikut
    dicatat — tanpa ini eksperimen tidak bisa direplikasi.
    """
    text: str
    finish_reason: str
    provider: str


_client = None


def _get_client() -> OpenAI:
    """Buat client OpenRouter sekali (lazy singleton)."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY belum di-set. "
                "PowerShell: $env:OPENROUTER_API_KEY = \"sk-or-...\""
            )
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _client


def _to_data_url(image: Image.Image) -> str:
    """Encode PIL Image → data URL base64 JPEG untuk dikirim ke API."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def predict(
    image: Image.Image,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    retries: int = 2,
) -> Hasil:
    """
    Inferensi satu gambar zero-shot via OpenRouter. Signature-nya sengaja
    beda dari inference.predict lokal (tak perlu model/processor).
    Kembalikan Hasil — pakai .text untuk parser.parse_output.

    max_tokens hanya BATAS ATAS, bukan target — prompt pendek berhenti sendiri
    dan tidak ikut terkena biayanya. Nilainya harus cukup untuk prompt yang
    minta reasoning bertahap (CoT): pada 256 token, 79 dari 140 output CoT
    terpotong sebelum sampai baris jawaban dan merusak hasil eksperimen.

    Dinaikkan lagi ke 2048 setelah penyaringan model: sebagian model menulis
    narasi panjang (llama-4-maverick) atau memakai token penalaran internal
    (gemini-3.1-pro) sebelum menuliskan baris jawaban. Plafon harus sama untuk
    semua model, kalau tidak yang bertele-tele dihukum bukan karena salah
    mengenali penyakit, melainkan karena kehabisan jatah token.

    ponytail: retry linear-backoff sederhana untuk error jaringan/rate-limit;
    ganti ke exponential + jitter kalau sering kena 429.
    """
    client = _get_client()
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _to_data_url(image)}},
        ],
    }]

    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,  # deterministik — penting untuk reproducibility penelitian
            )
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                # Output kepotong plafon token — jangan sampai lolos diam-diam
                # seperti eksperimen CoT sebelumnya.
                print(f"  [WARN] Output terpotong di {max_tokens} token; naikkan max_tokens.")
            return Hasil(
                text=(choice.message.content or "").strip(),
                finish_reason=choice.finish_reason or "",
                # 'provider' field khusus OpenRouter, di luar skema OpenAI —
                # SDK menaruhnya di model_extra, jadi ambil lewat dua jalur.
                provider=getattr(resp, "provider", None) or (resp.model_extra or {}).get("provider", ""),
            )
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 * (attempt + 1)
            print(f"  [WARN] API error ({e}); retry {attempt+1}/{retries} dalam {wait}s")
            time.sleep(wait)


if __name__ == "__main__":
    # Self-check offline (tanpa jaringan): encoding gambar → data URL valid.
    img = Image.new("RGB", (32, 32), (34, 139, 34))
    url = _to_data_url(img)
    assert url.startswith("data:image/jpeg;base64,"), "prefix data URL salah"
    assert base64.b64decode(url.split(",", 1)[1]), "payload base64 tidak bisa didecode"
    print("Self-check _to_data_url lulus.")
