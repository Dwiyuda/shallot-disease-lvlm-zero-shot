# Shallot Leaf Disease Identification — Zero-Shot LVLM

Can a Large Vision-Language Model diagnose shallot (*Allium cepa* var. *aggregatum*) leaf
diseases **without any training or fine-tuning**? This project answers that with a controlled
prompt-engineering experiment: the models are called as-is, and only the prompt design and the
model choice vary.

**Classes (5):** Purple Blotch · Botrytis Leaf Blight · Downy Mildew · Rust · Healthy Leaf
**Evaluation set:** 110 unique images, 22 per class (duplicates removed)
**Protocol:** dev split of 30 images to choose prompts, held-out test split of 80 for final numbers

## Results — 4 models, held-out test split

| Model | Best prompt | Accuracy | Macro-F1 | 95% CI |
|---|---|---:|---:|---|
| **`google/gemini-3.1-flash-lite`** | enhanced_v2 | **82.5%** | 82.6% | 72.7–89.3% |
| `qwen/qwen3-vl-235b-a22b-instruct` | enhanced_v2 | 76.2% | 76.4% | 65.9–84.2% |
| `meta-llama/llama-4-maverick` | enhanced | 72.5% | 72.5% | 61.9–81.1% |
| `qwen/qwen3-vl-8b-instruct` | enhanced | 67.5% | 66.5% | 56.6–76.8% |

Exact McNemar test with Holm correction: Gemini is statistically better than the baseline model
(p = 0.0452).

**Main finding — prompt sensitivity depends on model strength.** Across five prompts, accuracy
moved 5.0 points for Gemini and Llama 4, but 16.2–18.8 points for the two Qwen models. Prompt
engineering compensates for weaker models more than it amplifies strong ones.

**Downy Mildew** was the hardest class. Switching to Gemini brought the number of diseased
leaves wrongly called healthy down to **0 of 64** (from 10 of 64 with the baseline).

Average latency per image: 1.4–2.0 s via OpenRouter.

## Prompt variants

Five prompt templates in `src/prompts/templates.py`:

| Name | Technique |
|---|---|
| `direct` | Zero-shot + persona + class list |
| `cot` | Zero-shot chain-of-thought (3 steps) |
| `structured` | Constraint / output-format prompting |
| `enhanced` | Persona + detailed per-class visual cues + negative prompting |
| `enhanced_v2` | Enhanced + diffuse-symptom check order + healthy gate + second guess |

`enhanced_v2` came from error analysis of `enhanced`: the model said "Healthy Leaf" too easily
and missed the diffuse symptoms of Downy Mildew. All calls use temperature 0 for reproducibility.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add your OpenRouter API key
python run_openrouter.py      # runs the experiment; results go to results/<model>/
python analisis_statistik.py  # McNemar + Holm, confidence intervals
python grafik_perbandingan.py # comparison charts in results/perbandingan/
```

## Repo layout

| Path | Content |
|---|---|
| `src/` | Prompt templates, API inference, preprocessing, evaluation and plotting |
| `data/samples/` | The 110 evaluation images |
| `results/` | Per-model predictions, metrics and figures, plus cross-model comparison |
| `dedup_report.py`, `siapkan_dataset.py` | Duplicate detection and dataset preparation |

A detailed write-up in Indonesian is in [README.id.md](README.id.md).

## Stack

Python · OpenRouter API (Gemini, Qwen3-VL, Llama 4) · Matplotlib · Seaborn

---

Built by **Dwi Yuda** · [dwiyuda.is-a.dev](https://dwiyuda.is-a.dev)
