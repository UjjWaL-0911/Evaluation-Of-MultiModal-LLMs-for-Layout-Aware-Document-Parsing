# BTP Part 2: Project Context (read this first)

**Project:** Evaluation of Small and Large Multimodal Models for Layout-Aware Document Parsing
**Team:** Ujjwal Krishna Khanna, Pratham Kala, Anvesh Gupta · **Guide:** Dr. Anukriti Bansal (LNMIIT)
**Repo:** https://github.com/UjjWaL-0911/Evaluation-Of-MultiModal-LLMs-for-Layout-Aware-Document-Parsing

---

## 1. What Part 1 did (already finished, report submitted May 2025)

- **Data:** CC-OCR benchmark (Hugging Face `wulipc/CC-OCR`), **English doc-parsing subset only**:

  | Split | Task | Metric | Images |
  |---|---|---|---|
  | Scanned_Docs | image → LaTeX | NEDS (1 − normalized character edit distance) | 75 |
  | Photoed_Docs | image → LaTeX | NEDS | 75 |
  | Scanned_Tables | image → HTML | TEDS (tree edit distance similarity) | 75 |
  | Photoed_Tables | image → HTML | TEDS | 75 |

  **150 document images + 150 table images. We use nothing else from CC-OCR.**
- **Method:** sent every image to large models through the NVIDIA NIM API (`InferenceScript.ipynb`), scored with the
  official CC-OCR scorer (`doc_parsing_evaluator.py`), and wrote HTML reports with character-level diffs (`Evaluation_Results/<model>/`).
- **Results (NEDS / TEDS, higher is better):**

  | Model | Scan Docs | Photo Docs | Scan Tables | Photo Tables |
  |---|---|---|---|---|
  | Qwen3.5-122B | 0.749 | 0.787 | 0.737 | 0.681 |
  | Llama-4 Maverick | 0.610 | 0.564 | 0.735 | 0.636 |
  | Gemma-3-27B | 0.468 | 0.572 | 0.693 | 0.547 |
  | Llama-3.2-90B | 0.422 | 0.514 | 0.666 | 0.484 |
  | Phi-4-multimodal | 0.267 | 0.472 | 0.523 | 0.500 |
  | Llama-3.2-11B | 0.371 | 0.411 | 0.551 | 0.440 |
- **Key findings:** downsizing images to 768-1024 px *improved* scores and saved ~62% tokens; whitespace cropping helped
  documents (+12 pts) but hurt scanned tables (−8 pts); many low scores were format mismatches (HTML vs LaTeX, header
  flattening), not wrong content.
- **Future work promised in the report:** (1) few-shot prompting, (2) **Sparse Attention Vectors (SAVs)**, this Part 2.

## 2. The paper we implement: Sparse Attention Vectors

Mitra et al., *"Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features"*,
arXiv 2412.00142 (https://arxiv.org/abs/2412.00142; PDFs are not in the git repo). No fine-tuning:

1. Run ~20 labelled examples per class through a **frozen** open model; record the output of **every attention head
   at the last input token**.
2. For each head, average the vectors per class (class centroids) and measure how well "nearest centroid by cosine
   similarity" classifies the examples. **Keep the top 20 heads** (<5% of all heads).
3. Classify a new image by **majority vote** of those 20 heads.

The paper shows this beats zero-shot prompting, few-shot prompting and LoRA on classification tasks.

**Important limits:**
- SAVs needs the model's internals, so it **cannot use APIs** (NIM/Gemini). We run open weights on Colab.
- SAVs outputs **a label from a fixed set**, not text. It **cannot write the LaTeX/HTML itself**, so it cannot
  directly raise NEDS/TEDS. It can only help *indirectly* by making discrete decisions about the parsing (Part 2B).

## 3. What Part 2 does

### Part 2A: Does SAVs work on documents? (reproduction on our data)
Classification tasks built from our 300 images, labels derived automatically from the split name / ground truth:

| Task | Images | Classes (counts) |
|---|---|---|
| `acquisition` | all 300 | scanned 150 / photographed 150 |
| `has_math` | 150 docs | yes 114 / no 36 |
| `has_table` | 150 docs | yes 65 / no 85 |
| `merged_cells` | 150 tables | yes 72 / no 78 |

For each task we compare **SAVs** (20 examples/class to pick heads) against **zero-shot prompting** (ask the model
the same multiple-choice question) and the **majority class**, on the held-out images, averaged over 5 random seeds.
Metric: balanced accuracy (classes are imbalanced).

### Part 2B: Can SAVs improve parsing accuracy? (the actual NEDS/TEDS question)
Each image is parsed under 3 pre-processing configs: `base_1024` (resize longest edge to 1024),
`r768` (768 px), `crop_768` (whitespace crop + 768 px). SAVs, reading the activations of the *parsing prompt*,
predicts **which config to use for each image**. We report mean NEDS/TEDS of:
always-base · best single config · best config per split · **SAV router** · oracle (per-image best, upper bound).
We also test **failure prediction**: can SAVs tell in advance which images will score below 0.5?
The gain is capped by the oracle-vs-best-fixed gap; a small or zero gain is still a valid result to report.

## 4. Models and who runs what

Free Colab = T4 GPU (16 GB) + ~12.7 GB RAM. Only these fit:

| `MODEL_KEY` | Model | How it fits | Owner | Notes |
|---|---|---|---|---|
| `qwen25vl_7b` | Qwen2.5-VL-7B-Instruct | 4-bit | **Ujjwal** | Same family as our best Part 1 model; the paper used Qwen2-VL-7B |
| `phi4_mm` | Phi-4-multimodal-instruct | fp16 | **Ujjwal** | Part 1 model. Needs `transformers==4.48.2` (notebook handles it) |
| `llama32v_11b` | Llama-3.2-11B-Vision-Instruct | 4-bit | **Anvesh** | Part 1 model. **Gated**: accept Meta's license on HF first |
| `gemma3_12b` | gemma-3-12b-it | 4-bit, bf16 compute | **Pratham** | Smaller sibling of Part 1's Gemma-3-27B. **Gated**: accept Google's license on HF |

The other Part 1 models (Gemma-3-27B, Gemma-4-31B, Llama-3.2-90B, Llama-4-Maverick, Qwen3.5-122B) are too large
for a free T4. Qwen2.5-VL-7B and Gemma-3-12B are *smaller models from the same families*; say so in the report.

## 5. How to run (each person, for your own model)

1. Use your **own** Google account (free-tier GPU quota is per account).
2. **Gated models only (Llama, Gemma):** log in to huggingface.co, open the model page, accept the license
   (approval can take minutes to a few hours), then create a *read* token (Settings → Access Tokens).
3. Open `Part2_SAVs_Colab.ipynb` in Colab (File → Open notebook → GitHub tab → paste the repo URL).
   **Runtime → Change runtime type → T4 GPU.**
4. In Colab's 🔑 **Secrets** panel add `HF_TOKEN` = your token, and enable notebook access.
5. In cell 1 set `MODEL_KEY` to your model. Do **not** change anything else unless the team agrees.
6. Run all cells. The notebook clones the repo into `MyDrive/BTP/`, installs packages, loads the model,
   runs a one-image sanity check, then Part 2A, a results preview, then Part 2B.
7. **If Colab disconnects or you hit the GPU limit:** come back later and run all cells again. Finished images
   are skipped automatically (results are saved on your Drive after every image).
8. When done, the last cell writes `MyDrive/SAVs_results_<MODEL_KEY>.zip`; share it with Ujjwal.

The dataset is found automatically: from your Drive if you have the `CC-OCR_Dataset/doc_parsing` folder,
otherwise the 4 TSVs (~68 MB) are downloaded from Hugging Face.

**Rough time per model on a T4** (estimates; check the timings printed in the first minutes):
Part 2A ≈ 1,050 forward passes ≈ 1.5-3 h. Part 2B = 300 full generations ≈ 5-15 h, spread over several sessions.

## 6. Rules so our results are comparable

- Don't edit `savs/tasks.py` (prompts, tasks, configs, thresholds), `doc_parsing_evaluator.py` or the seeds
  on your own. If something must change, change it for **everyone** and re-run.
- Use the same `N_PARSE_PER_SPLIT` (default 25) as everyone else.
- Don't pick results by hand; the analysis uses held-out images and 5 seeds automatically.
- Note anything unusual (OOM, NaNs, many unparseable zero-shot answers, settings you had to change) and send it with the zip.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `401` / `GatedRepoError` | License not accepted yet, or `HF_TOKEN` secret missing / no notebook access |
| `FloatingPointError: Non-finite activations` | Set `DTYPE_OVERRIDE = "bfloat16"` (or `"float32"`) in cell 1 and re-run |
| `CUDA out of memory` | Runtime → Restart session, run again. Part 2B records OOM images as empty and continues |
| "modules offloaded to CPU" warning | GPU was not empty; restart the session |
| No GPU | Runtime → Change runtime type → T4 GPU (or quota exhausted: wait or use another day) |
| Phi-4 import errors | Make sure cell 3 ran in a *fresh* session (it pins old transformers) |

## 8. Code map

| Path | What it is |
|---|---|
| `dataset.py` | Loads the 4 splits (`load_split`, `decode_image`); finds data locally, on Drive, or downloads it |
| `doc_parsing_evaluator.py`, `common.py` | Official CC-OCR NEDS/TEDS scorer (now uses `rapidfuzz` with identical results; new nltk crashes on texts > 2000 chars) |
| `savs/models.py` | One wrapper per model (`load_model(key)`), `vlm.run(image, prompt)` |
| `savs/extract.py` | Hooks every LM attention layer's `o_proj` input to get per-head last-token vectors |
| `savs/sav.py` | The SAVs classifier (centroids, head selection, majority vote) |
| `savs/tasks.py` | **Fixed** shared tasks, prompts and parsing configs |
| `savs/pipeline.py` | Resumable runners; results go to `Evaluation_Results/SAVs/<model_key>/` |
| `savs/analysis.py`, `analyze_savs.py` | Combines all models into `Evaluation_Results/SAVs/summary.md` (CPU only) |
| `Part2_SAVs_Colab.ipynb` | The notebook everyone runs |
| `InferenceScript*.ipynb`, `*Resolution*.ipynb` | Part 1 notebooks (API-based) |
