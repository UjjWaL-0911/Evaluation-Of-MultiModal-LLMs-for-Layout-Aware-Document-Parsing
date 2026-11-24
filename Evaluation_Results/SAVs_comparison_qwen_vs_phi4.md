# Qwen2.5-VL-7B vs Phi-4-multimodal: like-for-like comparison

Both models scored on **the same images**: Part 2A uses the first 25 images per split, Part 2B the first 15 images per split x 3 configs (Qwen's reduced Kaggle run). Phi-4 numbers here are recomputed from the full Phi-4 run (`SAVs_kaggle_phi4/`) restricted to those images; same code, seeds, k=20 and n_shot=20.

## Part 2A: classification (balanced accuracy %, mean ± std over seeds)

| Model | Task | n | shots/class | Zero-shot | SAVs | Majority (acc) | ZS invalid |
|---|---|---|---|---|---|---|---|
| qwen25vl_7b | acquisition | 100 | 20 | 88.7 ± 2.2 | **93.0 ± 2.9** | 50.0 ± 0.0 | 0% |
| qwen25vl_7b | has_math | 50 | 9 | 64.8 ± 1.2 | **64.1 ± 8.7** | 29.4 ± 27.5 | 0% |
| qwen25vl_7b | has_table | 50 | 17 | 98.0 ± 4.0 | **95.3 ± 4.3** | 68.8 ± 0.0 | 0% |
| qwen25vl_7b | merged_cells | 50 | 17 | 66.2 ± 6.4 | **72.0 ± 11.8** | 31.2 ± 0.0 | 0% |
| phi4_mm (same images as Qwen) | acquisition | 100 | 20 | 50.0 ± 1.1 | **90.7 ± 1.7** | 50.0 ± 0.0 | 0% |
| phi4_mm (same images as Qwen) | has_math | 50 | 9 | 69.3 ± 0.9 | **71.0 ± 5.7** | 29.4 ± 27.5 | 0% |
| phi4_mm (same images as Qwen) | has_table | 50 | 17 | 98.0 ± 4.0 | **96.2 ± 4.7** | 68.8 ± 0.0 | 0% |
| phi4_mm (same images as Qwen) | merged_cells | 50 | 17 | 50.0 ± 0.0 | **74.4 ± 5.5** | 31.2 ± 0.0 | 0% |

## Part 2B: SAV routing of parsing configs (mean NEDS/TEDS on held-out half)

| Model | n | Always base | Best fixed | Best per split | **SAV router** | Oracle | Failure-pred bal. acc |
|---|---|---|---|---|---|---|---|
| qwen25vl_7b | 60 | 75.2 ± 1.9 | 73.9 ± 3.7 | 74.6 ± 1.7 | **72.5 ± 1.8** | 76.9 ± 1.7 | 53.6 ± 9.0 |
| phi4_mm (same images as Qwen) | 60 | 57.6 ± 2.5 | 54.4 ± 2.7 | 55.2 ± 2.2 | **54.4 ± 2.7** | 62.9 ± 1.6 | 44.1 ± 2.2 |

## Part 2B: mean score per split and config (same 15 images per split)

| Split | Model | base_1024 | r768 | crop_768 |
|---|---|---|---|---|
| Scanned_Docs | qwen25vl_7b | 0.783 | 0.761 | 0.709 |
| Scanned_Docs | phi4_mm | 0.593 | 0.559 | 0.391 |
| Photoed_Docs | qwen25vl_7b | 0.761 | 0.668 | 0.668 |
| Photoed_Docs | phi4_mm | 0.522 | 0.479 | 0.479 |
| Scanned_Tables | qwen25vl_7b | 0.706 | 0.722 | 0.641 |
| Scanned_Tables | phi4_mm | 0.631 | 0.641 | 0.616 |
| Photoed_Tables | qwen25vl_7b | 0.681 | 0.662 | 0.662 |
| Photoed_Tables | phi4_mm | 0.458 | 0.448 | 0.448 |
