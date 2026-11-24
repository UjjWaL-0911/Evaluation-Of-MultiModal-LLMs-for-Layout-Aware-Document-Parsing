## Part 2A: classification (balanced accuracy %, mean ± std over seeds)

| Model | Task | n | shots/class | Zero-shot | SAVs | Majority (acc) | ZS invalid |
|---|---|---|---|---|---|---|---|
| qwen25vl_7b | acquisition | 100 | 20 | 88.7 ± 2.2 | **93.0 ± 2.9** | 50.0 ± 0.0 | 0% |
| qwen25vl_7b | has_math | 50 | 9 | 64.8 ± 1.2 | **64.1 ± 8.7** | 29.4 ± 27.5 | 0% |
| qwen25vl_7b | has_table | 50 | 17 | 98.0 ± 4.0 | **95.3 ± 4.3** | 68.8 ± 0.0 | 0% |
| qwen25vl_7b | merged_cells | 50 | 17 | 66.2 ± 6.4 | **72.0 ± 11.8** | 31.2 ± 0.0 | 0% |

## Part 2B: SAV routing of parsing configs (mean NEDS/TEDS on held-out half)

| Model | n | Always base | Best fixed | Best per split | **SAV router** | Oracle | Failure-pred bal. acc |
|---|---|---|---|---|---|---|---|
| qwen25vl_7b | 60 | 75.2 ± 1.9 | 73.9 ± 3.7 | 74.6 ± 1.7 | **72.5 ± 1.8** | 76.9 ± 1.7 | 53.6 ± 9.0 |

## Part 2B: mean score per split and config

| Split | base_1024 | r768 | crop_768 |
|---|---|---|---|
| Scanned_Docs | 0.783 (n=15) | 0.761 (n=15) | 0.709 (n=15) |
| Photoed_Docs | 0.761 (n=15) | 0.668 (n=15) | 0.668 (n=15) |
| Scanned_Tables | 0.706 (n=15) | 0.722 (n=15) | 0.641 (n=15) |
| Photoed_Tables | 0.681 (n=15) | 0.662 (n=15) | 0.662 (n=15) |
