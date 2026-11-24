## Part 2A: classification (balanced accuracy %, mean ± std over seeds)

| Model | Task | n | shots/class | Zero-shot | SAVs | Majority (acc) | ZS invalid |
|---|---|---|---|---|---|---|---|
| llama32v_11b | acquisition | 300 | 20 | 78.0 ± 1.3 | **95.0 ± 1.9** | 50.0 ± 0.0 | 0% |
| llama32v_11b | has_math | 150 | 20 | 72.0 ± 1.9 | **75.3 ± 1.7** | 14.5 ± 0.0 | 0% |
| llama32v_11b | has_table | 150 | 20 | 89.8 ± 1.0 | **92.0 ± 0.9** | 59.1 ± 0.0 | 0% |
| llama32v_11b | merged_cells | 150 | 20 | 51.0 ± 0.0 | **66.4 ± 4.9** | 50.5 ± 2.7 | 0% |

## Part 2B: SAV routing of parsing configs (mean NEDS/TEDS on held-out half)

| Model | n | Always base | Best fixed | Best per split | **SAV router** | Oracle | Failure-pred bal. acc |
|---|---|---|---|---|---|---|---|
| llama32v_11b | 100 | 45.7 ± 1.0 | 44.2 ± 1.3 | 45.7 ± 1.0 | **44.7 ± 0.9** | 54.3 ± 1.5 | 48.7 ± 5.1 |

## Part 2B: mean score per split and config

| Split | base_1024 | r768 | crop_768 |
|---|---|---|---|
| Scanned_Docs | 0.543 (n=25) | 0.531 (n=25) | 0.507 (n=25) |
| Photoed_Docs | 0.421 (n=25) | 0.492 (n=25) | 0.492 (n=25) |
| Scanned_Tables | 0.489 (n=25) | 0.459 (n=25) | 0.457 (n=25) |
| Photoed_Tables | 0.448 (n=25) | 0.436 (n=25) | 0.436 (n=25) |
