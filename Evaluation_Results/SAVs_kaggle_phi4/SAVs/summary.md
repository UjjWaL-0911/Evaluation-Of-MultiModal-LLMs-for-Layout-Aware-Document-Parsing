## Part 2A: classification (balanced accuracy %, mean ± std over seeds)

| Model | Task | n | shots/class | Zero-shot | SAVs | Majority (acc) | ZS invalid |
|---|---|---|---|---|---|---|---|
| phi4_mm | acquisition | 300 | 20 | 49.8 ± 0.4 | **93.0 ± 1.4** | 50.0 ± 0.0 | 0% |
| phi4_mm | has_math | 150 | 20 | 77.1 ± 0.7 | **75.6 ± 1.9** | 14.5 ± 0.0 | 0% |
| phi4_mm | has_table | 150 | 20 | 93.1 ± 0.5 | **92.6 ± 0.7** | 59.1 ± 0.0 | 0% |
| phi4_mm | merged_cells | 150 | 20 | 49.9 ± 0.4 | **71.4 ± 2.9** | 50.5 ± 2.7 | 0% |

## Part 2B: SAV routing of parsing configs (mean NEDS/TEDS on held-out half)

| Model | n | Always base | Best fixed | Best per split | **SAV router** | Oracle | Failure-pred bal. acc |
|---|---|---|---|---|---|---|---|
| phi4_mm | 100 | 52.2 ± 2.3 | 52.2 ± 2.3 | 51.9 ± 2.7 | **51.8 ± 1.8** | 58.2 ± 2.3 | 60.2 ± 6.2 |

## Part 2B: mean score per split and config

| Split | base_1024 | r768 | crop_768 |
|---|---|---|---|
| Scanned_Docs | 0.555 (n=25) | 0.532 (n=25) | 0.398 (n=25) |
| Photoed_Docs | 0.599 (n=25) | 0.540 (n=25) | 0.540 (n=25) |
| Scanned_Tables | 0.559 (n=25) | 0.570 (n=25) | 0.535 (n=25) |
| Photoed_Tables | 0.490 (n=25) | 0.501 (n=25) | 0.501 (n=25) |
