"""
Combine everyone's SAV results (CPU only, runs locally or in Colab).

    python analyze_savs.py                       # all models found in Evaluation_Results/SAVs
    python analyze_savs.py qwen25vl_7b gemma3_12b
Writes Evaluation_Results/SAVs/summary.json and summary.md.
"""
import json
import os
import sys

from savs.analysis import analyze, to_markdown
from savs.pipeline import RESULTS_ROOT

if __name__ == "__main__":
    keys = sys.argv[1:] or sorted(d for d in os.listdir(RESULTS_ROOT)
                                  if os.path.isdir(os.path.join(RESULTS_ROOT, d, "activations")))
    report = analyze(keys)
    with open(os.path.join(RESULTS_ROOT, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    md = to_markdown(report)
    with open(os.path.join(RESULTS_ROOT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
