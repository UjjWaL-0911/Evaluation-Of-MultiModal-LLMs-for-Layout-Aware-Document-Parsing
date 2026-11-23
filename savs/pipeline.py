"""
Resumable runners. Everything is written to   Evaluation_Results/SAVs/<model_key>/

    activations/<task>/<sample_id>.pt   {"acts": [L,H,D] fp16, "label": str|None,
                                         "zs_raw": str, "zs_pred": str|None}
    parsing/<config>.jsonl              one line per image: sample_id, split, pred, score, seconds
    run_info.json                       model / library / GPU details

Re-running skips work that already exists, so a Colab disconnect only loses the current image.
"""
import json
import os
import platform
import time

import torch
from tqdm.auto import tqdm

from dataset import decode_image, load_split
from .extract import HeadExtractor
from .preprocess import preprocess
from .tasks import (EXTRACT_MAX_EDGE, PARSE_CONFIGS, PARSE_MAX_NEW_TOKENS, TASKS,
                    parse_option, task_prompt)

RESULTS_ROOT = os.path.join("Evaluation_Results", "SAVs")


def model_dir(model_key, root=RESULTS_ROOT):
    return os.path.join(root, model_key)


def write_run_info(vlm, extractor, root=RESULTS_ROOT):
    import transformers
    info = {
        "model_key": vlm.key, "hf_id": vlm.hf_id, "quant": vlm.spec["quant"],
        "compute_dtype": str(vlm.compute_dtype), "layers": extractor.n_layers, "heads_per_layer": extractor.n_heads,
        "transformers": transformers.__version__, "torch": torch.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(model_dir(vlm.key, root), exist_ok=True)
    with open(os.path.join(model_dir(vlm.key, root), "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def run_extraction(vlm, extractor, task, n_per_split=None, root=RESULTS_ROOT):
    """SAV Step 1 for one task: last-token head activations + zero-shot answer for every image."""
    spec = TASKS[task]
    out_dir = os.path.join(model_dir(vlm.key, root), "activations", task)
    os.makedirs(out_dir, exist_ok=True)
    for split in spec["splits"]:
        df = load_split(split, n_per_split)
        todo = [r for _, r in df.iterrows() if not os.path.exists(os.path.join(out_dir, f"{r['sample_id']}.pt"))]
        for row in tqdm(todo, desc=f"{task} | {split}", total=len(todo)):
            img = preprocess(decode_image(row["image"]), max_edge=EXTRACT_MAX_EDGE)
            extractor.reset()
            # classification tasks: generate a few tokens to get the zero-shot answer too;
            # parse task: 1 token is enough (we only need the prefill activations)
            raw = vlm.run(img, task_prompt(task, row), max_new_tokens=8 if spec["classes"] else 1)
            rec = {"acts": extractor.collect().half(),
                   "label": spec["label"](row) if spec["label"] else None,
                   "zs_raw": raw,
                   "zs_pred": parse_option(raw, spec["classes"]) if spec["classes"] else None}
            torch.save(rec, os.path.join(out_dir, f"{row['sample_id']}.pt"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_parsing(vlm, config, n_per_split=25, max_new_tokens=PARSE_MAX_NEW_TOKENS, root=RESULTS_ROOT,
                splits=("Scanned_Docs", "Photoed_Docs", "Scanned_Tables", "Photoed_Tables")):
    """Part 2B: full document/table parsing under one pre-processing config, scored with NEDS/TEDS."""
    from doc_parsing_evaluator import ParsingEvaluator
    from .tasks import PARSE_PROMPTS
    evaluator = ParsingEvaluator(group_name="SAVs_Part2")
    cfg = PARSE_CONFIGS[config]
    out_dir = os.path.join(model_dir(vlm.key, root), "parsing")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{config}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(l)["sample_id"] for l in f if l.strip()}
    for split in splits:
        df = load_split(split, n_per_split)
        todo = [r for _, r in df.iterrows() if r["sample_id"] not in done]
        for row in tqdm(todo, desc=f"parse {config} | {split}", total=len(todo)):
            img = preprocess(decode_image(row["image"]), max_edge=cfg["max_edge"], crop=cfg["crop"])
            t0 = time.time()
            try:
                pred = vlm.run(img, PARSE_PROMPTS[row["task"]], max_new_tokens=max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                pred = ""
                print(f"OOM on {row['sample_id']} - recorded as empty prediction")
            secs = time.time() - t0
            if row["task"] == "doc":
                score = evaluator.evaluate_single_doc_sample(row["answer"], pred) if pred else 0.0
            else:
                score = evaluator.evaluate_single_table_sample(row["answer"], pred)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": row["sample_id"], "split": split, "task": row["task"],
                                    "config": config, "score": score, "seconds": round(secs, 1),
                                    "pred": pred}, ensure_ascii=False) + "\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def progress(model_key, root=RESULTS_ROOT, n_parse_per_split=25):
    """Print how much of each phase is finished for one model."""
    d = model_dir(model_key, root)
    print(f"== {model_key} ({d})")
    for task, spec in TASKS.items():
        p = os.path.join(d, "activations", task)
        n = len([f for f in os.listdir(p) if f.endswith(".pt")]) if os.path.isdir(p) else 0
        print(f"  extraction {task:13s} {n:4d} / {75 * len(spec['splits'])}")
    for cfg in PARSE_CONFIGS:
        p = os.path.join(d, "parsing", f"{cfg}.jsonl")
        n = sum(1 for l in open(p, encoding="utf-8") if l.strip()) if os.path.exists(p) else 0
        print(f"  parsing    {cfg:13s} {n:4d} / {4 * n_parse_per_split}")
