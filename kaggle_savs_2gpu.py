"""
BTP Part 2 (SAVs): full run of one model on Kaggle "GPU T4 x2", same code/settings as Part2_SAVs_Colab.ipynb.

Both T4s work in parallel: each loads its own copy of the model and handles half of the splits
(GPU0: Scanned_*, GPU1: Photoed_*). Part 2A activations go straight to /kaggle/working/SAVs/<model>/;
Part 2B parsing is written per worker and merged into the usual parsing/<config>.jsonl at the end.
Resumable: attach an earlier version's output as input and finished images are skipped.

Run on Kaggle (Accelerator "GPU T4 x2", Internet on) as a two-cell notebook:
    cell 1:  %%writefile /tmp/run.py   followed by this file's contents
    cell 2:  !python -u /tmp/run.py
Set MODEL_KEY and the image counts below first. Results: /kaggle/working/SAVs (+ SAVs_results_<model>.zip);
the Parsing diff pages are then built locally with visualize_savs_parsing.py.
Gated models (gemma3_12b, llama32v_11b) also need a Hugging Face token with the licence accepted.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

MODEL_KEY = "qwen25vl_7b"
N_EXTRACT_PER_SPLIT = 25             # Part 2A: first 25 of the 75 images per split (team default: all 75)
N_PARSE_PER_SPLIT = 15               # Part 2B: first 15 images per split (team setting: 25)
TIME_BUDGET_HOURS = 10.5             # stop starting new work after this (Kaggle kills a session at 12 h)
REPO_URL = "https://github.com/UjjWaL-0911/Evaluation-Of-MultiModal-LLMs-for-Layout-Aware-Document-Parsing.git"

WORK = "/kaggle/working"
CODE = "/tmp/btp_code"
ROOT = f"{WORK}/SAVs"                # final results, same layout as the Colab runs
SPLITS = ["Scanned_Docs", "Photoed_Docs", "Scanned_Tables", "Photoed_Tables"]
GROUPS = [["Scanned_Docs", "Scanned_Tables"], ["Photoed_Docs", "Photoed_Tables"]]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(*cmd):
    subprocess.run(list(cmd), check=True)


def worker_root(i):                  # in the output, so nothing is lost if the session is killed before merging
    return f"{WORK}/parse_w{i}"


# ---------------------------------------------------------------- results bookkeeping (pure python)
def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def restore_results(input_root, dst_model_dir, key=MODEL_KEY):
    """Copy an earlier run's results (attached as input) into dst. .pt files: never overwrite.
    .jsonl (parsing, possibly split across workers of a run that was cut off): union by sample_id."""
    srcs = [d for d in glob.glob(f"{input_root}/**/{key}", recursive=True)
            if os.path.isdir(os.path.join(d, "activations")) or os.path.isdir(os.path.join(d, "parsing"))]
    for src in srcs:
        n = 0
        for f in glob.glob(os.path.join(src, "**", "*"), recursive=True):
            if not os.path.isfile(f):
                continue
            dst = os.path.join(dst_model_dir, os.path.relpath(f, src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if f.endswith(".jsonl"):
                rows = {r["sample_id"]: r for r in read_jsonl(dst)}
                new = [r for r in read_jsonl(f) if r["sample_id"] not in rows]
                with open(dst, "a", encoding="utf-8") as out:
                    for r in new:
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += len(new)
            elif not os.path.exists(dst):
                shutil.copy2(f, dst)
                n += 1
        log(f"restored {n} files/rows from {src}")


def seed_worker_parsing(model_dir, wroot, splits, configs, key=MODEL_KEY):
    """Give a worker the already-finished parsing lines of its splits, so run_parsing skips them."""
    out = os.path.join(wroot, key, "parsing")
    os.makedirs(out, exist_ok=True)
    for cfg in configs:
        done = [r for r in read_jsonl(os.path.join(model_dir, "parsing", f"{cfg}.jsonl")) if r["split"] in splits]
        with open(os.path.join(out, f"{cfg}.jsonl"), "w", encoding="utf-8") as f:
            for r in done:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def merge_parsing(model_dir, wroots, configs, key=MODEL_KEY):
    """Merge worker jsonls into model_dir/parsing/<cfg>.jsonl (one line per sample_id, split order kept)."""
    os.makedirs(os.path.join(model_dir, "parsing"), exist_ok=True)
    for cfg in configs:
        rows = {}
        for p in [os.path.join(model_dir, "parsing", f"{cfg}.jsonl")] + \
                 [os.path.join(w, key, "parsing", f"{cfg}.jsonl") for w in wroots]:
            for r in read_jsonl(p):
                rows[r["sample_id"]] = r
        order = sorted(rows.values(), key=lambda r: (SPLITS.index(r["split"]), int(r["sample_id"].rsplit("-", 1)[1])))
        with open(os.path.join(model_dir, "parsing", f"{cfg}.jsonl"), "w", encoding="utf-8") as f:
            for r in order:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def progress_line(model_dir, wroots, configs, key=MODEL_KEY):
    acts = {os.path.basename(d): len(glob.glob(f"{d}/*.pt")) for d in sorted(glob.glob(f"{model_dir}/activations/*"))}
    parse = {c: sum(len(read_jsonl(os.path.join(w, key, "parsing", f"{c}.jsonl"))) for w in wroots) for c in configs}
    return f"activations {acts} | parsing {parse}"


# ---------------------------------------------------------------- one GPU worker
def worker(i, n_workers, deadline):
    sys.path.insert(0, CODE)
    os.chdir(CODE)
    import torch
    import savs.pipeline as P
    from savs.extract import HeadExtractor
    from savs.models import load_model
    from savs.tasks import PARSE_CONFIGS, TASKS

    my = GROUPS[i] if n_workers == 2 else SPLITS
    P.TASKS = {t: dict(v, splits=[s for s in v["splits"] if s in my]) for t, v in TASKS.items()}
    t0 = time.time()
    vlm = load_model(MODEL_KEY)
    ext = HeadExtractor(vlm)
    log(f"[w{i}] {torch.cuda.get_device_name(0)} | loaded in {time.time()-t0:.0f}s | {ext.describe()} | "
        f"{torch.cuda.memory_allocated()/1e9:.1f} GB | splits {my}")
    if i == 0:
        log(f"[w{i}] run_info", P.write_run_info(vlm, ext, root=ROOT))
    open(f"/tmp/w{i}.loaded", "w").close()

    for task in TASKS:                                            # Part 2A
        if time.time() > deadline:
            log(f"[w{i}] time budget used up during Part 2A"); return
        t0 = time.time()
        P.run_extraction(vlm, ext, task, n_per_split=N_EXTRACT_PER_SPLIT, root=ROOT)
        log(f"[w{i}] 2A {task} done in {(time.time()-t0)/60:.1f} min")

    for cfg in PARSE_CONFIGS:                                     # Part 2B
        for split in my:
            if time.time() > deadline:
                log(f"[w{i}] time budget used up during Part 2B"); return
            t0 = time.time()
            P.run_parsing(vlm, cfg, n_per_split=N_PARSE_PER_SPLIT, root=worker_root(i), splits=(split,))
            log(f"[w{i}] 2B {cfg} {split} done in {(time.time()-t0)/60:.1f} min")
    log(f"[w{i}] finished")


# ---------------------------------------------------------------- driver
def main():
    t_start = time.time()
    deadline = t_start + TIME_BUDGET_HOURS * 3600
    os.environ["USE_TF"] = "0"
    os.environ["TQDM_DISABLE"] = "1"
    import socket
    try:
        socket.getaddrinfo("github.com", 443)
    except OSError:
        raise SystemExit("NO INTERNET in this Kaggle session (needs a phone-verified account and Internet = On). "
                         "Needed for git, pip and the Hugging Face model download.")

    if not os.path.isdir(CODE):
        sh("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, CODE)
        sh("git", "-C", CODE, "sparse-checkout", "add", "savs")
    log("code:", subprocess.run(["git", "-C", CODE, "log", "-1", "--oneline"], capture_output=True, text=True).stdout.strip())
    sh(sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.50", "accelerate", "bitsandbytes")
    sh(sys.executable, "-m", "pip", "install", "-q", "rapidfuzz", "apted", "distance", "lxml", "tabulate", "nltk")

    import torch
    n_gpu = torch.cuda.device_count()
    log("GPUs:", [torch.cuda.get_device_name(g) for g in range(n_gpu)])
    if n_gpu == 0:
        raise SystemExit("No GPU: push with --accelerator NvidiaTeslaT4")
    n_workers = 2 if n_gpu >= 2 else 1

    sys.path.insert(0, CODE)
    os.chdir(CODE)
    from huggingface_hub import snapshot_download
    from dataset import find_data_root
    from savs.models import MODELS
    from savs.tasks import PARSE_CONFIGS
    configs = list(PARSE_CONFIGS)
    log("data:", find_data_root())                                # download once, before workers start
    snapshot_download(MODELS[MODEL_KEY]["hf_id"])
    log("model downloaded")

    model_dir = os.path.join(ROOT, MODEL_KEY)
    restore_results("/kaggle/input", model_dir)
    wroots = [worker_root(i) for i in range(n_workers)]
    for i in range(n_workers):
        seed_worker_parsing(model_dir, wroots[i], GROUPS[i] if n_workers == 2 else SPLITS, configs)

    os.makedirs(f"{WORK}/logs", exist_ok=True)
    procs = []
    for i in range(n_workers):
        if i > 0:                                                 # stagger model loading (CPU RAM)
            t0 = time.time()
            while not os.path.exists(f"/tmp/w{i-1}.loaded") and procs[-1].poll() is None and time.time() - t0 < 1200:
                time.sleep(10)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        logf = open(f"{WORK}/logs/worker{i}.log", "w")
        procs.append(subprocess.Popen([sys.executable, "-u", __file__, "--worker", str(i), str(n_workers), str(deadline)],
                                      env=env, stdout=logf, stderr=subprocess.STDOUT))
        log(f"started worker {i}")

    while any(p.poll() is None for p in procs):
        time.sleep(600)
        log(progress_line(model_dir, wroots, configs))
    for i, p in enumerate(procs):
        log(f"worker {i} exit code {p.returncode}; last lines of its log:")
        with open(f"{WORK}/logs/worker{i}.log", encoding="utf-8", errors="replace") as f:
            print("".join(f.readlines()[-25:]), flush=True)

    merge_parsing(model_dir, wroots, configs)
    log("merged:", progress_line(model_dir, [ROOT], configs))
    from savs.analysis import analyze, load_parsing, to_markdown
    report = analyze([MODEL_KEY], root=ROOT)
    md = to_markdown(report)
    scores = load_parsing(MODEL_KEY, root=ROOT)                   # per split x config mean NEDS/TEDS
    lines = ["", "## Part 2B: mean score per split and config", "", "| Split | " + " | ".join(scores) + " |",
             "|---|" + "---|" * len(scores)]
    for s in SPLITS:
        cells = []
        for c, d in scores.items():
            v = [x for k, x in d.items() if k.rsplit("-", 1)[0] == s]
            cells.append(f"{sum(v)/len(v):.3f} (n={len(v)})" if v else "-")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    md += "\n" + "\n".join(lines)
    with open(os.path.join(ROOT, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(ROOT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md, flush=True)
    zip_path = shutil.make_archive(f"{WORK}/SAVs_results_{MODEL_KEY}", "zip", root_dir=ROOT, base_dir=MODEL_KEY)
    log(f"done in {(time.time()-t_start)/3600:.2f} h | {zip_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
    else:
        main()
