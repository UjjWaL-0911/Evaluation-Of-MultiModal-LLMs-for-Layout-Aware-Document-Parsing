"""
CPU-only analysis of saved activations / parsing results.

Part 2A  classification: SAVs vs zero-shot prompting vs majority class, on held-out images,
         averaged over seeds (support = n_shot per class, test = everything else).
Part 2B  routing: SAVs picks a parsing config per image from the 'parse' activations; compared
         with always-base, best-fixed-config, best-config-per-split and the oracle.
         failure prediction: SAVs predicts whether the base config scores < FAIL_THRESHOLD.
"""
import glob
import json
import os
from collections import Counter

import numpy as np
import torch

from .pipeline import RESULTS_ROOT, model_dir
from .sav import SAVClassifier, stratified_support
from .tasks import CLASSIFICATION_TASKS, FAIL_THRESHOLD, PARSE_CONFIGS

SEEDS = [0, 1, 2, 3, 4]


def load_task(model_key, task, root=RESULTS_ROOT):
    files = sorted(glob.glob(os.path.join(model_dir(model_key, root), "activations", task, "*.pt")))
    recs = [torch.load(f, weights_only=False) for f in files]
    ids = [os.path.basename(f)[:-3] for f in files]
    if not recs:
        return None
    return {"ids": ids, "acts": torch.stack([r["acts"] for r in recs]).float(),
            "labels": [r["label"] for r in recs], "zs": [r["zs_pred"] for r in recs]}


def _bal_acc(y, p):
    classes = sorted(set(y))
    return float(np.mean([np.mean([pp == c for yy, pp in zip(y, p) if yy == c]) for c in classes]))


def _n_shot(labels, n_shot, min_test=5):
    smallest = min(Counter(labels).values())
    return max(1, min(n_shot, smallest - min_test))


def classification(model_key, task, n_shot=20, k=20, root=RESULTS_ROOT):
    d = load_task(model_key, task, root)
    if d is None or len(set(d["labels"])) < 2:
        return None
    shots = _n_shot(d["labels"], n_shot)
    res = {"sav_acc": [], "sav_bal": [], "zs_acc": [], "zs_bal": [], "maj_acc": []}
    for s in SEEDS:
        sup, test = stratified_support(d["labels"], shots, np.random.default_rng(s))
        clf = SAVClassifier(k).fit(d["acts"][sup], [d["labels"][i] for i in sup])
        y = [d["labels"][i] for i in test]
        p = clf.predict(d["acts"][test])
        zs = [d["zs"][i] for i in test]
        maj = Counter(d["labels"][i] for i in sup).most_common(1)[0][0]
        res["sav_acc"].append(np.mean([a == b for a, b in zip(y, p)]))
        res["sav_bal"].append(_bal_acc(y, p))
        res["zs_acc"].append(np.mean([a == b for a, b in zip(y, zs)]))
        res["zs_bal"].append(_bal_acc(y, zs))
        res["maj_acc"].append(np.mean([a == maj for a in y]))
    out = {k_: (float(np.mean(v)), float(np.std(v))) for k_, v in res.items()}
    out.update(n=len(d["labels"]), shots_per_class=shots, k=k, test_n=len(test),
               zs_invalid_rate=float(np.mean([z is None for z in d["zs"]])),
               class_counts=dict(Counter(d["labels"])))
    return out


def load_parsing(model_key, root=RESULTS_ROOT):
    scores = {}
    for cfg in PARSE_CONFIGS:
        p = os.path.join(model_dir(model_key, root), "parsing", f"{cfg}.jsonl")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                scores[cfg] = {r["sample_id"]: r["score"] for r in map(json.loads, filter(str.strip, f))}
    return scores


def routing(model_key, k=20, root=RESULTS_ROOT):
    scores = load_parsing(model_key, root)
    if len(scores) < 2:
        return None
    acts = load_task(model_key, "parse", root)
    if acts is None:
        return None
    cfgs = list(scores)
    common = [i for i in acts["ids"] if all(i in scores[c] for c in cfgs)]
    if len(common) < 20:
        return None
    pos = {i: n for n, i in enumerate(acts["ids"])}
    X = acts["acts"][[pos[i] for i in common]]
    S = np.array([[scores[c][i] for c in cfgs] for i in common])          # [N, C]
    split_of = np.array([i.rsplit("-", 1)[0] for i in common])
    best = [cfgs[j] for j in S.argmax(1)]                                  # ties -> earlier config

    res = {k_: [] for k_ in ["always_base", "best_fixed", "best_per_split", "sav_router", "oracle",
                             "fail_sav_bal", "fail_maj_acc"]}
    fail = ["fail" if s < FAIL_THRESHOLD else "ok" for s in S[:, 0]]
    for s in SEEDS:
        rng = np.random.default_rng(s)
        sup = np.sort(rng.choice(len(common), size=len(common) // 2, replace=False))
        test = np.setdiff1d(np.arange(len(common)), sup)
        St = S[test]
        res["always_base"].append(St[:, 0].mean())
        res["oracle"].append(St.max(1).mean())
        res["best_fixed"].append(St[:, S[sup].mean(0).argmax()].mean())
        per_split = {sp: S[sup][split_of[sup] == sp].mean(0).argmax() if (split_of[sup] == sp).any() else 0
                     for sp in set(split_of)}
        res["best_per_split"].append(np.mean([S[i, per_split[split_of[i]]] for i in test]))
        sup_best = [best[i] for i in sup]
        if len(set(sup_best)) > 1:
            choice = SAVClassifier(k).fit(X[sup], sup_best).predict(X[test])
            res["sav_router"].append(np.mean([S[i, cfgs.index(c)] for i, c in zip(test, choice)]))
        else:
            res["sav_router"].append(St[:, cfgs.index(sup_best[0])].mean())
        sup_fail = [fail[i] for i in sup]
        if len(set(sup_fail)) > 1:
            p = SAVClassifier(k).fit(X[sup], sup_fail).predict(X[test])
            res["fail_sav_bal"].append(_bal_acc([fail[i] for i in test], p))
        maj = Counter(sup_fail).most_common(1)[0][0]
        res["fail_maj_acc"].append(np.mean([fail[i] == maj for i in test]))
    out = {k_: (float(np.mean(v)), float(np.std(v))) for k_, v in res.items() if v}
    out.update(n=len(common), configs=cfgs, best_config_counts=dict(Counter(best)),
               fail_counts=dict(Counter(fail)),
               mean_score_per_config={c: float(S[:, j].mean()) for j, c in enumerate(cfgs)})
    return out


def analyze(model_keys, root=RESULTS_ROOT, n_shot=20, k=20):
    report = {}
    for m in model_keys:
        report[m] = {"classification": {t: classification(m, t, n_shot, k, root) for t in CLASSIFICATION_TASKS},
                     "routing": routing(m, k, root)}
    return report


def to_markdown(report):
    f = lambda v: f"{v[0]*100:.1f} ± {v[1]*100:.1f}"
    lines = ["## Part 2A: classification (balanced accuracy %, mean ± std over seeds)", "",
             "| Model | Task | n | shots/class | Zero-shot | SAVs | Majority (acc) | ZS invalid |",
             "|---|---|---|---|---|---|---|---|"]
    for m, r in report.items():
        for t, c in r["classification"].items():
            if c:
                lines.append(f"| {m} | {t} | {c['n']} | {c['shots_per_class']} | {f(c['zs_bal'])} | "
                             f"**{f(c['sav_bal'])}** | {f(c['maj_acc'])} | {c['zs_invalid_rate']*100:.0f}% |")
    lines += ["", "## Part 2B: SAV routing of parsing configs (mean NEDS/TEDS on held-out half)", "",
              "| Model | n | Always base | Best fixed | Best per split | **SAV router** | Oracle | Failure-pred bal. acc |",
              "|---|---|---|---|---|---|---|---|"]
    for m, r in report.items():
        x = r["routing"]
        if x:
            fp = f(x["fail_sav_bal"]) if "fail_sav_bal" in x else "n/a"
            lines.append(f"| {m} | {x['n']} | {f(x['always_base'])} | {f(x['best_fixed'])} | "
                         f"{f(x['best_per_split'])} | **{f(x['sav_router'])}** | {f(x['oracle'])} | {fp} |")
    return "\n".join(lines)
