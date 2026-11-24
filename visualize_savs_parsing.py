"""Bucketed diff visualizer (same layout as the Part 1 report, from InferenceScript.ipynb) for the
Part 2B parsing outputs written by savs.pipeline.run_parsing.

Usage: python visualize_savs_parsing.py <model_results_dir> [out_dir]
  e.g. python visualize_savs_parsing.py Evaluation_Results/SAVs_kaggle_v4/SAVs/qwen25vl_7b

Writes <out_dir>/<config>/bucketed_visual_<split>.html plus an index.html (default out_dir:
<model_results_dir>/visual). Each card shows the image exactly as the model saw it under that config.
"""
import base64
import difflib
import html
import json
import os
import re
import sys
from io import BytesIO

from common import convert_to_halfwidth
from dataset import decode_image, load_split
from doc_parsing_evaluator import extract_and_clean_tables, patterns
from savs.preprocess import preprocess
from savs.tasks import PARSE_CONFIGS

SPLITS = ["Scanned_Docs", "Photoed_Docs", "Scanned_Tables", "Photoed_Tables"]
BUCKETS = ["76-100%", "51-75%", "26-50%", "0-25%"]


def generate_inline_diff(gt_text, pred_text):
    """Generates inline HTML diffs focusing ONLY on content mismatches."""
    matcher = difflib.SequenceMatcher(None, gt_text or "", pred_text or "", autojunk=False)
    gt_diff, pred_diff = [], []
    red = '<mark style="background-color: #ffcccc; color: #900;">{}</mark>'
    green = '<mark style="background-color: #ccffcc; color: #090;">{}</mark>'
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        gt_sub, pred_sub = gt_text[i1:i2], pred_text[j1:j2]
        if tag == "equal" or (tag == "replace" and gt_sub.strip() == pred_sub.strip() and gt_sub.strip()):
            gt_diff.append(html.escape(gt_sub))
            pred_diff.append(html.escape(pred_sub))
            continue
        if tag in ("replace", "delete"):
            gt_diff.append(red.format(html.escape(gt_sub)) if gt_sub.strip() else html.escape(gt_sub))
        if tag in ("replace", "insert"):
            pred_diff.append(green.format(html.escape(pred_sub)) if pred_sub.strip() else html.escape(pred_sub))
    return "".join(gt_diff), "".join(pred_diff)


def clean_for_visualizer(text, task_type="doc"):
    """Applies the same normalization as the evaluator to ensure diffs are fair."""
    if task_type == "table":
        text = extract_and_clean_tables(text)
    else:
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        text = re.sub(r"```(latex|html|text)?", "", text).replace("```", "")
    return convert_to_halfwidth(text).strip()


def bucket_of(score):
    return BUCKETS[0] if score > 0.75 else BUCKETS[1] if score > 0.50 else BUCKETS[2] if score > 0.25 else BUCKETS[3]


HEAD = """<html><head><meta charset="utf-8"><title>{title}</title>
<style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background-color: #f0f2f5; }}
    .bucket-header {{ background: #2c3e50; color: white; padding: 15px; border-radius: 5px; margin-top: 40px; }}
    .container {{ display: flex; flex-direction: column; gap: 20px; border: 1px solid #ccc; margin-bottom: 40px; padding: 25px; border-radius: 8px; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    .col-img {{ width: 100%; border-bottom: 2px solid #eee; padding-bottom: 15px; }}
    .col-img img {{ max-height: 400px; max-width: 100%; object-fit: contain; display: block; margin-top: 10px; border: 1px solid #ddd; }}
    .col-code {{ width: 100%; background: #272822; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; box-sizing: border-box; }}
    .col-render {{ width: 100%; background: #ffffff; color: #333; padding: 15px; border: 1px solid #ccc; border-radius: 5px; overflow-x: auto; box-sizing: border-box; }}
    pre {{ white-space: pre-wrap; font-family: 'Consolas', monospace; font-size: 13px; margin: 0; }}
    .score-tag {{ font-weight: bold; color: white; background: #007bff; padding: 6px 12px; border-radius: 15px; font-size: 15px; display: inline-block; margin-bottom: 10px; }}
    .other {{ color: #555; font-size: 13px; margin-left: 10px; }}
    table {{ border-collapse: collapse; width: max-content; font-size: 13px; background: white; }}
    th, td {{ border: 1px solid #444; padding: 6px 10px; text-align: left; }}
    details {{ margin-top: 15px; cursor: pointer; color: #0056b3; font-weight: bold; padding: 10px; background: #f8f9fa; border: 1px solid #ddd; border-radius: 5px; }}
    mark {{ padding: 2px; border-radius: 3px; font-weight: bold; }}
</style></head><body><h1>{title}</h1><p>{subtitle}</p>
"""


def card(item, task_type):
    gt_diff, pred_diff = generate_inline_diff(item["gt_vis"], item["pred_vis"])
    if task_type == "table":
        gt_display = f'<div class="col-render"><strong style="font-size: 16px; color: #d9534f;">GROUND TRUTH HTML</strong><br>{item["gt"]}<details><summary>Click to view Raw HTML Diffs</summary><pre style="color:#333; background:#eee; padding:10px;">{gt_diff}</pre></details></div>'
        pred_display = f'<div class="col-render"><strong style="font-size: 16px; color: #5cb85c;">PREDICTION HTML</strong><br>{item["pred"]}<details><summary>Click to view Raw HTML Diffs</summary><pre style="color:#333; background:#eee; padding:10px;">{pred_diff}</pre></details></div>'
    else:
        gt_display = f'<div class="col-code"><strong style="color: #ae81ff; font-size: 16px;">GT (RAW TEXT)</strong><pre>{gt_diff}</pre></div>'
        pred_display = f'<div class="col-code" style="background: #1e1e1e;"><strong style="color: #66d9ef; font-size: 16px;">PRED (RAW TEXT)</strong><pre>{pred_diff}</pre></div>'
    return f"""
    <div class="container">
        <div class="col-img">
            <span class="score-tag">Score: {item['score']:.4f}</span><span class="other">{item['others']}</span>
            <h3 style="margin: 5px 0;">Image #{item['index']}</h3>
            <img src="data:image/jpeg;base64,{item['image_b64']}" />
        </div>
        {gt_display}
        {pred_display}
    </div>"""


def to_jpeg_b64(img):
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main(model_dir, out_dir=None):
    out_dir = out_dir or os.path.join(model_dir, "visual")
    model = os.path.basename(os.path.normpath(model_dir))
    preds = {}
    for config in PARSE_CONFIGS:
        path = os.path.join(model_dir, "parsing", f"{config}.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                preds[config] = {r["sample_id"]: r for r in map(json.loads, f)}

    summary = {}   # (config, split) -> (mean, n)
    for split in SPLITS:
        df = load_split(split)
        rows = {r["sample_id"]: r for _, r in df.iterrows()}
        for config, recs in preds.items():
            items = []
            for sid, rec in recs.items():
                if rec["split"] != split:
                    continue
                row = rows[sid]
                cfg = PARSE_CONFIGS[config]
                img = preprocess(decode_image(row["image"]), max_edge=cfg["max_edge"], crop=cfg["crop"])
                gt, pred, task = str(row["answer"]), rec["pred"] or "", rec["task"]
                others = " · ".join(f"{c}: {preds[c][sid]['score']:.4f}" for c in preds if c != config and sid in preds[c])
                items.append(dict(image_b64=to_jpeg_b64(img), gt=gt, pred=pred, index=sid, score=rec["score"],
                                  gt_vis=clean_for_visualizer(gt, task), pred_vis=clean_for_visualizer(pred, task),
                                  others=f"({others})" if others else ""))
            if not items:
                continue
            items.sort(key=lambda it: -it["score"])
            summary[(config, split)] = (sum(it["score"] for it in items) / len(items), len(items))
            os.makedirs(os.path.join(out_dir, config), exist_ok=True)
            title = f"Bucketed Report - {split} ({model}, {config})"
            subtitle = (f"{len(items)} images · mean score {summary[(config, split)][0]:.4f} · "
                        f"image shown as the model saw it (max edge {cfg['max_edge']}px"
                        f"{', whitespace cropped' if cfg['crop'] else ''})")
            with open(os.path.join(out_dir, config, f"bucketed_visual_{split}.html"), "w", encoding="utf-8") as f:
                f.write(HEAD.format(title=html.escape(title), subtitle=html.escape(subtitle)))
                for b in BUCKETS:
                    in_bucket = [it for it in items if bucket_of(it["score"]) == b]
                    f.write(f"<h2 class='bucket-header'>Bucket: {b} ({len(in_bucket)} items)</h2>")
                    f.writelines(card(it, "doc" if "Docs" in split else "table") for it in in_bucket)
                f.write("</body></html>")
            print(f"wrote {config}/bucketed_visual_{split}.html ({len(items)} images)")

    # index page linking every report, with the mean score per split x config
    configs = [c for c in PARSE_CONFIGS if c in preds]
    cells = "".join(f"<th>{c}</th>" for c in configs)
    body = "".join(
        f"<tr><td>{s}</td>" + "".join(
            f'<td><a href="{c}/bucketed_visual_{s}.html">{summary[(c, s)][0]:.4f}</a> ({summary[(c, s)][1]})</td>'
            if (c, s) in summary else "<td>-</td>" for c in configs) + "</tr>"
        for s in SPLITS)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(HEAD.format(title=html.escape(f"Parsing diff reports - {model}"),
                            subtitle="Mean NEDS (docs) / TEDS (tables) per split and config; click a score to open its report.")
                + f"<table><tr><th>split</th>{cells}</tr>{body}</table></body></html>")
    print(f"wrote {os.path.join(out_dir, 'index.html')}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
