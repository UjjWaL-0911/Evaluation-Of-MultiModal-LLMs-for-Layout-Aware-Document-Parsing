"""
Shared, FIXED experiment definitions. Everyone must use these unchanged so results are comparable
across models. Change them only after agreeing as a team.

Classification tasks (labels derived automatically from the split name / ground truth):
    acquisition   all 300 images   scanned vs photographed        (150 / 150)
    has_math      150 doc images   GT contains LaTeX math          (yes 114 / no 36)
    has_table     150 doc images   GT contains a table             (yes 65 / no 85)
    merged_cells  150 table images GT has rowspan/colspan >= 2     (yes 72 / no 78)
    parse         all 300 images   no label: activations of the PARSING prompt, used for
                                   SAV-based routing and failure prediction (Part 2B)

Parsing configurations (Part 2B): how an image is pre-processed before generation.
"""
import re

PARSE_PROMPTS = {  # same minimal reading-order prompts as the Part 1 resolution experiment
    "table": (
        "You are a Table Parsing assistant. Extract the table structure into clean HTML. "
        "Use common tags like <body>, <table>, <tr>, <td>, and <th> tags. "
        "Use rowspan/colspan only if cells are merged. "
        "Maintain the exact reading order of cells (left-to-right, row-by-row). "
        "Do NOT include any CSS or style attributes. Provide only the raw HTML code."
    ),
    "doc": (
        "You are an expert Document Intelligence assistant. Extract text strictly in its logical reading order. "
        "Do NOT include document boilerplate (like \\documentclass). Use minimal LaTeX tags: "
        "\\section{...} for headers and \\textbf{...} for bold. Use LaTeX ($$) ONLY for math formulas. "
        "Output plain text with basic line breaks to reflect the flow. Do not attempt visual layout recreation."
    ),
}

_SUFFIX = "\nAnswer with the option letter only."
_MATH = re.compile(r"(?<!\\)\$|\\\(|\\\[|\\frac|\\begin\{(equation|align)")
_TABLE = re.compile(r"\\begin\{tabular\}|<table")
_MERGED = re.compile(r"(rowspan|colspan)=\"?([2-9]|\d\d)")

TASKS = {
    "acquisition": dict(
        splits=["Scanned_Docs", "Photoed_Docs", "Scanned_Tables", "Photoed_Tables"],
        prompt="Was this image captured with a flatbed scanner or photographed with a camera?\n"
               "(A) scanned\n(B) photographed" + _SUFFIX,
        classes=["scanned", "photographed"],
        label=lambda row: "scanned" if row["split_name"].startswith("Scanned") else "photographed",
    ),
    "has_math": dict(
        splits=["Scanned_Docs", "Photoed_Docs"],
        prompt="Does this document contain mathematical formulas or equations?\n(A) yes\n(B) no" + _SUFFIX,
        classes=["yes", "no"],
        label=lambda row: "yes" if _MATH.search(row["answer"]) else "no",
    ),
    "has_table": dict(
        splits=["Scanned_Docs", "Photoed_Docs"],
        prompt="Does this document page contain a table?\n(A) yes\n(B) no" + _SUFFIX,
        classes=["yes", "no"],
        label=lambda row: "yes" if _TABLE.search(row["answer"]) else "no",
    ),
    "merged_cells": dict(
        splits=["Scanned_Tables", "Photoed_Tables"],
        prompt="Does this table contain merged cells, i.e. cells that span multiple rows or columns?\n"
               "(A) yes\n(B) no" + _SUFFIX,
        classes=["yes", "no"],
        label=lambda row: "yes" if _MERGED.search(row["answer"]) else "no",
    ),
    "parse": dict(
        splits=["Scanned_Docs", "Photoed_Docs", "Scanned_Tables", "Photoed_Tables"],
        prompt=None,                # PARSE_PROMPTS[row task] is used
        classes=None,
        label=None,
    ),
}
CLASSIFICATION_TASKS = [t for t, v in TASKS.items() if v["classes"]]

# Image pre-processing used for ALL SAV activation extraction (fixed).
EXTRACT_MAX_EDGE = 1024

# Part 2B parsing configurations. Order matters: ties in score go to the earlier config.
PARSE_CONFIGS = {
    "base_1024":  dict(max_edge=1024, crop=False),
    "r768":       dict(max_edge=768,  crop=False),
    "crop_768":   dict(max_edge=768,  crop=True),
}
PARSE_MAX_NEW_TOKENS = 2048   # same as Part 1
FAIL_THRESHOLD = 0.5          # base-config NEDS/TEDS below this = "failure"


def task_prompt(task, row):
    return TASKS[task]["prompt"] or PARSE_PROMPTS[row["task"]]


def parse_option(text, classes):
    """Map a zero-shot answer like 'A', '(B)', 'B) photographed' or 'yes' to a class name (or None)."""
    t = text.strip()
    m = re.match(r"^\(?([AB])\b", t, re.I) or re.search(r"\(([AB])\)", t) or re.search(r"\b([AB])\b", t)
    if m:
        return classes["AB".index(m.group(1).upper())]
    low = t.lower()
    hits = [c for c in classes if c in low]
    return hits[0] if len(hits) == 1 else None
