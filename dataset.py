"""
Loader for the CC-OCR English doc-parsing subset used in this project.

Scope: 4 splits x 75 images = 150 document images + 150 table images.
    doc/doc_scan_eng_75.tsv       -> Scanned_Docs   (LaTeX, NEDS)
    doc/doc_photo_eng_75.tsv      -> Photoed_Docs   (LaTeX, NEDS)
    table/table_scan_eng_75.tsv   -> Scanned_Tables (HTML,  TEDS)
    table/table_photo_eng_75.tsv  -> Photoed_Tables (HTML,  TEDS)

TSV columns: index, image (base64), image_name, question (official CC-OCR prompt),
             answer (ground truth), category, l2-category, split

Works the same locally and on Colab:
    from dataset import load_split, load_all, SPLITS
    df = load_split("Scanned_Docs")

Data location is resolved in this order:
    1. CC_OCR_DIR environment variable (points at the folder containing doc/ and table/)
    2. Colab Google Drive: /content/drive/MyDrive/<...>/CC-OCR_Dataset/doc_parsing
    3. ./CC-OCR_Dataset/doc_parsing next to this file (local copy)
    4. otherwise the 4 TSVs are downloaded from Hugging Face into ./CC-OCR_Dataset
"""
import base64
import functools
import glob
import os
from io import BytesIO

import pandas as pd

SPLITS = {
    "Scanned_Docs":   {"file": "doc/doc_scan_eng_75.tsv",     "task": "doc"},
    "Photoed_Docs":   {"file": "doc/doc_photo_eng_75.tsv",    "task": "doc"},
    "Scanned_Tables": {"file": "table/table_scan_eng_75.tsv",  "task": "table"},
    "Photoed_Tables": {"file": "table/table_photo_eng_75.tsv", "task": "table"},
}
DOC_SPLITS = [k for k, v in SPLITS.items() if v["task"] == "doc"]
TABLE_SPLITS = [k for k, v in SPLITS.items() if v["task"] == "table"]

HF_REPO = "wulipc/CC-OCR"
_HERE = os.path.dirname(os.path.abspath(__file__))


def _is_valid_root(path):
    return path and all(os.path.exists(os.path.join(path, s["file"])) for s in SPLITS.values())


@functools.lru_cache(maxsize=1)
def find_data_root():
    """Return the directory that contains doc/ and table/ TSV folders."""
    candidates = [os.environ.get("CC_OCR_DIR"), os.path.join(_HERE, "CC-OCR_Dataset", "doc_parsing")]
    # Colab: look under My Drive (2 levels deep first; a full recursive scan of Drive is slow)
    if os.path.isdir("/content/drive/MyDrive"):
        for pat in ("*/CC-OCR_Dataset/doc_parsing", "*/*/CC-OCR_Dataset/doc_parsing",
                    "CC-OCR_Dataset/doc_parsing", "*/*/*/CC-OCR_Dataset/doc_parsing"):
            candidates += glob.glob(os.path.join("/content/drive/MyDrive", pat))
    for c in candidates:
        if _is_valid_root(c):
            return c
    # Last resort: download just the 4 TSVs (~68 MB) from Hugging Face next to this file.
    print("CC-OCR TSVs not found locally/Drive; downloading the 4 English splits from Hugging Face...")
    from huggingface_hub import hf_hub_download
    local = os.path.join(_HERE, "CC-OCR_Dataset")
    for s in SPLITS.values():
        hf_hub_download(HF_REPO, f"doc_parsing/{s['file']}", repo_type="dataset", local_dir=local)
    root = os.path.join(local, "doc_parsing")
    if _is_valid_root(root):
        return root
    raise FileNotFoundError("CC-OCR doc_parsing TSVs not found and download failed.")


def load_split(name, n=None):
    """Load one split as a DataFrame (optionally only the first n rows)."""
    if name not in SPLITS:
        raise KeyError(f"Unknown split {name!r}; choose from {list(SPLITS)}")
    df = pd.read_csv(os.path.join(find_data_root(), SPLITS[name]["file"]), sep="\t")
    df["answer"] = df["answer"].astype(str)
    df["split_name"] = name
    df["task"] = SPLITS[name]["task"]
    df["sample_id"] = [f"{name}-{i}" for i in df["index"]]  # matches 'index' used in HTML reports
    return df.head(n) if n else df


def load_all(n=None):
    """Concatenate all 4 splits (300 rows)."""
    return pd.concat([load_split(s, n) for s in SPLITS], ignore_index=True)


def decode_image(b64_str):
    """base64 string -> PIL.Image (RGB)."""
    from PIL import Image
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")


def image_bytes(b64_str):
    """base64 string -> raw bytes (for APIs that take bytes)."""
    return base64.b64decode(b64_str)


if __name__ == "__main__":
    root = find_data_root()
    print("Data root:", root)
    for s in SPLITS:
        df = load_split(s)
        print(f"  {s:15s} {len(df):3d} rows  task={SPLITS[s]['task']}")
