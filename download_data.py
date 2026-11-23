"""Download only the 4 English doc/table parsing TSVs (150 docs + 150 tables) from CC-OCR."""
from huggingface_hub import hf_hub_download

from dataset import HF_REPO, SPLITS

for split in SPLITS.values():
    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=f"doc_parsing/{split['file']}",
        local_dir="./CC-OCR_Dataset",
    )
    print("ok", path)

print("Download complete! Check the 'CC-OCR_Dataset/doc_parsing' folder.")
