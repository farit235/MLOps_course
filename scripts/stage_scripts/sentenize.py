import sys
from pathlib import Path

root = str(Path(__file__).parent.parent.parent)
sys.path.append(root)

import json
import os

from dvc.api import params_show

from src.data.load_dataset import load_all_sents

if __name__ == "__main__":
    params = params_show()

    splits_path = params["paths"]["splits_path"]
    txt_path = params["paths"]["txt_dir"]
    min_sent_words = params["training"]["min_sent_words"]
    with open(splits_path) as f:
        splits = json.load(f)
    train_filenames = splits["train"]
    test_filenames = splits["test"]

    train_sents = load_all_sents(
        [os.path.join(txt_path, filename) for filename in train_filenames],
        min_sent_words,
    )
    test_sents = load_all_sents(
        [os.path.join(txt_path, filename) for filename in test_filenames],
        min_sent_words,
    )

    with open(params["paths"]["train_sents"], "w") as f:
        json.dump(train_sents, f, ensure_ascii=False)
    with open(params["paths"]["test_sents"], "w") as f:
        json.dump(test_sents, f, ensure_ascii=False)
