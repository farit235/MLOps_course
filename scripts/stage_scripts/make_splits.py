import sys
from pathlib import Path

root = str(Path(__file__).parent.parent.parent)
sys.path.append(root)

import json
import os

from dvc.api import params_show
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    params = params_show()
    txt_dir = params["paths"]["txt_dir"]
    split_path = params["paths"]["splits_path"]
    test_size = params["training"]["test_size"]

    all_txt_filenames = os.listdir(txt_dir)
    train_filenames, test_filenames = train_test_split(
        all_txt_filenames, test_size=test_size, random_state=42
    )
    with open(split_path, "w") as f:
        json.dump({"train": train_filenames, "test": test_filenames}, f)
