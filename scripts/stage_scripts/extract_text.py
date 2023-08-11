import sys
from pathlib import Path

root = str(Path(__file__).parent.parent.parent)
sys.path.append(root)

import os

from dvc.api import params_show

from src.data.parsing import save_extraction

if __name__ == "__main__":
    params = params_show()
    pdf_dir = params["paths"]["pdf_dir"]
    txt_dir = params["paths"]["txt_dir"]

    os.makedirs(txt_dir, exist_ok=True)

    TXT_EXTENSION = ".txt"
    for pdf_filename in os.listdir(pdf_dir):
        base_name, file_extension = os.path.splitext(pdf_filename)
        txt_filename = base_name + TXT_EXTENSION
        save_extraction(
            os.path.join(pdf_dir, pdf_filename), os.path.join(txt_dir, txt_filename)
        )
