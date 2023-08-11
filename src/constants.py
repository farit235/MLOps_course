import os
from pathlib import Path

ROOT_DIR = str(Path(__file__).parent.parent)

# SOURCE_BERT = "allenai/scibert_scivocab_uncased"
SOURCE_BERT = "prajjwal1/bert-tiny"

DATA_DIR = "data"
PDF_DIR = os.path.join(ROOT_DIR, DATA_DIR, "raw")
EXTRACTED_TEXT_DIR = os.path.join(ROOT_DIR, DATA_DIR, "processed")
MIN_SENT_WORDS = 7
TEST_SIZE = 0.1

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)
