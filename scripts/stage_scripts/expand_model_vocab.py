import sys
from pathlib import Path

root = str(Path(__file__).parent.parent.parent)
sys.path.append(root)

import json

from dvc.api import params_show

from src.models.finetune_mlm.expand_vocab import get_initional_model_and_tokenizer

if __name__ == "__main__":
    params = params_show()

    source_bert_path = params["training"]["source_bert"]
    new_words_path = params["paths"]["extra_words"]
    save_expanded_vocab_model_path = params["paths"]["expanded_vocab_model"]

    with open(new_words_path) as f:
        new_words = json.load(f)
    get_initional_model_and_tokenizer(
        source_bert_path, save_expanded_vocab_model_path, new_words
    )
