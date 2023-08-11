import sys
from pathlib import Path

root = str(Path(__file__).parent.parent.parent)
sys.path.append(root)

import json

from dvc.api import params_show
from transformers import BertTokenizer

from src.models.finetune_mlm.collect_extra_tokens import get_common_unknown_words

if __name__ == "__main__":
    params = params_show()

    train_sents_path = params["paths"]["train_sents"]
    source_bert_path = params["training"]["source_bert"]
    threshold = params["training"]["new_word_threshold"]
    save_new_words_path = params["paths"]["extra_words"]

    tokenizer = BertTokenizer.from_pretrained(source_bert_path)
    with open(train_sents_path) as f:
        sents = json.load(f)
    extra_words = get_common_unknown_words(tokenizer, sents, threshold)
    with open(save_new_words_path, "w") as f:
        json.dump(extra_words, f)
