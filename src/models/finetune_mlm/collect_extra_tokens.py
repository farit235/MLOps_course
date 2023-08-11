import re
import typing as t
from collections import Counter

import razdel
from transformers import BertTokenizer


def is_like_english_word(char_seq: str) -> t.Any:
    return re.search("[a-zA-Z]", char_seq)


def get_english_tokens(text: str) -> t.List[str]:
    """
    :param text:
    :return: list of tokens, which look like English words
    """
    # toDo check if bert is cased
    return [
        tok.text.lower()
        for tok in razdel.tokenize(text)
        if is_like_english_word(tok.text)
    ]


def get_common_unknown_words(
    tokenizer: BertTokenizer, sents: t.List[str], threshold: int = 50
) -> t.List[str]:
    """
    :param tokenizer: previous version of tokenizer
    :param sents: train sents
    :param threshold: minimal corpus frequency of common word
    :return: list of common English words, which are not presented in
    previous version of tokenizer as single tokens
    """
    vocab = tokenizer.vocab.keys()
    unknown_words_counter = Counter()  # type: Counter
    for sent in sents:
        english_tokens = get_english_tokens(sent)
        unknown_tokens = [tok for tok in english_tokens if tok not in vocab]
        unknown_words_counter.update(unknown_tokens)

    # toDo: filter proper names
    common_unknown_tokens = [
        tok
        for tok in unknown_words_counter
        if unknown_words_counter[tok] >= threshold
        # filter results of line breaks
        and not tok.startswith("-") and not tok.endswith("-")
    ]
    return common_unknown_tokens
