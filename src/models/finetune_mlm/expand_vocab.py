import os
import re
import shutil
import typing as t
from collections import Counter

import razdel
import torch
from transformers import BertModel, BertTokenizer


def is_like_english_word(char_seq: str) -> bool:
    return re.search("[a-zA-Z]", char_seq) is not None


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
    :param file_paths: train data plain text paths
    :param threshold: minimal corpus frequency of common word
    :return: list of common English words, which are not presented in
    previous version of tokenizer as single tokens
    """
    vocab = tokenizer.vocab.keys()
    unknown_words_counter: Counter = Counter()
    for sent in sents:
        english_tokens = get_english_tokens(sent)
        unknown_tokens = [tok for tok in english_tokens if tok not in vocab]
        unknown_words_counter.update(unknown_tokens)
    common_unknown_tokens = [
        tok
        for tok in unknown_words_counter
        if unknown_words_counter[tok] >= threshold
        and not tok.endswith("-")
        and not tok.startswith("-")
    ]
    return common_unknown_tokens


def update_tokenizer(
    old_tokenizer: BertTokenizer, new_words: t.List[str]
) -> BertTokenizer:
    """
    see https://t.me/natural_language_processing/57138
    """
    tmp_tok = "tmp_tok"
    old_tokenizer.save_pretrained(tmp_tok)
    with open(os.path.join(tmp_tok, "vocab.txt"), "a") as f:
        for token in new_words:
            f.write(token + "\n")
    new_tokenizer = BertTokenizer.from_pretrained(tmp_tok)
    shutil.rmtree(tmp_tok)
    return new_tokenizer


def get_embedding_weights(bert: BertModel) -> torch.Tensor:
    return bert.embeddings.word_embeddings.weight.data.clone().detach()


def get_initial_representation(
    bert_model: BertModel, old_tokenizer: BertTokenizer, new_token: str
) -> torch.Tensor:
    old_encoding = old_tokenizer(new_token)
    # toDo use attentions to get weighted average
    special_tokens_mask = old_tokenizer.get_special_tokens_mask(old_encoding)
    meaningful_old_tokens = [
        id_
        for id_, is_masked in zip(old_encoding["input_ids"], special_tokens_mask)
        if not is_masked
    ]
    if meaningful_old_tokens:
        return get_embedding_weights(bert_model)[meaningful_old_tokens].mean(0)
    else:
        raise NotImplementedError("Random initialization is not implemented")


def get_initional_model_and_tokenizer(
    old_model_name_or_path: str, save_path: str, new_words: t.List[str]
) -> None:
    model = BertModel.from_pretrained(old_model_name_or_path)
    old_tokenizer = BertTokenizer.from_pretrained(old_model_name_or_path)

    new_tokenizer = update_tokenizer(old_tokenizer, new_words)
    new_words_initional_representation = [
        get_initial_representation(model, old_tokenizer, word) for word in new_words
    ]
    old_embeddings_weights = get_embedding_weights(model)
    new_embeddings_weights = torch.vstack(
        [old_embeddings_weights, *new_words_initional_representation]
    )
    model.resize_token_embeddings(len(new_tokenizer))
    model.embeddings.word_embeddings.weight.data = new_embeddings_weights

    model.save_pretrained(save_path)
    new_tokenizer.save_pretrained(save_path)
    return
