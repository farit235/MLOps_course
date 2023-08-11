import typing as t

from razdel import sentenize


def get_file_sents(filepath: str) -> t.List[str]:
    with open(filepath) as f:
        return [i.text for i in sentenize(f.read())]


def load_all_sents(filepaths: t.List[str], min_sent_words: int = 10) -> t.List[str]:
    sents = []
    for filepath in filepaths:
        sents += get_file_sents(filepath)
    return [sent for sent in sents if len(sent.split()) > min_sent_words]
