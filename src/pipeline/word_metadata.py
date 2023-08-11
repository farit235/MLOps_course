import typing as t

import stanza


class SentenceWithGap:
    GAP_FILLER = "_____"

    def __init__(
        self,
        original_sentence: str,
        start_char_in_sent: int,
        end_char_in_sent: int,
        correct_form: str,
        lemma: str,
        pos: str,
    ):
        self.original_sentence = original_sentence
        self.sentence_with_filler = (
            original_sentence[:start_char_in_sent]
            + self.GAP_FILLER
            + original_sentence[end_char_in_sent:]
        )
        self.form = correct_form
        self.lemma = lemma
        self.pos = pos
        # toDo: get level
        self.other_options: t.List[str] = []

    def set_other_options_info(self, other_options):
        self.other_options = other_options


class WordMetadataGenerator:
    UPOS_WHITELIST: t.List[str] = ["ADJ", "ADV", "NOUN", "VERB"]

    def __init__(self):
        # toDo predownload
        # stanza.download('en')
        self.nlp = stanza.Pipeline("en", processors="tokenize,mwt,pos,lemma")

    def process_text(self, text: str) -> t.List[SentenceWithGap]:
        text = text.replace(SentenceWithGap.GAP_FILLER, " ")
        sents_with_gaps = []

        doc = self.nlp(text)
        for sent in doc.sentences:
            sent_start_char = sent.tokens[0].start_char
            sent_end_char = sent.tokens[-1].end_char
            sent_text = text[sent_start_char:sent_end_char]
            for token in sent.tokens:
                for word in token.words:
                    upos = word.upos
                    if upos not in self.UPOS_WHITELIST:
                        continue
                    lemma = word.lemma
                    form = word.text
                    start_in_sent = word.start_char - sent_start_char
                    end_in_sent = word.end_char - sent_start_char
                    sents_with_gaps.append(
                        SentenceWithGap(
                            original_sentence=sent_text,
                            start_char_in_sent=start_in_sent,
                            end_char_in_sent=end_in_sent,
                            correct_form=form,
                            lemma=lemma,
                            pos=upos,
                        )
                    )
        return sents_with_gaps
