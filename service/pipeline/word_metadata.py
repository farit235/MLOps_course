import typing as t
import razdel
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from nltk.stem.snowball import StawballStemmer
from nltk.tag.perceptron import PerceptronTagger
from nltk.tag.mapping import map_tag
import re


class SentenceWithGap:
    GAP_FILLER = "_____"

    def __init__(
        self,
        original_sentence: str,
        start_char_in_sent: int,
        end_char_in_sent: int,
        correct_form: str,
        pos: str,
        stem:str
    ):
        self.original_sentence = original_sentence
        self.sentence_with_filler = (
            original_sentence[:start_char_in_sent]
            + self.GAP_FILLER
            + original_sentence[end_char_in_sent:]
        )
        self.form = correct_form
        self.pos = pos
        # toDo: get level
        self.other_options: t.List[str] = []
        self.stem:str = stem

    def set_other_options_info(self, other_options):
        self.other_options = other_options


class WordMetadataGenerator:
    UPOS_WHITELIST: t.List[str] = ["ADJ", "ADV", "NOUN", "VERB"]

    def __init__(self, language="english"):
        self.tagger = PerceptronTagger()
        self.language = language
        self.stopwords = set(stopwords.words(language))
        self.stemmer = StawballStemmer(language)

    def _get_pos_tags(self, tokenized_sent: t.List[str]) -> t.List[str]:
        # do not use nltk.pos_tag, as nltk.pos_tag_sents is much faster
        ptb_pos_tags = [tag for _, tag in self.tagger.tag(tokenized_sent)]
        universal_tags = [
                map_tag("en-ptb", "universal", ptb_pos_tag)
                for ptb_pos_tag in ptb_pos_tags
            ]
        return universal_tags

    def process_text(self, text: str) -> t.List[SentenceWithGap]:
        text = text.replace(SentenceWithGap.GAP_FILLER, " ")
        sents_with_gaps = []

        for sent_text in sent_tokenize(text):
            tokens = list(razdel.tokenize(sent_text))
            pos_tags = self._get_pos_tags([token.text for token in tokens])
            for token, pos_tag in zip(tokens, pos_tags):
                form = token.text
                if len(form) < 3:
                    continue
                if not re.search("[a-zA-Z]", form):
                    continue
                if form in self.stopwords:
                    continue
                if pos_tag not in self.UPOS_WHITELIST:
                    continue
                start_in_sent = token.start
                end_in_sent = token.stop
                sents_with_gaps.append(
                    SentenceWithGap(
                        original_sentence=sent_text,
                        start_char_in_sent=start_in_sent,
                        end_char_in_sent=end_in_sent,
                        correct_form=form,
                        pos=pos_tag,
                        stem=self.stemmer.stem(form)
                    )
                )
        return sents_with_gaps
