from importlib_metadata import Pair
from nltk import ngrams
from pandas import DataFrame
import re

from rb import Lang
from rb.core.document import Document
from rb.similarity.wordnet import leacock_chodorow_similarity

from svm.zero_tags import ZeroTags

class ZeroRules:
    # delta was included from the previous results
    SHORT_THRESHOLD_LOW = 0.4
    SHORT_THRESHOLD_HIGH = 0.55
    NGRAM_DIM = 5
    SEMANTIC_SIMILARITY = 0.2

    def __init__(self, has_previous_sentence: bool = False):
        self.previous_sentence_enabled = has_previous_sentence
        self.regex_list = self.read_frozen_expressions()

    def predict_zeros(self, df: DataFrame):
        result_df = df.copy()
        for index, row in df.iloc[0:].iterrows():
            result_df.loc[index, "Tags"] = str(
                self.filter_and_tag_zeros(row["SelfExplanation"], row["TargetSentence"],
                                          row["PreviousSentence"] if self.previous_sentence_enabled is True else None))
        return result_df

    def filter_and_tag_zeros(self, se: str, target: str, previous: str):
        tags = []
        clean_se = self.remove_frozen_expressions(se)
        se_clean_doc = Document(Lang.EN, clean_se)
        target_doc = Document(Lang.EN, target)
        previous_doc = None
        if previous is not None:
            previous_doc = Document(Lang.EN, previous)
        tags.append(self.check_frozen_expression(se, clean_se))
        tags.append(self.check_irrelevant(se_clean_doc, se, target, previous))
        tags.append(self.check_short(se_clean_doc, target_doc))
        tags.append(self.check_copy_paste_ngram(se_clean_doc, target_doc, previous_doc))
        return [tag.name for tag in tags if tag is not None]

    def remove_frozen_expressions(self, se_text: str):
        se_text = se_text.lower()
        for tag, regex_compiled in self.regex_list:
            if regex_compiled.search(se_text):
                se_text = re.sub(regex_compiled, '', se_text)
        return se_text.strip()

    def check_frozen_expression(self, se: str, clean_se: str):
        words_se = re.split('\\W+', se)
        words_clean_se = re.split('\\W+', clean_se)
        if len(words_se) == len(words_clean_se):
            return
        if len(words_clean_se) / len(words_se) < 0.25:
            return ZeroTags.FROZEN_EXPRESSIONS

    def read_frozen_expressions(self):
        frozen_expr_file = open('frozen_expressions.txt', 'r')
        lines = frozen_expr_file.readlines()
        regex_list = []
        for line in lines:
            if line != "\n":
                index_of_separation = line.index(":")
                frozen_expr_tag = line[0: index_of_separation]
                regex = line[index_of_separation + 1:]
                regex_list.append(Pair(frozen_expr_tag, re.compile(regex.strip(), re.A)))

        return regex_list

    def check_irrelevant(self, se_clean_doc: Document, se_text: str, target_text: str, previous_text: str):
        content_words_list = [word for word in se_clean_doc.get_words() if word.is_content_word()]
        if len(content_words_list) < 2:
            return ZeroTags.IRR
        target_cohesion = leacock_chodorow_similarity(se_text, target_text, Lang.EN)
        previous_cohesion = leacock_chodorow_similarity(se_text, previous_text, Lang.EN) if self.previous_sentence_enabled else 1
        if target_cohesion < self.SEMANTIC_SIMILARITY or previous_cohesion < self.SEMANTIC_SIMILARITY:
            return ZeroTags.IRR_COH

    def check_short(self, se_clean_doc: Document, target_doc: Document):
        no_words_se_clean = len(se_clean_doc.get_words())
        if no_words_se_clean <= self.get_short_threshold(target_doc):
            return ZeroTags.SH

    def get_short_threshold(self, target_doc: Document):
        no_words_target = len(target_doc.get_words())
        if no_words_target >= 10:
            return min(no_words_target, 20) * self.SHORT_THRESHOLD_LOW
        return no_words_target * self.SHORT_THRESHOLD_HIGH

    def check_copy_paste_ngram(self, se_clean_doc: Document, target_doc: Document, previous_doc: Document):
        if self._check_copy_paste_ngram(se_clean_doc, [target_doc]):
            return ZeroTags.COPY_PASTE_TARGET
        elif self._check_copy_paste_ngram(se_clean_doc, [previous_doc]):
            return ZeroTags.COPY_PASTE_PREV

    def _check_copy_paste_ngram(self, se_clean_doc: Document, target_docs: list):
        se_clean_ngrams = self.get_n_grams_from_doc(se_clean_doc)
        target_docs_ngrams = {n_gram for target_doc in target_docs if target_doc is not None
                              for n_gram in self.get_n_grams_from_doc(target_doc)}
        for ngram_from_se in se_clean_ngrams:
            if ngram_from_se not in target_docs_ngrams:
                return False
        return True

    def get_n_grams_from_doc(self, doc: Document):
        return ngrams(doc.get_words(), self.NGRAM_DIM)
