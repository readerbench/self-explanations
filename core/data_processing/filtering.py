import re
from enum import Enum
from importlib_metadata import Pair
from nltk import ngrams
from pandas import DataFrame


from rb import Lang
from rb.core.document import Document
from rb.similarity.wordnet import leacock_chodorow_similarity


class ZeroTags(Enum):
    FROZEN_EXPRESSIONS = "Frozen expressions are more than 75% of the entire SE"
    SH = "Too few words"
    IRR = "Too few content words"
    IRR_COH = "Extremely low cohesion"
    COPY_PASTE_PREV = "Copy & paste content words from previous"
    COPY_PASTE_TARGET = "Copy & paste content words from target"
    COPY_PASTE_BOTH = "Copy & paste content words from target and previous text"


class ZeroRules:
    # delta was included from the previous results
    SHORT_THRESHOLD_LOW = 0.4
    SHORT_THRESHOLD_HIGH = 0.55
    FE_THRESH = 0.25
    IRR_THRESH = 1
    COPY_THRESH = 0.1
    NGRAM_DIM = 5
    SEMANTIC_SIMILARITY = 0.2

    def __init__(self, has_previous_sentence: bool = False):
        self.previous_sentence_enabled = has_previous_sentence
        self.regex_list = self.read_frozen_expressions()

    def predict_zeros(self, df: DataFrame):
        result_df = df.copy()
        for index, row in df.iloc[0:].iterrows():
            result_df.loc[index, "Tags"] = str(
                self.get_filter_scores(row["SelfExplanation"], row["TargetSentence"],
                                       row["PreviousSentence"] if self.previous_sentence_enabled is True else None))
        return result_df

    def get_filter_scores(self, se: str, target: str, previous: str):
        # return [1.0, 1, 25, 1, 49, 1, 1.0, 1]
        try:
            clean_se = self.remove_frozen_expressions(se)
            clean_se = re.sub(' +', ' ', clean_se)

            se_clean_doc = Document(Lang.EN, clean_se)
            target_doc = Document(Lang.EN, target)
            previous_doc = Document(Lang.EN, previous) if previous is not None else None

            fe_score = self.frozen_expression_score(se, clean_se)
            irr_score = self.irrelevant_score(se_clean_doc, se, target, previous)
            sh_score = self.short_score(se_clean_doc)
            num_ngrams, num_copied_ngrams = self.copy_paste_ngram_score(se_clean_doc, [target_doc, previous_doc])
            copied_score = 1 - num_copied_ngrams / num_ngrams if num_ngrams > 0 else 0 # 0 if copied, 1 if 100% original
            return [
                fe_score, self.flag(fe_score, ZeroRules.FE_THRESH),
                irr_score, self.flag(irr_score, ZeroRules.IRR_THRESH),
                sh_score, self.flag(sh_score, self.get_short_threshold(target_doc)),
                copied_score, self.flag(copied_score, ZeroRules.COPY_THRESH)
            ]
        except:
            return [0] * 8

    def flag(self, x, thresh):
        return 1 if x > thresh else 0

    def remove_frozen_expressions(self, se_text: str):
        se_text = se_text.lower()
        for tag, regex_compiled in self.regex_list:
            if regex_compiled.search(se_text):
                se_text = re.sub(regex_compiled, '', se_text)
        return se_text.strip()

    def frozen_expression_score(self, se: str, clean_se: str):
        words_se = re.split('\\W+', se)
        words_clean_se = re.split('\\W+', clean_se)

        return len(words_clean_se) / len(words_se)

    def read_frozen_expressions(self):
        frozen_expr_file = open('/home/bnicula/snap/PASTEL/data/frozen_expressions.txt', 'r')
        lines = frozen_expr_file.readlines()
        regex_list = []
        for line in lines:
            if line != "\n":
                index_of_separation = line.index(":")
                frozen_expr_tag = line[0: index_of_separation]
                regex = line[index_of_separation + 1:]
                regex_list.append(Pair(frozen_expr_tag, re.compile(regex.strip(), re.A)))

        return regex_list

    def irrelevant_score(self, se_clean_doc: Document, se_text: str, target_text: str, previous_text: str):
        content_words_list = [word for word in se_clean_doc.get_words() if word.is_content_word()]
        return len(content_words_list)
        # # TODO 24.11.2022 see why leacock_chodorow_similarity ALWAYS returns 0
        # target_cohesion = leacock_chodorow_similarity(se_text, target_text, Lang.EN)
        # previous_cohesion = leacock_chodorow_similarity(se_text, previous_text, Lang.EN) if self.previous_sentence_enabled else 1
        # if target_cohesion < self.SEMANTIC_SIMILARITY and previous_cohesion < self.SEMANTIC_SIMILARITY:
        #     return ZeroTags.IRR_COH

    def short_score(self, se_clean_doc: Document):
        no_words_se_clean = len(se_clean_doc.get_words())

        return no_words_se_clean

    def get_short_threshold(self, target_doc: Document):
        no_words_target = len(target_doc.get_words())
        if no_words_target >= 10:
            return min(no_words_target, 20) * self.SHORT_THRESHOLD_LOW
        return no_words_target * self.SHORT_THRESHOLD_HIGH

    def copy_paste_ngram_score(self, se_clean_doc: Document, target_docs: list):
        se_clean_ngrams = self.get_n_grams_from_doc(se_clean_doc)
        target_docs_ngrams = {n_gram for target_doc in target_docs if target_doc is not None
                              for n_gram in self.get_n_grams_from_doc(target_doc)}
        ngram_score = 0
        total_ngrams = 0
        for ngram_from_se in se_clean_ngrams:
            total_ngrams += 1
            if ngram_from_se in target_docs_ngrams:
                ngram_score += 1
        return total_ngrams, ngram_score

    def get_n_grams_from_doc(self, doc: Document):
        return ngrams([w.text.lower() for w in doc.get_words()], self.NGRAM_DIM)


if __name__ == '__main__':
    z = ZeroRules()
    print(z.get_filter_scores("It is believed that a pair of old skates hung in the center of the door as decor.", "While a wreath is by far the most common and traditional way to decorate a front door, consider thinking outside of the box this year. We love using a pair of vintage skates hung in the center of the door as decor. A rustic snowshoe and of course a few strands of garland really finish the look.", ""))
    print(z.get_filter_scores("While a wreath is by far the most common and traditional way to decorate a front door", "While a wreath is by far the most common and traditional way to decorate a front door, consider thinking outside of the box this year. We love using a pair of vintage skates hung in the center of the door as decor. A rustic snowshoe and of course a few strands of garland really finish the look.", ""))
    print(z.get_filter_scores("Hello!", "While a wreath is by far the most common and traditional way to decorate a front door, consider thinking outside of the box this year. We love using a pair of vintage skates hung in the center of the door as decor. A rustic snowshoe and of course a few strands of garland really finish the look.", ""))
    # print(z.get_filter_scores("Self-explanations are useful for improving information attainment.", "Self-explanation has been shown to enhance learning across a variety of tasks, ranging from learning from texts to solving math problems. See the following examples of classic self-explanation studies that provide evidence of the effectiveness of this strategy", ""))