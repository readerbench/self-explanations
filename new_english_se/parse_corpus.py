from os.path import exists

import pandas
import re

from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.document import Document
from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model


def clear_text(text: str):
    return re.sub('[^a-zA-Z0-9 \n.]', '', str(text))


def get_matching_sentence(df_target_sentences, dataset_row, sentence_index):
    return df_target_sentences.loc[
        # TEXT_ID is a string type (e.g. NIU 1 - 338) and needs to be stripped to overcome any cleanup dataset issues
        (df_target_sentences[SelfExplanations.TEXT_ID].str.strip() == dataset_row[SelfExplanations.TEXT_ID].strip()) &
        (df_target_sentences[SelfExplanations.SENT_NO] == dataset_row[SelfExplanations.SENT_NO])
        ][sentence_index].values[0]


class SelfExplanations:
    TEXT_ID = "TextID"
    SENT_NO = "SentNo"

    SE = "SelfExplanation"
    TARGET_SENTENCE = "TargetSentence"
    PREVIOUS_SENTENCE = "PreviousSentence"
    TOO_SHORT = "tooshort"
    NON_SENSE = "nonsense"
    IRRELEVANT = "irrelevant"
    COPY_PASTE = "copypaste"
    MISCONCEPTION = "misconception"
    MONITORING = "monitoring"
    PARAPHRASE = "paraphrasepresence"
    PR_LEXICAL_CHANGE = "lexical change"
    PR_SYNTACTIC_CHANGE = "syntactic change"
    BRIDGING = "bridgepresence"
    BR_CONTRIBUTION = "bridgecontribution"
    ELABORATION = "elaborationpresence"
    EL_LIFE_EVENT = "lifeevent"

    def parse_se_scoring_from_csv(self, path_to_csv_file: str):
        df = pandas.read_csv(path_to_csv_file, delimiter=',', dtype={self.SENT_NO: "Int64"}).dropna(how='all')
        # print(df.sample(5))
        df[self.SE] = df[self.SE].map(clear_text)
        self.df = df

        enhanced_dataset_file = 'new_english_se2_enhanced.csv'
        # compute target and previous sentences if necessary
        if not exists(enhanced_dataset_file):
            self.df_target_sentences = pandas.read_csv("targetsentences.csv", delimiter=',', dtype={self.SENT_NO: "Int64"})
            self.df[self.TARGET_SENTENCE] = self.df.apply(
                lambda x: get_matching_sentence(self.df_target_sentences, x, SelfExplanations.TARGET_SENTENCE), axis=1)
            self.df[self.PREVIOUS_SENTENCE] = self.df.apply(
                lambda x: get_matching_sentence(self.df_target_sentences, x, SelfExplanations.PREVIOUS_SENTENCE), axis=1)
            self.df.to_csv(enhanced_dataset_file, index=False)

        return enhanced_dataset_file


    def compute_complexity_indices(self):
        model = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
        result_df = self.df.copy()
        # compute for target sentences
        for index, row in self.df.iloc[0:].iterrows():
            print("Processing index ", index)
            doc_se = Document(lang=Lang.EN, text=row[self.SE])
            if len(doc_se.get_words()) > 5:
                cna_se = CnaGraph(docs=doc_se, models=[model])
                compute_indices(doc_se, cna_se)
                for key, value in doc_se.indices.items():
                    result_df.loc[index, "se." + repr(key)] = value
                doc_target = Document(lang=Lang.EN, text=row[self.TARGET_SENTENCE])
                cna_target = CnaGraph(docs=doc_target, models=[model])
                compute_indices(doc_target, cna_target)
                for key, value in doc_target.indices.items():
                    result_df.loc[index, "target." + repr(key)] = value

        result_df.to_csv("new_english_se_indices2.csv", index=False)

        return result_df


    def load_se_with_indices(self):
        df = pandas.read_csv("new_english_se_indices2.csv", delimiter=',').dropna()
        return df


    def compute_copy_paste(self, se: str, target: str):
        return False
