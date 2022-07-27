import pandas

from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.document import Document
from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model


class SelfExplanations:

    def parse_se_scoring_from_csv(self, path_to_csv_file: str):
        df = pandas.read_csv(path_to_csv_file, delimiter=',').dropna()

        print(df.sample(5))

        self.sentences = list(zip(df.target_text.values,
                                  df.self_explanation.values))
        self.labels_garbage = df.garbage.values
        self.frozen = df.frozen
        self.vague_irrelevant = df.vague_irrelevant
        self.labels_repeat = df.paraphrase.values
        self.labels_paraphrase = df.paraphrase.values
        self.labels_local_bridging = df.paraphrase.values
        self.labels_elaboration = df.elaboration.values
        self.labels_final = df.final.values
        self.df = df

    def compute_complexity_indices(self):
        model = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
        # compute for target sentences
        for index, row in self.df.iterrows():
            doc_target = Document(lang=Lang.EN, text=row['target_text'])
            cna_target = CnaGraph(docs=doc_target, models=[model])
            compute_indices(doc_target, cna_target)
            for key, value in doc_target.indices.items():
                self.df.loc[index, "target."+repr(key)] = value
            if len(row['self_explanation']) > 5:
                print("Processing index ", index)
                doc_se = Document(lang=Lang.EN, text=row['self_explanation'])
                cna_se = CnaGraph(docs=doc_se, models=[model])
                if len(doc_se.get_words()) > 0:
                    compute_indices(doc_se, cna_se)
                    for key, value in doc_se.indices.items():
                        self.df.loc[index, "se." + repr(key)] = value

        self.df.to_csv("millington_indices.csv", index=False)

        return self.df

    def load_se_with_indices(self):
        df = pandas.read_csv("millington_indices.csv", delimiter=',').dropna()
        return df
