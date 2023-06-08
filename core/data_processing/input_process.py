import numpy
import pandas as pd
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.complexity.index_category import IndexCategory
from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model

from core.data_processing.se_dataset import SelfExplanations
from core.utils.rcdoc import create_doc
from core.utils.word_overlap import get_overlap_metrics, get_edit_distance_metrics, \
    get_pos_ngram_overlap_metrics

DATA_FOLDER = "../../data/"

ORIG_FILE_FIELDS = ["Dataset", "Pop.og", "Pop", "Prod.from", "Proto", "Condition.training", "Condition", "ID.og", "ID",
          "iSTARTorWPal", "Pretext", "TextID.count", "Cohort", "School", "Dev", "Domain", "Text", "TextID",
          "SentNo", "SelfExplanation", "tooshort", "irrelevant", "copypaste", "evaluativestatements",
          "misconception", "monitoring", "paraphrasepresence", "lexicalchange", "syntacticchange",
          "bridgepresence", "bridgecontribution", "elaborationpresence", "lifeevent", "overall", "irre_2",
          "copy_2", "eval_2", "toos_di", "irre_di", "copy_di", "eval_di", "misc_di", "moni_di", "para_",
          "brip_", "elab_", "para_di", "brip_di", "elab_di", "over_", "over_correct", "over_0", "over_1",
          "over_2", "over_3", "combocode", "combostring", "combo", "para_only", "brip_only", "elab_only",
          "para_brip", "para_elap", "brip_elap", "all_", "none_", "comboquality", "comboquality_descript",
          "poor_none", "fair_none", "good_none", "great_none", "poor_para", "fair_para", "good_para",
          "great_para", "poor_brip", "fair_brip", "good_brip", "great_brip", "poor_elap", "fair_elap",
          "good_elap", "great_elap", "poor_parabrip", "fair_parabrip", "good_parabrip", "great_parabrip",
          "poor_paraelab", "fair_paraelab", "good_paraelab", "great_paraelab", "poor_bripelab", "fair_bripelab",
          "good_bripelab", "great_bripelab", "poor_all", "fair_all", "good_all", "great_all", "over0_filter",
          "over0_filtdesc", "none_0", "toos_0", "irre_0", "copy_0", "eval_0", "Training", "TextID.pre",
          "PrePost", "TargetSentence", "PreviousSentence"]

def get_indexes(lang_model, doc, index_categories):
    cna_graph = CnaGraph(docs=[doc], models=[lang_model])
    try:
        compute_indices(doc=doc, cna_graph=cna_graph)
    except IndexError:
        print("Skip index error")
        print({repr(ind): doc.indices[ind] for ind in doc.indices if ind.category in index_categories})

    local_relevant_indices = {repr(ind): doc.indices[ind] for ind in doc.indices if ind.category in index_categories}

    return local_relevant_indices


def read_and_process_se():
    file = "../../data/CR_all.columns_10.17.22_enhanced.xlsx"
    df = pd.read_excel(file)

    df['source'] = df[SelfExplanations.TARGET_SENTENCE]
    df['paraphrase'] = df[SelfExplanations.SE]
    return df


def write_dataset_header(f):
    f.write(f"Source\tProduction")
    for field in ORIG_FILE_FIELDS:
        f.write(f"\t{field}")


def write_dataset_info(f, doc, source, prod):
    f.write(f"{source}\t{prod}\t")
    for field in ORIG_FILE_FIELDS:
        f.write(f"{doc[field]}\t")


def extract_newline(data):
    return "" if type(data) != str else data.replace("\n", " ")


def skip_conditions(data_line):
    if type(data_line['source']) != str or len(data_line['source']) == 0:
        return True
    if type(data_line['paraphrase']) != str or len(data_line['paraphrase']) == 0:
        return True
    return False


def process_dataset(dataset, use_prev_sentence=False):
    """
    Enhances the dataset by adding ReaderBench-based features.
    :param dataset:
    :param use_prev_sentence:
    :return:
    """
    dataset_name = dataset
    doc_mod = read_and_process_se()

    w2v = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
    index_categories = [IndexCategory.WORD, IndexCategory.SURFACE, IndexCategory.MORPHOLOGY, IndexCategory.SYNTAX,
                        IndexCategory.DISCOURSE]
    index_categories_combined = [IndexCategory.COHESION]

    indices_list = []
    combined_indices_list = []
    overlap_list = []
    if use_prev_sentence:
        filename = f"{DATA_FOLDER}results_{dataset}_withprev.csv"
    else:
        filename = f"{DATA_FOLDER}results_{dataset}.csv"

    f = open(filename, "w")
    i = 0
    for index, data_line in doc_mod.iterrows():
        if use_prev_sentence:
            data_line['source'] = extract_newline(data_line['PreviousSentence']) + " " + data_line['source']
        else:
            data_line['source'] = extract_newline(data_line['source'])

        data_line['paraphrase'] = extract_newline(data_line['paraphrase'])
        data_line['PreviousSentence'] = extract_newline(data_line['PreviousSentence'])
        data_line['TargetSentence'] = extract_newline(data_line['TargetSentence'])
        data_line['SelfExplanation'] = extract_newline(data_line['SelfExplanation'])

        source = data_line["source"]
        prod = data_line["paraphrase"]
        if skip_conditions(data_line):
            continue
        source_doc = create_doc(source.strip().capitalize())
        prod_doc = create_doc(prod.strip().capitalize())
        print(f"{id} |{source.strip().capitalize()}|-|{prod.strip().capitalize()}|")
        overlap_metrics = get_overlap_metrics(source_doc, prod_doc)
        edit_dist_metrics = get_edit_distance_metrics(source_doc, prod_doc)
        pos_overlap_metrics = get_pos_ngram_overlap_metrics(source_doc, prod_doc)
        source_indexes = get_indexes(w2v, source_doc, index_categories)
        prod_indexes = get_indexes(w2v, prod_doc, index_categories)
        combined_indexes = get_indexes(w2v, prod_doc, index_categories_combined)
        similarity = w2v.similarity(source_doc, prod_doc)

        if len(indices_list) == 0:
            write_dataset_header(f)
            indices_list = list(source_indexes.keys())
            combined_indices_list = list(combined_indexes.keys())
            overlap_list = [key for key in overlap_metrics]
            edit_dist_list = [key for key in edit_dist_metrics]
            pos_overlap_list = [key for key in pos_overlap_metrics]
            indices_list.sort()
            combined_indices_list.sort()
            overlap_list.sort()
            edit_dist_list.sort()
            pos_overlap_list.sort()
            f.write("\t" + "\t".join([f"{i}_source" for i in indices_list]))
            f.write("\t" + "\t".join([f"{i}_prod" for i in indices_list]))
            f.write("\t" + "\t".join([f"{i}_combined" for i in combined_indices_list]))
            f.write("\t" + "\t".join(overlap_list))
            f.write("\t" + "\t".join(edit_dist_list))
            f.write("\t" + "\t".join(pos_overlap_list))
            f.write("\tw2v_similarity\n")
        i += 1
        if i % 250 == 0:
            print(i)
        write_dataset_info(f, data_line, source, prod)
        f.write("\t".join([str(source_indexes[key]) for key in indices_list]) + "\t")
        f.write("\t".join([str(prod_indexes[key]) for key in indices_list]) + "\t")
        f.write("\t".join([str(combined_indexes[key]) for key in combined_indices_list]) + "\t")
        f.write("\t".join([str(overlap_metrics[key]) for key in overlap_list]) + "\t")
        f.write("\t".join([str(edit_dist_metrics[key]) for key in edit_dist_list]) + "\t")
        f.write("\t".join([str(pos_overlap_metrics[key]) for key in pos_overlap_list]) + "\t")
        f.write(f"{similarity}\n")
    f.close()
    return filename


def clean_csv(file):
    df = pd.read_csv(file, delimiter='\t')

    # eliminate None entries
    df = df.replace('None', 0)

    # eliminating constant columns
    df = df.loc[:, (df != df.iloc[0]).any()]

    # normalize N/A datapoints
    for val in SelfExplanations.MTL_TARGETS:
        df[val][df[val] == 'BLANK '] = 9
        df[val][df[val] == 'BLANK'] = 9
        df[val][df[val] == 'blANK'] = 9
        df[val] = df[val].astype(int)

    #  fixing dtype errors
    feature_columns = df.columns.tolist()[38:]
    removable_cols = []
    for i, column in enumerate(feature_columns):
        if df[column].dtype.type not in [numpy.int64, numpy.float64]:
            try:
                df.loc[:, column] = df[column].astype(float)
                print("b", i, df[column].dtype.type)
            except:
                removable_cols.append(column)
                print("b", column, i, df[column].dtype.type, df[column].unique())
    print(f"Removed {removable_cols} because of datatype issues")
    df = df.drop(columns=removable_cols)

    df.to_csv(f"{file[:-4]}_clean.csv")
    return df


if __name__ == '__main__':
    dataset_options = ["se_aggregated_dataset"]

    for dataset in dataset_options:
        filename = process_dataset(dataset)
        clean_csv(filename)

