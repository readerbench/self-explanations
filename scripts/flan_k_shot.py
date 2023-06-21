import random
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from core.data_processing.se_dataset import SelfExplanations
import warnings
# warnings.filterwarnings('ignore')

def map_train_test(x):
    if x['Dataset'] in ['ASU 5']:
        return 'train'
    if x['Dataset'] == 'CRaK':
        return 'dump'
    if x['Dataset'] == 'ASU 1':
        if 'RBC' in x['TextID'].upper():
            return 'test'
        if x['PrePost'] == 'Post':
            return 'dev'
        return 'train'
    if x['Dataset'] == 'ASU 4':
        if not str(x['ID']).startswith('ISTARTREF'):
            return 'train'
        else:
            return 'dev'
    return 'dump'


def get_new_train_test_split(df, target_sentence_mode="target"):
    df.loc[df[SelfExplanations.ELABORATION] == 2, SelfExplanations.ELABORATION] = 1
    df.loc[df[SelfExplanations.BRIDGING] == 3, SelfExplanations.BRIDGING] = 2

    if target_sentence_mode == "none":
        df["Source"] = ""
        df[SelfExplanations.TARGET_SENTENCE] = ""
        df_cols_keep = df.columns[:114].tolist() + [c for c in df.columns if "source" in c]
        df = df[df_cols_keep]
    elif target_sentence_mode == "targetprev":
        df["Source"] = df[SelfExplanations.PREVIOUS_SENTENCE].astype(str) + " " + df[SelfExplanations.TARGET_SENTENCE].astype(str)
        df[SelfExplanations.TARGET_SENTENCE] = df[SelfExplanations.PREVIOUS_SENTENCE].astype(str) + " " + df[SelfExplanations.TARGET_SENTENCE].astype(str)

    df['EntryType'] = df.apply(lambda x: map_train_test(x), axis=1)
    return df[(df['EntryType'] == 'train') | (df['EntryType'] == 'dev')], df[df['EntryType'] == 'dev'], df[df['EntryType'] == 'test']
    # return df[(df['EntryType'] == 'train')], df[df['EntryType'] == 'dev'], df[df['EntryType'] == 'test']


def load_model(flan_size):
    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{flan_size}", device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{flan_size}")

    return model, tokenizer

def batch_eval(model, tokenizer, sentences, batch_size=256, targets=[], task_name=""):
    predictions = []
    grades = ["A", "B", "C", "D"]
    for i in range(1 + len(sentences) // batch_size):
        inputs = tokenizer(
            sentences[i*batch_size: (i+1) * batch_size],
            return_tensors="pt",
            padding=True)
        if i % 10 == 0:
            print(f"Seen {i} batches.")
        # print(sentences[i*batch_size])
        outputs = model.generate(**inputs, max_length=20)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(result)
        result = [f"{x[1]}" if x.startswith("(") else x for x in result]
        result = [grades.index(x) if x in grades else 0 for x in result]
        # print(result)
        predictions += result
    print("=" * 33)
    targets = np.array(targets)
    predictions = np.array(predictions)
    print(f"task:{task_name} f1:{f1_score(targets, predictions, average='weighted')}")
    print(classification_report(targets, predictions))
    print(confusion_matrix(targets, predictions))
    print("=" * 33)

def get_prompt(prompt_structure, num_classes, class_name, class_definition, class_meaning, source, production, source_ex=None, production_ex=None, result_ex=None):
    if prompt_structure == 0:
        grades = ["A", "B", "C", "D"]
        options = "\n".join([f"({grades[i]}) {class_meaning[i]}" for i in class_meaning])
        based_on_text = ", based on the S2." if len(source) > 0 else "."
        # based_on_text += f"(Accepted answers: {','.join([str(x) for x in range(num_classes)])})"
        if class_name == SelfExplanations.OVERALL:
            question = f"What is the overall quality of the self-explanation in S1{based_on_text}"
        else:
            question = f"Assess the {class_name} score for the self-explanation in S1{based_on_text}"
        example = ""
        if source_ex is not None:
            for i in range(len(source_ex)):
                example += f"Question: {question}\n" \
                           f"S1: {production_ex[i]}\n" \
                           f"S2: {source_ex[i]}\n" \
                           f"Answer: ({grades[result_ex[i]]})\n"
                           # f"Options:\n{options}\n" \
                           # f"Answer: ({grades[result_ex[i]]})\n"
        source_text = (f"S2: {source}\n" if len(source) > 0 else "")
        return f"Context: {class_definition}\n" \
               f"{example}" \
               f"Question: {question}\n" \
               f"S1: {production}\n" \
               f"{source_text}" \
               f"Options:\n{options}\n" \
               f"Answer: "
    return ""

def get_examples(df, task_df_label, seed=1, num_examples=1):
    if num_examples == 0:
        return None, None, None
    if num_examples == 1:
        row = df.sample(random_state=13+seed)
        return [row['Source'].values[0]], [row['Production'].values[0]], [row[task_df_label].values[0]]
    else:
        source = []
        prod = []
        label = []
        class_list = list(range(SelfExplanations.MTL_CLASS_DICT[task_df_label]))
        random.shuffle(class_list)
        for i in class_list:
            df_reduced = df[df[task_df_label] == i]
            row = df_reduced.sample(random_state=13+seed)
            source.append(row['Source'].values[0])
            prod.append(row['Production'].values[0])
            label.append(row[task_df_label].values[0])

        return [source, prod, label]


if __name__ == '__main__':
    class_item_meaning_dict = {
        SelfExplanations.PARAPHRASE: {
            0: "not present",
            1: "contains at least one full clause that overlaps; may include single words; contains some of (about half) of the main idea units from the target sentence",
            2: "contains most of (50% or more) of the main idea units from the target sentence"
        },
        SelfExplanations.BRIDGING: {
            0: "not present",
            1: "includes anaphoric reference or one or two words from prior text",
            #2: "includes a complete idea that is from the prior text, but vaguely conveyed",
            2: "includes one or more complete ideas from previous ideas in the text (an idea is not necessarily a sentence, but it is a complete idea unit)"
        },
        SelfExplanations.ELABORATION: {
            0: "No apparent connection between text and elaborated ideas (random ideas)",
            # 1: "Moderate relationship between ideas – relation is evident and relevant to the text, but the connection is not strong and/or not explained; or is not related to the larger discourse context",
            # 2: "Strong relationship between ideas – relation is clearly evident and relevant to the larger discourse context of the text"
            1: "A relationship between the text and the elaborated ideas exists.     "
        },
        SelfExplanations.OVERALL: {
            0: "Poor: Self-explanations that contain unrelated or non-informative information or are very short or too similar to the target sentence",
            1: "Fair quality: Self-explanations that either 1) only focus on the target sentence,(e.g. paraphrase, comprehension monitoring, brief but related idea, brief prediction) OR 2)has only unclear or incoherent connections to other parts of the text (e.g., weak connections to prior text, simple one-word anaphor resolution)",
            2: "Good quality: Self-explanations that include 1-2 ideas from text outside the target sentence (local bridging, brief elaboration, brief or weak distal bridges)",
            3: "Great: High quality self-explanations that incorporate information at a global level(e.g., high quality local bridges, distal bridging, or meaningful elaborations)"
        }
    }
    shortened_class_item_meaning_dict = {
        SelfExplanations.PARAPHRASE: {
            0: "Not present",
            1: "Contains one example",
            2: "Contains multiple examples"
        },
        SelfExplanations.BRIDGING: {
            0: "Not present",
            1: "Contains one example",
            2: "Contains multiple examples"
        },
        SelfExplanations.ELABORATION: {
            0: "Not present",
            1: "Contains one example"
        },
        SelfExplanations.OVERALL: {
            0: "Poor",
            1: "Fair quality",
            2: "Good quality",
            3: "High quality"
        }
    }
    class_definitions = {
        SelfExplanations.PARAPHRASE: "Paraphrasing involves restating text in one's own words.",
        SelfExplanations.BRIDGING: "Bridging involves linking multiple ideas in the text. Bridging inferences help the reader to understand how those ideas are related to one another.",
        SelfExplanations.ELABORATION: "Elaboration means that the reader is connecting information from the text to their own knowledge base. The relevant information may come from previously acquired knowledge, or the reader may use logic and common sense to elaborate beyond the textbase.",
        SelfExplanations.OVERALL: "The overall score assesses the quality of the self explanation in terms of paraphrasing, bridging and elaboration. A high quality self-explanation is one that incorporates information at a global level(e.g., high quality local bridges, distal bridging, or meaningful elaborations)."
    }
    print("Starting program")
    self_explanations = SelfExplanations()
    print("Loading SEs")
    self_explanations.parse_se_from_csv("../data/results_se_aggregated_dataset_clean.csv")
    print("Loaded SEs")

    for flan_size in ["small", "base", "large", "xl", "xxl"]:
    # for flan_size in ["large"]:
        model, tokenizer = load_model(flan_size)
        print("Loaded model")
        # for sentence_mode in ["none", "target", "targetprev"]:
        for sentence_mode in ["target", "targetprev"]:
            df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df, sentence_mode)
            for num_examples in [0, 1, 2]:
                random.seed(13)
                print("$" + "=" * 33)
                print("$" + "=" * 33)
                print(f">>Model:{flan_size} sentence_mode:{sentence_mode} num_examples:{num_examples}")
                targets = []
                for num_classes, task_name, task_df_label in [
                    (3, "paraphrasing", SelfExplanations.PARAPHRASE),
                    (3, "elaboration", SelfExplanations.ELABORATION),
                    (2, "bridging", SelfExplanations.BRIDGING),
                    (4, "overall", SelfExplanations.OVERALL),
                ]:
                    sentences = []
                    for index, line in df_test.iterrows():
                        source, prod, label = get_examples(df_dev, task_df_label, seed=index, num_examples=num_examples)
                        sentences.append(get_prompt(0, num_classes, task_name, class_definitions[task_df_label],
                                                    shortened_class_item_meaning_dict[task_df_label],
                                                    line['Source'], line['Production'],
                                                    source, prod, label))
                    targets = df_test[task_df_label].values.tolist()

                    batch_eval(model, tokenizer, sentences, batch_size=8, targets=targets, task_name=task_name)