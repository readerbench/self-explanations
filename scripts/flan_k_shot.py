import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchmetrics import F1Score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from core.data_processing.se_dataset import SelfExplanations
from scripts.mtl_bert_train import get_new_train_test_split, get_filterable_cols
import warnings
warnings.filterwarnings('ignore')

def load_model(model_size):
    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{flan_size}", device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{flan_size}")

    return model, tokenizer

def batch_eval(model, tokenizer, sentences, batch_size=256, targets=[], task_name=""):
    predictions = []
    for i in range(1 + len(sentences) // batch_size):
        inputs = tokenizer(
            sentences[i*batch_size: (i+1) * batch_size],
            return_tensors="pt",
            padding=True)
        # print(sentences[i*batch_size])
        outputs = model.generate(**inputs)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(result)
        result = [min(int(x), max(targets)) if x.isnumeric() else 0 for x in result]

        predictions += result
    print("=" * 33)
    targets = np.array(targets)
    predictions = np.array(predictions)
    print(f"task:{task_name} f1:{f1_score(targets, predictions, average='weighted')}")
    print(classification_report(targets, predictions))
    print(confusion_matrix(targets, predictions))
    print("=" * 33)



def get_prompt(prompt_structure, class_name, class_definition, class_meaning, source, production, source_ex=None, production_ex=None, result_ex=None):
    if prompt_structure == 0:
        options = "\n".join([f"{i} - {class_meaning[i]}" for i in class_meaning])
        if class_name == SelfExplanations.OVERALL:
            question = "What is the overall quality of the self-explanation in the Target Sentence, based on the Source Sentence."
        else:
            question = f"Assess the {class_name} score for self-explanation in the Target Sentence, based on the Source Sentence."
        example = ""
        if source_ex is not None:
            for i in range(len(source_ex)):
                example += f"Question: {question}\nSource Sentence: {source_ex[i]}\nTarget Sentence: {production_ex[i]}\nAnswer: {result_ex[i]}\n"
        return f"Context: {class_definition} This class has the following options: \n{options}\n\n" \
               f"{example}" \
               f"Question: {question}\nSource Sentence: {source}\nTarget Sentence: {production}\nAnswer: "
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

        for i in range(SelfExplanations.MTL_CLASS_DICT[task_df_label]):
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
            2: "Contains multiple qualitative examples"
        },
        SelfExplanations.BRIDGING: {
            0: "Not present",
            1: "Contains one example",
            2: "Contains multiple qualitative examples"
        },
        SelfExplanations.ELABORATION: {
            0: "Not present",
            1: "Contains at least one example"
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

    self_explanations = SelfExplanations()
    self_explanations.parse_se_from_csv("../data/results_paraphrase_se_aggregated_dataset_2.csv")

    for flan_size in ["small", "base", "large", "xl", "xxl"]:
        model, tokenizer = load_model(flan_size)
        for sentence_mode in ["none", "target", "targetprev"]:
            df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df, sentence_mode)
            for num_examples in [0, 1, 2]:
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
                        source, prod, label = get_examples(df_dev, task_df_label, seed=index, num_examples=0)
                        sentences.append(get_prompt(0, task_name, class_definitions[task_df_label],
                                                    shortened_class_item_meaning_dict[task_df_label],
                                                    line['Source'], line['Production'],
                                                    source, prod, label))
                    targets = df_test[task_df_label].values.tolist()

                    batch_eval(model, tokenizer, sentences, batch_size=256, targets=targets, task_name=task_name)