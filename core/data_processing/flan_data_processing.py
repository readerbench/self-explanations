import random
from core.data_processing.se_dataset import SelfExplanations, map_train_test


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


def get_prompt(prompt_structure, class_name, class_definition, class_meaning, source, production,
               source_ex=None, production_ex=None, result_ex=None, config=""):
    if prompt_structure == 0:
        sentenceA, sentenceB = "Student Sentence", "Source Sentence"
        if config["S1S2notSource"]:
            sentenceA, sentenceB = "S1", "S2"

        if config["numberingAlpha"]:
            grades = ["A", "B", "C", "D"]
        else:
            grades = ["1", "2", "3", "4"]
        options = "\n".join([f"({grades[i]}) {class_meaning[i]}" for i in class_meaning])
        based_on_text = f", based on the {sentenceB}." if len(source) > 0 else "."

        if class_name == SelfExplanations.OVERALL:
            question = f"What is the overall quality of the self-explanation in {sentenceA}{based_on_text}"
        else:
            question = f"Assess the {class_name} score for the self-explanation in {sentenceA}{based_on_text}"
        example = ""
        if source_ex is not None:
            for i in range(len(source_ex)):
                if config["S1S2before"]:
                    example += f"{sentenceA}: {production_ex[i]}\n" \
                               f"{sentenceB}: {source_ex[i]}\n" \
                               f"Question: {question}\n" \
                               f"Answer: ({grades[result_ex[i]]})\n"
                else:
                    example += f"Question: {question}\n" \
                               f"{sentenceA}: {production_ex[i]}\n" \
                               f"{sentenceB}: {source_ex[i]}\n" \
                               f"Answer: ({grades[result_ex[i]]})\n"
        source_text = (f"{sentenceB}: {source}\n" if len(source) > 0 else "")
        context = ""
        if config["context"]:
            context = f"Context: {class_definition}\n"

        if config["S1S2before"]:
            return f"{context}" \
                   f"{example}" \
                   f"{sentenceA}: {production}\n" \
                   f"{source_text}" \
                   f"Question: {question}\n" \
                   f"Options:\n{options}\n" \
                   f"Answer: "
        else:
            return f"{context}" \
                   f"{example}" \
                   f"Question: {question}\n" \
                   f"{sentenceA}: {production}\n" \
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


def get_data(df_test, df_train, task_df_label, task_name, num_examples, config):
    sentences = []
    for index, line in df_test.iterrows():
        source, prod, label = get_examples(df_train, task_df_label, seed=index, num_examples=num_examples)
        sentences.append(get_prompt(0, task_name, PromptUtils.class_definitions[task_df_label],
                                    PromptUtils.shortened_class_item_meaning_dict[task_df_label],
                                    line['Source'], line['Production'],
                                    source, prod, label,
                                    config))
    targets = df_test[task_df_label].values.tolist()

    return sentences, targets


def get_best_config():
    return {'context': False, "S1S2notSource": True, "numberingAlpha": True, "S1S2before": False}


def process_score(grades, text, optimistic=True):
    not_found = 0 if optimistic else -1
    start = text.find("(")
    if start < 0 or len(text[start:]) < 2:
        return not_found
    else:
        output_letter = text[start:][1]
    return grades.index(output_letter) if output_letter in grades else not_found


def get_targets_and_preds(predictions_raw, targets, grades, targets_raw_flag=False, is_optimistical=False):
    predictions = [process_score(grades, x, is_optimistical) for x in predictions_raw]
    if targets_raw_flag:
        targets = [process_score(grades, x, True) for x in targets]

    if not is_optimistical:
        skip_indices = [i for i, x in enumerate(predictions) if x < 0]
        res_targets = [targets[i] for i in range(len(targets)) if i not in skip_indices]
        res_predictions = [predictions[i] for i in range(len(targets)) if i not in skip_indices]
    else:
        res_targets = targets
        res_predictions = predictions
    return res_targets, res_predictions


class PromptUtils:
    class_item_meaning_dict = {
        SelfExplanations.PARAPHRASE: {
            0: "not present",
            1: "contains at least one full clause that overlaps; may include single words; contains some of (about half) of the main idea units from the target sentence",
            2: "contains most of (50% or more) of the main idea units from the target sentence"
        },
        SelfExplanations.BRIDGING: {
            0: "not present",
            1: "includes anaphoric reference or one or two words from prior text",
            2: "includes one or more complete ideas from previous ideas in the text (an idea is not necessarily a sentence, but it is a complete idea unit)"
        },
        SelfExplanations.ELABORATION: {
            0: "No apparent connection between text and elaborated ideas (random ideas)",
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