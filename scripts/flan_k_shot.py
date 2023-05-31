import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from core.data_processing.se_dataset import SelfExplanations
from scripts.mtl_bert_train import get_new_train_test_split, get_filterable_cols


def load_model(model_size):
    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{flan_size}", device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{flan_size}")

    return model, tokenizer

def batch_eval(model, tokenizer, sentences, batch_size=256, targets=[]):
    predictions = []

    for i in range(1 + len(sentences) // batch_size):
        inputs = tokenizer(
            sentences[i*batch_size: (i+1) * batch_size],
            return_tensors="pt",
            padding=True)

        outputs = model.generate(**inputs)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result = [int(x) if x.isnumeric() else 0 for x in result]
        print(len(result))
        predictions += result
    print("task", classification_report(targets, predictions))
    k = 2

if __name__ == '__main__':
    task_name = "elaboration"
    num_classes = 2

    flan_size = "small"
    target_sentence_mode = "target"
    model, tokenizer = load_model(flan_size)

    self_explanations = SelfExplanations()
    self_explanations.parse_se_from_csv("../data/results_paraphrase_se_aggregated_dataset_2.csv")

    df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df, target_sentence_mode)

    filterable_cols = get_filterable_cols(df_train) + ["EntryType"]
    df_train = df_train.drop(filterable_cols, axis=1, inplace=False)
    df_dev = df_dev.drop(filterable_cols, axis=1, inplace=False)
    df_test = df_test.drop(filterable_cols, axis=1, inplace=False)
    feature_columns = df_test.columns.tolist()[114:]

    targets = []
    for num_classes, task_name, task_df_label in [
        (3, "paraphrasing", SelfExplanations.PARAPHRASE),
        (3, "elaboration", SelfExplanations.ELABORATION),
        (2, "bridging", SelfExplanations.BRIDGING),
        (4, "self-explanation", SelfExplanations.OVERALL),
    ]:
        task_prefix = f"Evaluate the presence of textual {task_name} between the source and target text with a value between 0 and {num_classes - 1}?\n"
        sentences = []
        for index, line in df_test.iterrows():
            sentences.append(f"{task_prefix}source: {line['Source']}\ntarget: {line['Production']}\n")
        targets = df_test[task_df_label].values.tolist()

        batch_eval(model, tokenizer, sentences, batch_size=256, targets=targets)