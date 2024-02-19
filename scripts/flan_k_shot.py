import logging
import random
import torch
import pickle
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from core.data_processing.se_dataset import SelfExplanations
from core.data_processing.flan_data_processing import get_new_train_test_split, get_prompt, get_examples, PromptUtils, \
    get_best_config, get_data, get_targets_and_preds
from core.data_processing.se_dataset import load_split

logging.basicConfig(level=logging.NOTSET)


def load_model(flan_size):
    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{flan_size}", device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{flan_size}")

    return model, tokenizer


def batch_eval(model, tokenizer, sentences, batch_size=256, targets=[], task_name="", details="", config={}):
    predictions = []
    if config["numberingAlpha"]:
        grades = ["A", "B", "C", "D"]
    else:
        grades = ["1", "2", "3", "4"]
    for i in range(1 + len(sentences) // batch_size):
        inputs = tokenizer(
            sentences[i*batch_size: (i+1) * batch_size],
            return_tensors="pt",
            padding=True)
        outputs = model.generate(**inputs, max_length=4)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if i % 50 == 0:
            logging.info(f"Seen {i} batches.")
            logging.info(sentences[i*batch_size])
            logging.info(result[i*batch_size])
            logging.info(f"target: {targets[i*batch_size]}")
        predictions += result

    logging.info("=" * 33)
    logging.info(predictions)
    logging.info(targets)
    targets_opt, preds_opt = get_targets_and_preds(predictions, targets, grades, targets_raw_flag=False, is_optimistical=True)
    targets_opt = np.array(targets_opt)
    preds_opt = np.array(preds_opt)
    logging.info(f"Optimistic estimation")
    logging.info(f"task:{task_name} details:opt-{details} f1:{f1_score(targets_opt, preds_opt, average='weighted')}")
    logging.info(classification_report(targets_opt, preds_opt))
    logging.info(confusion_matrix(targets_opt, preds_opt))
    logging.info("=" * 33)
    targets_pes, preds_pes = get_targets_and_preds(predictions, targets, grades, targets_raw_flag=False, is_optimistical=False)
    targets_pes = np.array(targets_pes)
    preds_pes = np.array(preds_pes)
    logging.info(f"Pessimistic estimation")
    logging.info(f"task:{task_name} details:pes-{details} f1:{f1_score(targets_pes, preds_pes, average='weighted')}")
    logging.info(classification_report(targets_pes, preds_pes))
    logging.info(confusion_matrix(targets_pes, preds_pes))
    logging.info("=" * 33)
    logging.info(f"Sentences: {len(sentences)}\tOptimistic: {len(targets_opt)}\tPessimistic: {len(targets_pes)}\tPerc: {100.0 * len(targets_pes) / len(sentences)}")
    logging.info("=" * 33)


if __name__ == '__main__':
    logging.info("Starting program")
    self_explanations = SelfExplanations()
    logging.info("Loading SEs")
    self_explanations.parse_se_from_csv("../data/dataset_clean.csv")
    logging.info("Loaded SEs")

    for flan_size in ["small", "base", "large", "xl", "xxl"]:
        model, tokenizer = load_model(flan_size)
        logging.info("Loaded model")
        for sentence_mode in ["target"]:
            df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df, sentence_mode)
            for num_examples in [0, 1, 2]:
                config = get_best_config()
                random.seed(13)
                logging.info("$" + "=" * 33)
                logging.info(f">>Model:{flan_size} sentence_mode:{sentence_mode} num_examples:{num_examples}")
                for num_classes, task_name, task_df_label in [
                    (3, "elaboration", SelfExplanations.ELABORATION),
                    (2, "bridging", SelfExplanations.BRIDGING),
                    (4, "overall", SelfExplanations.OVERALL),
                    (3, "paraphrasing", SelfExplanations.PARAPHRASE),
                ]:
                    sentences, targets = get_data(df_test, df_train, task_df_label, task_name, num_examples, config)
                    bs = 64
                    if model == "large":
                        bs = 32
                    elif model == "xl":
                        bs = 4
                    elif model == "xxl":
                        bs = 1

                    config = get_best_config()
                    with open('data.pickle', 'rb') as f:
                        subset2 = pickle.load(f)

                    batch_eval(model, tokenizer, sentences, batch_size=bs, targets=targets, task_name=task_name,
                               details=f"{flan_size}|{sentence_mode}|{num_examples}|{config}", config=config)