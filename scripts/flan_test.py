import random
import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import login as login_hf

from scripts.flan_train import validate
from core.data_processing.flan_data_processing import get_best_config, get_data, get_new_train_test_split
from core.data_processing.se_dataset import SelfExplanations

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    login_hf(token="secret")
    logging.info("Starting program")
    self_explanations = SelfExplanations()
    logging.info("Loading SEs")
    self_explanations.parse_se_from_csv("../data/results_se_aggregated_dataset_clean.csv")
    logging.info("Loaded SEs")

    random.seed(13)
    sentence_mode = "target"

    df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df, sentence_mode)
    df_train = df_train[:1] # ignoring the training subset
    df_dev = df_dev[:100]
    df_test = df_test[:100]
    # for flan_size in ["small", "base", "large", "xl", "xxl"]:
    for flan_size in ["small"]:
        for num_examples in [0, 1, 2]:
            batch_size = 1
            logging.info("=" * 33)
            logging.info(f"Starting {flan_size} - {num_examples} - {batch_size}")
            logging.info("=" * 33)
            for num_classes, task_name, task_df_label in [
                (4, "overall", SelfExplanations.OVERALL),
                (3, "paraphrasing", SelfExplanations.PARAPHRASE),
                (3, "elaboration", SelfExplanations.ELABORATION),
                (2, "bridging", SelfExplanations.BRIDGING),
            ]:
                adapter_name = f"flant5-{flan_size}-{batch_size}bs-{num_examples}ex-{task_name}"
                adapter_name = "nbogdan/flant5-small-1bs-0ex-overall"
                config = get_best_config()

                logging.info("Generating dev data %d", len(df_dev))
                sentences_dev, targets_dev = get_data(df_dev, df_train, task_df_label, task_name, num_examples, config)
                logging.info("Generating test data %d", len(df_test))
                sentences_test, targets_test = get_data(df_test, df_train, task_df_label, task_name, num_examples, config)

                base_model = f"google/flan-t5-{flan_size}"
                # the tokenizer that we'll be using
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                # start with the pretrained base model
                model = AutoModelForSeq2SeqLM.from_pretrained(base_model, device_map="auto")

                remote_adapter_name = model.load_adapter(adapter_name)
                model.set_active_adapters([remote_adapter_name])

                validate(model, sentences_test, targets_test, tokenizer)