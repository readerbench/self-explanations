import random
import logging

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers.trainer_callback import PrinterCallback
from huggingface_hub import login as login_hf
from transformers.utils import logging as transf_logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, AdapterTrainer, TrainerCallback
from transformers.adapters import LoRAConfig

from core.data_processing.flan_data_processing import get_best_config, get_data, get_new_train_test_split, \
    get_targets_and_preds
from core.data_processing.se_dataset import SelfExplanations, load_split

logging.basicConfig(level=logging.INFO)


# validate trained model
def validate(model, sentences_test, targets_test, tokenizer):
    grades = ["A", "B", "C", "D"]
    validation_dataset = load_split(sentences_test, targets_test, "test", tokenizer)
    logging.info("Validating results: %d", len(validation_dataset))
    logging.info(transf_logging.get_logger('transformers.generation.configuration_utils'))
    transf_logging.get_logger('transformers.generation.configuration_utils').setLevel(transf_logging.ERROR)

    targets = []
    predictions = []
    for i in range(len(validation_dataset)):
        # load the input and label
        input_ids = validation_dataset[i]['input_ids'].unsqueeze(0).to(0)
        label_ids = validation_dataset[i]['labels'].unsqueeze(0).to(0)
        # use the model to generate the output
        output = model.generate(input_ids, max_length=15)
        # convert the tokens to text
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        label_text = tokenizer.decode(label_ids[0], skip_special_tokens=True)
        targets.append(label_text)
        predictions.append(output_text)
        if i % 200 == 0:
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            logging.info(f"Seen {i} batches.")
            logging.info(f"input: {input_text}")
            logging.info(f"output: {output_text}")
            logging.info(f"label: {label_text}")

    transf_logging.get_logger('transformers.generation.configuration_utils').setLevel(transf_logging.INFO)
    logging.info("=" * 33)
    logging.info(predictions)
    logging.info(targets)
    targets_opt, preds_opt = get_targets_and_preds(predictions, targets, grades, targets_raw_flag=True,
                                                   is_optimistical=True)
    targets_opt = np.array(targets_opt)
    preds_opt = np.array(preds_opt)
    logging.info(f"Optimistic estimation")
    logging.info(
        f"task:{task_name} details:opt-{flan_size}-{num_examples} f1:{f1_score(targets_opt, preds_opt, average='weighted')}")
    logging.info(classification_report(targets_opt, preds_opt))
    logging.info(confusion_matrix(targets_opt, preds_opt))
    logging.info("=" * 33)
    targets_pes, preds_pes = get_targets_and_preds(predictions, targets, grades, targets_raw_flag=True,
                                                   is_optimistical=False)
    targets_pes = np.array(targets_pes)
    preds_pes = np.array(preds_pes)
    logging.info(f"Pessimistic estimation")
    logging.info(
        f"task:{task_name} details:pes-{flan_size}-{num_examples} f1:{f1_score(targets_pes, preds_pes, average='weighted')}")
    logging.info(classification_report(targets_pes, preds_pes))
    logging.info(confusion_matrix(targets_pes, preds_pes))
    logging.info("=" * 33)
    logging.info(
        f"Sentences: {len(validation_dataset)}\tOptimistic: {len(targets_opt)}\tPessimistic: {len(targets_pes)}\tPerc: {100.0 * len(targets_pes) / len(validation_dataset)}")
    logging.info("=" * 33)


if __name__ == '__main__':
    upload_adapter = True
    batch_size = 1
    epochs = 1

    login_hf(token="secret")

    logging.info("Starting program")
    self_explanations = SelfExplanations()
    logging.info("Loading SEs")
    self_explanations.parse_se_from_csv("../data/dataset_clean.csv")
    logging.info("Loaded SEs")

    random.seed(13)
    sentence_mode = "target"

    df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df, sentence_mode)

    for flan_size in ["small", "base", "large", "xl", "xxl"]:
        for num_examples in [0, 1, 2]:
            logging.info("=" * 33)
            logging.info(f"Starting {flan_size} - {num_examples} - {batch_size}")
            logging.info("=" * 33)
            for num_classes, task_name, task_df_label in [
                (4, "overall", SelfExplanations.OVERALL),
                (3, "paraphrasing", SelfExplanations.PARAPHRASE),
                (3, "elaboration", SelfExplanations.ELABORATION),
                (2, "bridging", SelfExplanations.BRIDGING),
            ]:
                adapter_name = f"flant5-{flan_size}-{num_examples}ex-{task_name}-{epochs}epochs"
                config = get_best_config()

                logging.info("Generating training data %d", len(df_train))
                sentences_train, targets_train = get_data(df_train, df_train, task_df_label, task_name, num_examples, config)
                logging.info("Generating dev data %d", len(df_dev))
                sentences_dev, targets_dev = get_data(df_dev, df_train, task_df_label, task_name, num_examples, config)
                logging.info("Generating test data %d", len(df_test))
                sentences_test, targets_test = get_data(df_test, df_train, task_df_label, task_name, num_examples, config)

                # the base model that we'll be using
                base_model = f"google/flan-t5-{flan_size}"
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                model = AutoModelForSeq2SeqLM.from_pretrained(base_model, device_map="auto")
                lora_config = LoRAConfig(r=8, alpha=16, intermediate_lora=True, output_lora=True)

                # make a new adapter
                model.add_adapter(adapter_name, config=lora_config)
                model.train_adapter(adapter_name)
                model.set_active_adapters([adapter_name])

                training_args = TrainingArguments(
                    report_to="none",
                    learning_rate=3e-4,
                    num_train_epochs=epochs,
                    # evaluation_strategy="epoch",
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    logging_steps=200,
                    output_dir="./training_output",
                    overwrite_output_dir=True,
                    remove_unused_columns=False
                )

                trainer = AdapterTrainer(
                    model=model,
                    args=training_args,
                    tokenizer=tokenizer,
                    train_dataset=load_split(sentences_train, targets_train, "train", tokenizer),
                    eval_dataset=load_split(sentences_dev, targets_dev, "dev", tokenizer),
                )
                trainer.remove_callback(PrinterCallback)
                trainer.train()

                if upload_adapter:
                    logging.info("Uploading adapter.")
                    model.push_adapter_to_hub(
                        adapter_name,
                        adapter_name,
                        adapterhub_tag="self-explanations",
                        datasets_tag="self-explanations"
                    )

                # merge the adapter with the model
                model.merge_adapter(adapter_name)

                validate(model, sentences_test, targets_test, tokenizer)