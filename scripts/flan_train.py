import random
import logging
import wandb
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from datasets import Dataset
from transformers.utils import logging as transf_logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, AdapterTrainer
from transformers.adapters import LoRAConfig

from core.data_processing.flan_data_processing import get_best_config, get_data, get_new_train_test_split, \
    get_targets_and_preds
from core.data_processing.se_dataset import SelfExplanations
from transformers.trainer_callback import PrinterCallback

logging.basicConfig(level=logging.INFO)

# tokenize the dataset
def encode_batch(examples):
    # the name of the input column
    text_column = 'prompt'
    # the name of the target column
    summary_column = 'label'
    # used to format the tokens
    padding = "max_length"

    # convert to lists of strings
    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])

    # finally we can tokenize the inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)
    labels = tokenizer(targets, max_length=512, padding=padding, truncation=True)

    # rename to labels for training
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# load the dataset
def load_split(train_data, train_labels, split_name):
    dataset = Dataset.from_dict({
        "prompt": train_data,
        "label": [["(A)", "(B)", "(C)", "(D)"][x] for x in train_labels]
    })

    dataset = dataset.map(
        encode_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on " + split_name + " dataset",
    )
    # set the format to torch
    dataset.set_format(type="torch", columns=["input_ids", "labels"])

    return dataset


def get_batch_size(flan_size, num_examples):
    index = ["small", "base", "large", "xl", "xxl"].index(flan_size)
    increments = [1, 3, 10, 40, 160]
    example_increment = num_examples + 1

    return max(1, int(20 / increments[index] / example_increment))


if __name__ == '__main__':
    STUDY_NAME = "rb_feats_importance_none"
    PROJECT = "optuna-a100"
    ENTITY = "bogdan-nicula22"
    wandb.init(
        project=PROJECT,
        entity=ENTITY,  # NOTE: this entity depends on your wandb account.
        config={},
        group=STUDY_NAME,
        reinit=True,
    )
    logging.info("Starting program")
    self_explanations = SelfExplanations()
    logging.info("Loading SEs")
    self_explanations.parse_se_from_csv("../data/results_se_aggregated_dataset_clean.csv")
    logging.info("Loaded SEs")

    random.seed(13)
    sentence_mode = "target"

    df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df, sentence_mode)
    for flan_size in ["small", "base", "large", "xl", "xxl"]:
        for num_examples in [2]:
            batch_size = get_batch_size(flan_size, num_examples)
            logging.info("=" * 33)
            logging.info(f"Starting {flan_size} - {num_examples} - {batch_size}")
            logging.info("=" * 33)
            for num_classes, task_name, task_df_label in [
                (3, "paraphrasing", SelfExplanations.PARAPHRASE),
                (3, "elaboration", SelfExplanations.ELABORATION),
                (2, "bridging", SelfExplanations.BRIDGING),
                (4, "overall", SelfExplanations.OVERALL),
            ]:
                config = get_best_config()
                logging.info("Generating training data %d", len(df_train))
                sentences_train, targets_train = get_data(df_train, df_train, task_df_label, task_name, num_examples, config)
                logging.info("Generating dev data %d", len(df_dev))
                sentences_dev, targets_dev = get_data(df_dev, df_train, task_df_label, task_name, num_examples, config)
                logging.info("Generating test data %d", len(df_test))
                sentences_test, targets_test = get_data(df_test, df_train, task_df_label, task_name, num_examples, config)

                # sentences_train = sentences_train[:800]
                # targets_train = targets_train[:800]
                # sentences_test = sentences_test[:1000]
                # targets_test = targets_test[:1000]
                # the base model that we'll be using
                base_model = f"google/flan-t5-{flan_size}"
                # the tokenizer that we'll be using
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                # start with the pretrained base model
                model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
                # set the parameters for LoRA
                lora_config = LoRAConfig(r=8, alpha=16, intermediate_lora=True, output_lora=True)

                # make a new adapter for the XSum dataset
                model.add_adapter("xsum", config=lora_config)
                # enable the adapter for training
                model.train_adapter("xsum")
                model.set_active_adapters(["xsum"])

                training_args = TrainingArguments(
                    report_to="none",
                    learning_rate=3e-4,
                    num_train_epochs=1,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    logging_steps=200,
                    output_dir="./training_output",
                    overwrite_output_dir=True,
                    remove_unused_columns=False
                )

                # create the trainer
                trainer = AdapterTrainer(
                    model=model,
                    args=training_args,
                    tokenizer=tokenizer,
                    # load the dataset
                    train_dataset=load_split(sentences_train, targets_train, "train"),
                    eval_dataset=load_split(sentences_dev, targets_dev, "dev"),
                )
                trainer.remove_callback(PrinterCallback)
                trainer.train()
                # trainer.evaluate()
                # merge the adapter with the model
                # this will add the adapter weight matrices to the model weight matrices
                model.merge_adapter("xsum")
                num_validation = 10


                grades = ["A", "B", "C", "D"]
                validation_dataset = load_split(sentences_test, targets_test, "test")
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