import torch
import pytorch_lightning as pl
from transformers import BertTokenizer, RobertaTokenizer

from core.data_processing.se_dataset import SelfExplanations, create_data_loader
from core.models.mtl import BERTMTL
from scripts.mtl_bert_train import get_new_train_test_split, get_filterable_cols


if __name__ == '__main__':
    num_tasks = 4
    predefined_version = ""
    PRE_TRAINED_MODEL_NAME = 'roberta-base'
    MAX_LEN_P = 80
    BATCH_SIZE = 128
    if "roberta" in PRE_TRAINED_MODEL_NAME:
        tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    else:
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    self_explanations = SelfExplanations()
    target_sent_enhanced = self_explanations.parse_se_from_csv(
        "../data/results_se_aggregated_dataset_clean.csv")

    df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df)

    filterable_cols = get_filterable_cols(df_train)
    df_train = df_train.drop(filterable_cols, axis=1, inplace=False)
    df_dev = df_dev.drop(filterable_cols, axis=1, inplace=False)
    df_test = df_test.drop(filterable_cols, axis=1, inplace=False)

    val_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=True, use_filtering=True)
    # train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=True, use_filtering=True)

    task_names = [SelfExplanations.MTL_TARGETS[task_id] for task_id in range(num_tasks)]
    model = BERTMTL(task_names, PRE_TRAINED_MODEL_NAME, rb_feats=val_data_loader.dataset.rb_feats.shape[1], use_filtering=True)

    model_path = "./mtl/lightning_logs/version_24/checkpoints/epoch=21-step=1848.ckpt"

    # Note: always make sure that the params are the same as in the training run.
    model = model.load_from_checkpoint(model_path,
                                       task_names=task_names,
                                       pretrained_bert_model=PRE_TRAINED_MODEL_NAME,
                                       rb_feats=val_data_loader.dataset.rb_feats.shape[1],
                                       use_filtering=True,
                                       use_grad_norm=True,
                                       task_imp_weights=[2, 2, 1, 5],
                                       hidden_units=175)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=1)
    # trainer.test(model, dataloaders=train_data_loader)
    trainer.test(model, dataloaders=val_data_loader)
