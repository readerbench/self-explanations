import os.path
import wandb
import optuna
import random

import pandas as pd
import numpy as np
import pytorch_lightning as pl
from transformers import BertTokenizer, RobertaTokenizer
import torch
import transformers

from core.models.mtl import BERTMTL
from core.data_processing.se_dataset import SelfExplanations, create_data_loader
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

transformers.logging.set_verbosity_error()

def map_train_test(x):
    if x['Dataset'] in ['ASU 5']:
        return 'test'
    if x['Dataset'] == 'CRaK':
        return 'train'
    if x['Dataset'] == 'ASU 1':
        if x['PrePost'] == 'Post':
            return 'dev'
        return 'train'
    if x['Dataset'] == 'ASU 4':
        if not str(x['ID']).startswith('ISTARTREF'):
            return 'train'
        else:
            return 'dev'
    return 'dump'


def get_new_train_test_split(df):
    df['EntryType'] = df.apply(lambda x: map_train_test(x), axis=1)

    df.loc[df[SelfExplanations.ELABORATION] == 2, SelfExplanations.ELABORATION] = 1
    df.loc[df[SelfExplanations.BRIDGING] == 3, SelfExplanations.BRIDGING] = 2


    return df[(df['EntryType'] == 'train') | (df['EntryType'] == 'dev')], df[df['EntryType'] == 'dev'], df[df['EntryType'] == 'test']
    # return df[(df['EntryType'] == 'train')], df[df['EntryType'] == 'dev'], df[df['EntryType'] == 'test']


def filter_rb_df(df):
    filterable_df = df[df.columns[114:-1]]
    # print("original len", len(filterable_df.columns))
    filterable_df = filterable_df.loc[:, filterable_df.apply(pd.Series.nunique) != 1]
    # print("rem constant", len(filterable_df.columns))
    filterable_df = filterable_df._get_numeric_data()
    # print("rem garbage vals", len(filterable_df.columns))
    cor_matrix = np.corrcoef(filterable_df.values, rowvar=False)
    cor_matrix = np.abs(cor_matrix)

    a = np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool)
    to_drop = []
    for i in range(len(cor_matrix[0])):
        upper = cor_matrix[:, i][a[:, i]]
        if any(upper > 0.95):
            to_drop.append(i)

    filterable_df.drop(filterable_df.columns[to_drop], axis=1, inplace=True)
    # print("rem intercorr vals", len(filterable_df.columns))

    filterable_df_orig = df[df.columns[114:-1]]
    cols_to_drop = [c for c in filterable_df_orig.columns if c not in filterable_df.columns]
    # print("total cols to drop", len(cols_to_drop))

    return df.drop(cols_to_drop, axis=1, inplace=False)


def experiment(task_imp_weights=[], bert_model="bert-base-cased", lr=1e-3, num_epochs=20, task_name="none",
               use_filtering=True, use_grad_norm=True, trial=None, hidden_units=100):
    if task_name == "none":
        num_tasks = 4
    else:
        num_tasks = 1
    predefined_version = ""
    MAX_LEN_P = 80
    BATCH_SIZE = 128
    if "roberta" in bert_model:
        tokenizer = RobertaTokenizer.from_pretrained(bert_model)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model)

    self_explanations = SelfExplanations()
    self_explanations.parse_se_from_csv("../data/results_paraphrase_se_aggregated_dataset_2.csv")

    self_explanations.df = filter_rb_df(self_explanations.df)

    df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df)
    print(f"len train {len(df_train)}")
    print(f"len dev {len(df_dev)}")
    print(f"len test {len(df_test)}")

    # toggle 0 or 1 for using rb_features
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=True,
                                           task_name=task_name, use_filtering=use_filtering)
    val_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=True,
                                         task_name=task_name, use_filtering=use_filtering)
    rb_feats = train_data_loader.dataset.rb_feats.shape[1]
    task_sample_weights = []
    if num_tasks > 1:
        task_names = [SelfExplanations.MTL_TARGETS[task_id] for task_id in range(num_tasks)]
        for task in range(num_tasks):
            df_aux = df_train[df_train[SelfExplanations.MTL_TARGETS[task]] < 9]
            values = df_aux[SelfExplanations.MTL_TARGETS[task]].value_counts()
            total = len(df_aux[SelfExplanations.MTL_TARGETS[task]]) * 1.0
            task_sample_weights.append(torch.Tensor([total / values[i] if i in values else 0 for i in range(SelfExplanations.MTL_CLASS_DICT[SelfExplanations.MTL_TARGETS[task]])]))
    else:
        task_names = [task_name]
        df_aux = df_train[df_train[task_name] < 9]
        values = df_aux[task_name].value_counts()
        total = len(df_aux[task_name]) * 1.0
        task_sample_weights.append(torch.Tensor([total / values[i] if i in values else 0 for i in range(SelfExplanations.MTL_CLASS_DICT[task_name])]))

    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="test_loss", every_n_epochs=10)
    model = BERTMTL(task_names, bert_model, rb_feats=rb_feats, task_sample_weights=task_sample_weights,
                    task_imp_weights=task_imp_weights, lr=lr, num_epochs=num_epochs, use_filtering=use_filtering,
                    use_grad_norm=use_grad_norm, trial=trial, hidden_units=hidden_units)

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=WandbLogger(),
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        # limit_train_batches=10,
        max_epochs=num_epochs)
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
    return model.last_loss

def objective(trial):
    class_weighting = trial.suggest_categorical("class_weighting", ["[1,1,1,1]", "[1,1,1,3]", "[2,2,1,5]"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_units = trial.suggest_int("hidden_units", 50, 200, step=50)
    # filtering = trial.suggest_categorical("filtering", ["true", "false"])
    grad_norm = trial.suggest_categorical("grad_norm", ["true", "false"])

    config = dict(trial.params)
    config["trial.number"] = trial.number
    wandb.init(
        project="optuna",
        entity="bogdan-nicula22",  # NOTE: this entity depends on your wandb account.
        config=config,
        group="param-search",
        reinit=True,
    )
    loss = experiment([int(c) for c in class_weighting[1:-1].split(",")], bert_model="roberta-base",
                      lr=lr, num_epochs=30, use_grad_norm=grad_norm, use_filtering=True, trial=trial, hidden_units=hidden_units)


    # report the final validation accuracy to wandb
    wandb.run.summary["final loss"] = loss
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)

    return loss



if __name__ == '__main__':
    study = optuna.create_study(
        direction="minimize",
        study_name="param-search-study",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=25, timeout=None)

    # print("=" * 33)
    # experiment([2, 2, 1, 5], bert_model="roberta-base", lr=2e-4, num_epochs=25, use_grad_norm=True, use_filtering=False)
    # print("=" * 33)
    # experiment([1, 1, 1, 1], bert_model="roberta-base", lr=2e-4, num_epochs=25, use_grad_norm=True, use_filtering=True)
