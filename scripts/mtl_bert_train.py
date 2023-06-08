import wandb
import optuna

import pandas as pd
import numpy as np
import pytorch_lightning as pl
from transformers import BertTokenizer, RobertaTokenizer
import torch
import transformers

from core.models.mtl import BERTMTL
from core.data_processing.se_dataset import SelfExplanations, create_data_loader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

transformers.logging.set_verbosity_error()

STUDY_NAME = "rb_feats_importance_none"
PROJECT = "optuna-a100"
ENTITY = "bogdan-nicula22"


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

    # Train on train+dev
    return df[(df['EntryType'] == 'train') | (df['EntryType'] == 'dev')], df[df['EntryType'] == 'dev'], df[df['EntryType'] == 'test']
    # Train on train
    # return df[(df['EntryType'] == 'train')], df[df['EntryType'] == 'dev'], df[df['EntryType'] == 'test']


def get_filterable_cols(df):
    """
    Compiles a list of columns which can be dropped.
    If 2 columns have strong intercorrelation (> .95), one will be dropped.
    :param df:
    :return:
    """
    filterable_df = df[df.columns[114:-1]]
    print("Original number of columns", len(filterable_df.columns))
    filterable_df = filterable_df.loc[:, filterable_df.apply(pd.Series.nunique) != 1]
    filterable_df = filterable_df._get_numeric_data()
    cor_matrix = np.corrcoef(filterable_df.values, rowvar=False)
    cor_matrix = np.abs(cor_matrix)

    a = np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool)
    to_drop = []
    for i in range(len(cor_matrix[0])):
        upper = cor_matrix[:, i][a[:, i]]
        if any(upper > 0.95):
            to_drop.append(i)

    filterable_df.drop(filterable_df.columns[to_drop], axis=1, inplace=True)

    filterable_df_orig = df[df.columns[114:-1]]
    cols_to_drop = [c for c in filterable_df_orig.columns if c not in filterable_df.columns]
    print("Total columns to drop", len(cols_to_drop))

    return cols_to_drop


def experiment(task_imp_weights=[], bert_model="bert-base-cased", lr=1e-3, num_epochs=20, task_name="none",
               use_filtering=True, use_grad_norm=True, trial=None, hidden_units=100, lr_warmup=5, target_sentence_mode="none", use_rb_feats=True):
    assert target_sentence_mode in ["none", "target", "targetprev"], f"Invalid target_sentence_mode {target_sentence_mode}"

    if task_name == "none":
        num_tasks = 4
    else:
        num_tasks = 1

    MAX_LEN_P = 80
    BATCH_SIZE = 128
    if "roberta" in bert_model:
        tokenizer = RobertaTokenizer.from_pretrained(bert_model)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model)

    self_explanations = SelfExplanations()
    self_explanations.parse_se_from_csv("../data/results_se_aggregated_dataset_clean.csv")

    df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df, target_sentence_mode)

    filterable_cols = get_filterable_cols(df_train)
    df_train = df_train.drop(filterable_cols, axis=1, inplace=False)
    df_dev = df_dev.drop(filterable_cols, axis=1, inplace=False)
    df_test = df_test.drop(filterable_cols, axis=1, inplace=False)

    # toggle 0 or 1 for using rb_features
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=use_rb_feats,
                                           task_name=task_name, use_filtering=use_filtering)
    val_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=use_rb_feats,
                                         task_name=task_name, use_filtering=use_filtering)
    rb_feats = train_data_loader.dataset.rb_feats.shape[1] if use_rb_feats else 0
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

    checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="test_loss", every_n_epochs=1)
    model = BERTMTL(task_names, bert_model, rb_feats=rb_feats, task_sample_weights=task_sample_weights,
                    task_imp_weights=task_imp_weights, lr=lr, num_epochs=num_epochs, use_filtering=use_filtering,
                    use_grad_norm=use_grad_norm, trial=trial, hidden_units=hidden_units, lr_warmup=lr_warmup)

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=WandbLogger(),
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        # limit_train_batches=10, # Limits an epoch to 10 minibatches. Uncomment for faster 'pipe cleaning' runs.
        max_epochs=num_epochs)
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
    return model.last_loss


def objective(trial, target_sentence_mode):
    # Note: the functionality used to explore the parameter space is commented out
    class_weighting = "[2, 2, 1, 5]"  # trial.suggest_categorical("class_weighting", ["[1,1,1,1]", "[1,1,1,3]", "[2,2,1,5]"])
    target_sentence_mode = target_sentence_mode   # trial.suggest_categorical("target_sentence_mode", ["none", "target", "targetprev"])
    lr = trial.suggest_float("lr", 1e-4, 3e-4, log=True)
    lr_warmup = 6  # trial.suggest_int("lr_warmup", 5, 10, step=1)
    hidden_units = 175  # trial.suggest_int("hidden_units", 125, 200, step=25)
    filtering = "true"  # trial.suggest_categorical("filtering", ["true", "false"])
    grad_norm = "true"  # trial.suggest_categorical("grad_norm", ["true", "false"])
    rb_feats = "true"

    config = dict(trial.params)
    config["trial.number"] = trial.number
    config["filtering"] = filtering
    config["grad_norm"] = grad_norm
    config["rb_feats"] = rb_feats
    wandb.init(
        project=PROJECT,
        entity=ENTITY,  # NOTE: this entity depends on your wandb account.
        config=config,
        group=STUDY_NAME,
        reinit=True,
    )
    loss = experiment([int(c) for c in class_weighting[1:-1].split(",")], bert_model="roberta-base",
                      lr=lr, num_epochs=25, use_grad_norm=grad_norm == "true", use_filtering=filtering == "true", trial=trial,
                      hidden_units=hidden_units, lr_warmup=lr_warmup, target_sentence_mode=target_sentence_mode, use_rb_feats=rb_feats == "true")

    # report the final validation accuracy to wandb
    wandb.run.summary["final loss"] = loss
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)

    return loss


def legacy_exp(single_task=False):
    """
    Runs the legacy experiments from the AIED paper.
    :param single_task:
    :return:
    """
    if not single_task:
        config = {"trial.number": -1}
        wandb.init(
            project=PROJECT,
            entity=ENTITY,  # NOTE: this entity depends on your wandb account.
            config=config,
            group=STUDY_NAME,
            reinit=True,
        )

        loss = experiment([2, 2, 1, 5], bert_model="roberta-base", lr=2e-4, num_epochs=30, use_grad_norm=False,
                          use_filtering=True, trial=None, hidden_units=100, lr_warmup=7, target_sentence_mode="target", use_rb_feats=True)

        # report the final validation accuracy to wandb
        wandb.run.summary["final loss"] = loss
        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)
    else:
        for i in range(4):
            config = {"trial.number": -10 * i}
            wandb.init(
                project=PROJECT,
                entity=ENTITY,  # NOTE: this entity depends on your wandb account.
                config=config,
                group=STUDY_NAME,
                reinit=True,
            )
            task_name = SelfExplanations.MTL_TARGETS[i]
            loss = experiment([], bert_model="roberta-base", lr=2e-4, num_epochs=30, use_grad_norm=False,
                              use_filtering=True, trial=None, hidden_units=100, lr_warmup=7, task_name=task_name, target_sentence_mode="target", use_rb_feats=True)

            # report the final validation accuracy to wandb
            wandb.run.summary["final loss"] = loss
            wandb.run.summary["state"] = "completed"
            wandb.finish(quiet=True)


def best_so_far():
    options = [
        {'split': [2, 2, 1, 5], 'lr': 0.000101, 'hidden_units': 175, 'id': -3},
        {'split': [2, 2, 1, 5], 'lr': 127e-6, 'hidden_units': 125, 'id': -2}
    ]
    for option in options:
        config = {"trial.number": option['id']}
        wandb.init(
            project=PROJECT,
            entity=ENTITY,  # NOTE: this entity depends on your wandb account.
            config=config,
            group=STUDY_NAME,
            reinit=True,
        )
        loss = experiment(option['split'], bert_model="roberta-base", lr=option['lr'], num_epochs=25, use_grad_norm=True,
                          use_filtering=True, trial=None, hidden_units=option['hidden_units'], lr_warmup=60, target_sentence_mode="target", use_rb_feats=True)

        # report the final validation accuracy to wandb
        wandb.run.summary["final loss"] = loss
        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)


if __name__ == '__main__':
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        pruner=optuna.pruners.MedianPruner(),
    )
    # Runs the 2 best configurations found so far
    # best_so_far()

    # experiments without a target sentence
    study.optimize(lambda x: objective(x, "none"), n_trials=20, timeout=None)

    # experiments with target sentence
    # study.optimize(lambda x: objective(x, "target"), n_trials=20, timeout=None)

    # experiments with previous sentence + target sentence
    # study.optimize(lambda x: objective(x, "targetprev"), n_trials=20, timeout=None)

