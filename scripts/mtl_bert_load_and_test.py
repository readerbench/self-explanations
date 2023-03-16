from transformers import BertTokenizer, RobertaTokenizer

from core.data_processing.se_dataset import SelfExplanations, create_data_loader
from core.models.mtl import BERTMTL
from scripts.mtl_bert_train import get_train_test_IDs, get_new_train_test_split, filter_rb_df
import pytorch_lightning as pl
import torch
import numpy as np

if __name__ == '__main__':
    num_tasks = 4
    predefined_version = ""
    PRE_TRAINED_MODEL_NAME = 'roberta-base'
    MAX_LEN_P = 80
    BATCH_SIZE = 1
    if "roberta" in PRE_TRAINED_MODEL_NAME:
        tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    else:
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    self_explanations = SelfExplanations()
    target_sent_enhanced = self_explanations.parse_se_from_csv(
        "../data/results_paraphrase_se_aggregated_dataset_v2.csv")
    self_explanations.df = filter_rb_df(self_explanations.df)

    # IDs = self_explanations.df['ID'].unique().tolist()
    # _, test_IDs = get_train_test_IDs(IDs)
    # df_test = self_explanations.df[self_explanations.df['ID'].isin(test_IDs)]
    df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df)

    val_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=True, use_filtering=True)
    # train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN_P, BATCH_SIZE, num_tasks, use_rb_feats=True, use_filtering=True)
    task_names = [SelfExplanations.MTL_TARGETS[task_id] for task_id in range(num_tasks)]
    model = BERTMTL(task_names, PRE_TRAINED_MODEL_NAME, rb_feats=val_data_loader.dataset.rb_feats.shape[1], use_filtering=True)


    model = model.load_from_checkpoint("./mtl/lightning_logs/version_24/checkpoints/epoch=21-step=1848.ckpt",
                                       task_names=task_names,
                                       pretrained_bert_model=PRE_TRAINED_MODEL_NAME,
                                       rb_feats=val_data_loader.dataset.rb_feats.shape[1], use_filtering=True)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      # BERT params - with WD
      {'params': [p for n, p in model.named_parameters() if
                  not any(nd in n for nd in no_decay) and n.find("bert") != -1],
       'weight_decay': 0.0001, 'lr': 1e-5},
      # BERT params - no WD
      {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n.find("bert") != -1],
       'weight_decay': 0.0, 'lr': 1e-5},
      # non-BERT params - with WD
      {'params': [p for n, p in model.named_parameters() if
                  not any(nd in n for nd in no_decay) and n.find("bert") == -1],
       'weight_decay': 0.0001},
      # non-BERT params - no WD
      {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n.find("bert") == -1],
       'weight_decay': 0.0}
    ]

    print("Total params:", params)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        # limit_train_batches=100,
        max_epochs=50)
    # trainer.test(model, dataloaders=train_data_loader)
    trainer.test(model, dataloaders=val_data_loader)