import numpy as np
import torch
import pandas
import re

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from os.path import exists

from core.data_processing.filtering import ZeroRules


class SelfExplanations:
  TEXT_ID = "TextID"
  SENT_NO = "SentNo"

  SE = "SelfExplanation"
  TARGET_SENTENCE = "TargetSentence"
  PREVIOUS_SENTENCE = "PreviousSentence"
  TOO_SHORT = "tooshort"
  NON_SENSE = "nonsense"
  IRRELEVANT = "irrelevant"
  COPY_PASTE = "copypaste"
  MISCONCEPTION = "misconception"
  MONITORING = "monitoring"
  PARAPHRASE = "paraphrasepresence"
  PR_LEXICAL_CHANGE = "lexicalchange"
  PR_SYNTACTIC_CHANGE = "syntacticchange"
  BRIDGING = "bridgepresence"
  BR_CONTRIBUTION = "bridgecontribution"
  ELABORATION = "elaborationpresence"
  EL_LIFE_EVENT = "lifeevent"
  OVERALL = "overall"

  MTL_TARGETS = [PARAPHRASE, BRIDGING, ELABORATION, OVERALL]

  MTL_CLASS_DICT = {
    PARAPHRASE: 3,
    BRIDGING: 3,
    ELABORATION: 2,
    OVERALL: 4
  }

  def parse_se_from_csv(self, path_to_csv_file: str):
    df = pandas.read_csv(path_to_csv_file, delimiter='\t', dtype={self.SENT_NO: "Int64"}).dropna(how='all')
    self.df = df
    self.df['Production'] = self.df['SelfExplanation']
    self.df['Source'] = self.df['TargetSentence']
    for val in self.MTL_TARGETS:

      self.df[val][self.df[val] == 'BLANK '] = 0
      self.df[val][self.df[val] == 'BLANK'] = 0
      self.df[val][self.df[val] == 'blANK'] = 0
      self.df[val][self.df[val] == 9] = 0
      self.df[val] = self.df[val].astype(int)

    return df


class SEDataset(Dataset):
  def __init__(self, source, production, targets, tokenizer, max_len, rb_feats=None, filter_data=None):
    self.source = source
    self.production = production
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.filter_data = filter_data
    if rb_feats is not None:
      rb_feats[rb_feats == 'None'] = 0
      rb_feats[rb_feats == 'True'] = 1
      self.rb_feats = rb_feats.astype(float)
    else:
      self.rb_feats = None


    self.targets = np.vectorize(int)(self.targets)

  def __len__(self):
    return len(self.source)

  def __getitem__(self, item):
    source = str(self.source[item])
    production = str(self.production[item])
    target = self.targets[item]
    rb_feats = self.rb_feats[item] if self.rb_feats is not None else []
    filter_data = self.filter_data[item] if self.filter_data is not None else []

    encoding = self.tokenizer.encode_plus(
      text=source,
      text_pair=production,
      truncation=True,
      truncation_strategy="longest_first",
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'text_s': source,
      'text_p': production,
      'rb_feats': rb_feats,
      'filter_data': filter_data,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.LongTensor(target),
      'item': item,
    }

def create_data_loader(df, tokenizer, max_len, batch_size, num_tasks, use_rb_feats=False, use_filtering=False, task_name=None):
  if num_tasks == 1:
    targets = [task_name]
  else:
    targets = SelfExplanations.MTL_TARGETS[:num_tasks]
  feats = df[df.columns[114:-1]].to_numpy() if use_rb_feats else None

  filter_data = None
  if use_filtering:
    data = []
    zR = ZeroRules()
    for index, row in df.iloc[0:].iterrows():
      data.append(zR.get_filter_scores(row["Production"], row["Source"], ""))

    filter_data = np.array(data)
  ds = SEDataset(
    source=df['Source'].to_numpy(),
    production=df['Production'].to_numpy(),
    rb_feats=feats,
    filter_data=filter_data,
    targets=np.array([df[t] for t in targets]).transpose(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
  )
