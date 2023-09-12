import numpy as np
import torch
import pandas

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


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

    TASK_TARGETS = [PARAPHRASE, BRIDGING, ELABORATION, OVERALL]

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
        for val in self.TASK_TARGETS:
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

def create_data_loader(df, tokenizer, max_len, batch_size, num_tasks, use_rb_feats=False, task_name=None):
    if num_tasks == 1:
        targets = [task_name]
    else:
        targets = SelfExplanations.TASK_TARGETS[:num_tasks]

    filter_data = None
    ds = SEDataset(
        source=df['Source'].to_numpy(),
        production=df['Production'].to_numpy(),
        rb_feats=None,
        filter_data=filter_data,
        targets=np.array([df[t] for t in targets]).transpose(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )


def map_train_test(x):
    if x['Dataset'] in ['Dataset 3']:
        return 'train'
    if x['Dataset'] == 'Dataset 1':
        if 'RBC' in x['TextID'].upper():
            return 'test'
        if x['PrePost'] == 'Post':
            return 'dev'
        return 'train'
    if x['Dataset'] == 'Dataset 2':
        if not str(x['ID']).startswith('IREF'):
            return 'train'
        else:
            return 'dev'
    return 'dump'


# load the dataset
def load_split(train_data, train_labels, split_name, tokenizer):
    dataset = Dataset.from_dict({
        "prompt": train_data,
        "label": [["(A)", "(B)", "(C)", "(D)"][x] for x in train_labels]
    })

    dataset = dataset.map(
        lambda x: encode_batch(tokenizer, x),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on " + split_name + " dataset",
    )
    # set the format to torch
    dataset.set_format(type="torch", columns=["input_ids", "labels"])

    return dataset


# tokenize the dataset
def encode_batch(tokenizer, examples):
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