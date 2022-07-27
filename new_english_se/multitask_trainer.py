from enum import Enum, unique

import tensorflow as tf

tf.keras.datasets.cifar10.load_data()

import pandas
from keras.layers import Bidirectional, LSTM, Dense
from tensorflow.python.keras import Sequential
from transformers import AutoTokenizer, TFAutoModel, DistilBertTokenizer, TFDistilBertModel
import capslayer as cl

from rb.processings.istart.new_english_se.parse_corpus import SelfExplanations
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import confusion_matrix


@unique
class Task(Enum):
    BASE = 'base'
    MONITORING = 'monitoring'
    PARAPHRASE = 'paraphrasepresence'
    PR_LEXICAL_CHANGE = "lexicalchange"
    PR_SYNTACTIC_CHANGE = "syntacticchange"
    BRIDGING = "bridgepresence"
    BR_CONTRIBUTION = "bridgecontribution"
    ELABORATION = "elaborationpresence"
    EL_LIFE_EVENT = "lifeevent"
    OVERALL = "overall"


class MultiTaskTrainer:

    def __init__(self, path_to_csv_file):
        self.distitokenizer = None
        self.distibert = None
        self.df = pandas.read_csv(path_to_csv_file, delimiter=',')
        print(self.df.size)

    def tokenize(self):
        return self.df

    def confusion_matrix(self, df_test):
        cm = confusion_matrix(df_test[SelfExplanations.PR_LEXICAL_CHANGE],
                              df_test[SelfExplanations.PR_SYNTACTIC_CHANGE])
        df_cm = pandas.DataFrame(cm, range(cm.shape[0]), range(cm.shape[1]))
        plt.figure(figsize=(10, 10))
        sn.heatmap(df_cm, annot=True)

    def load_berttweet(self):
        self.bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
        inputs = self.tokenizer(self.df[SelfExplanations.SE][0] + self.df[SelfExplanations.TARGET_SENTENCE][0],
                                return_tensors="tf")

        self.df["SE_Tokenized"] = self.df[SelfExplanations.SE].apply(str).apply(self.tokenizer, return_tensors="tf")
        self.df["Target_Tokenized"] = self.df[SelfExplanations.TARGET_SENTENCE].apply(str).apply(self.tokenizer,
                                                                                                 return_tensors="tf")

        self.df["SE_Bert"] = self.df["SE_Tokenized"].apply(self.bertweet)
        self.df["Target_Bert"] = self.df["Target_Tokenized"].apply(self.bertweet)

        return inputs

    def load_distibert(self):
        MODEL_NAME = 'distilbert-base-uncased'
        self.distitokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.distibert = TFDistilBertModel.from_pretrained(MODEL_NAME)

        encodings = self.distitokenizer.encode_plus(
            text=self.df[SelfExplanations.SE][0],
            text_pair=self.df[SelfExplanations.TARGET_SENTENCE][0],
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="tf"
        )

        # self.df["SE_Tokenized"] = self.df[SelfExplanations.SE].apply(str).apply(self.distitokenizer,
        #                                                                         return_tensors="tf")
        # self.df["Target_Tokenized"] = self.df[SelfExplanations.TARGET_SENTENCE].apply(str).apply(self.distitokenizer,
        #                                                                                          return_tensors="tf")
        #
        # self.df["SE_DBert"] = self.df["SE_Tokenized"].apply(self.distibert)
        # self.df["Target_DBert"] = self.df["Target_Tokenized"].apply(self.distibert)

        outputs = self.distibert(encodings)
        print(outputs.last_hidden_state)
        return outputs

    def build_base_layer(self, input_size, output_size, batch_size):
        strides = 2

        model_base = Sequential(name="base")
        model_base.add(Bidirectional(LSTM(input_size)))
        model_base.add(Bidirectional(LSTM(input_size)))
        model_base.add(cl.layers.primaryCaps(inputs=[input_size],
                                             filters=8,
                                             kernel_size=[3, 1],
                                             out_caps_dims=[output_size, 1],
                                             strides=strides))
        model_base.add(cl.layers.conv1d(inputs=[input_size],
                                        activation=[batch_size, input_size, input_size],
                                        filters=4,
                                        kernel=[3, 1],
                                        out_caps_dims=[output_size, 1],
                                        strides=strides))
        model_base.add(cl.layers.dense(inputs=[input_size],
                                       activation=[batch_size, input_size, input_size],
                                       out_caps_dims=[output_size, 1],
                                       num_outputs=output_size))
        model_base.add(Dense(output_size, activation='relu'))
        model_base.summary()
        return model_base

    def build_task_layer(self, input_size, output_size, input_base, task: Task):
        model_task = Sequential(name=f"${task.name}")
        model_task.add(Bidirectional(LSTM(input_size)))
        model_task.add(Bidirectional(LSTM(input_size)))
        model_task.add(cl.layers.primaryCaps(input_size, out_caps_dims=output_size))
        model_task.add(cl.layers.conv1d(input_size, out_caps_dims=output_size))
        model_task.add(cl.layers.dense(input_size, out_caps_dims=output_size))
        model_task.keras.layers.Dense(input_size + input_base, activation='relu')
        model_task.keras.layers.Dense(output_size),

        return model_task

    def build_multitask(self, task_list: list[Task], input_size):
        model_base = self.build_task_layer(input_size, 5, Task.BASE)
        model_tasks = []
        for task in task_list:
            model_tasks.append(self.build_task_layer(input_size, 5, task))

        return model_base, model_tasks

    def train_model(self, dataset, batch_size, epochs):
        multitask_model = self.build_base_layer(100, 100, batch_size)
        multitask_model.compile(optimizer='adam', loss='mse')
        multitask_model.fit(dataset, dataset, batch_size=batch_size, epochs=epochs)
