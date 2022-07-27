from keras.utils.data_utils import Sequence
from pandas import DataFrame
from transformers import BertTokenizer

from rb.processings.istart.new_english_se.self_explanation import SelfExplanation


class SelfExplanationSequence(Sequence):

    def __init__(self, df: DataFrame, tokenizer: BertTokenizer, use_previous_sentence=False):
        self.df = df
        self.tokenizer = tokenizer
        self.use_previous_sentence = use_previous_sentence

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass



    def load_dataset(self):
        return SelfExplanation()