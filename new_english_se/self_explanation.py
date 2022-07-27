from dataclasses import dataclass

import tf as tf


@dataclass
class SelfExplanation:
    self_explanation: str
    target_sentence: str
    previous_sentence: str
    encodings: tf.Tensor

    def __init__(self, self_explanation: str,
                 target_sentence: str,
                 encodings: tf.Tensor,
                 previous_sentence: str = ''):
        super().__init__()
        self.self_explanation = self_explanation
        self.target_sentence = target_sentence
        self.encodings = encodings
        self.previous_sentence = previous_sentence
