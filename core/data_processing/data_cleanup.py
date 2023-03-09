import pandas as pd
import numpy
from core.data_processing.input_process import process_dataset
from core.data_processing.se_dataset import SelfExplanations




filename = process_dataset("se_aggregated_dataset", use_prev_sentence=True)
clean_csv(filename)
filename = process_dataset("se_aggregated_dataset", use_prev_sentence=False)
clean_csv(filename)

