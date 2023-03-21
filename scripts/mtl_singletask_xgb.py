import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from core.data_processing.se_dataset import SelfExplanations
from scripts.mtl_bert_train import get_new_train_test_split, filter_rb_df

file = "../data/results_paraphrase_se_aggregated_dataset_2.csv"

df = pd.read_csv(file, delimiter='\t')
df = filter_rb_df(df)

feature_columns = df.columns.tolist()[114:]

df_train, df_dev, df_test = get_new_train_test_split(df)
for task in SelfExplanations.MTL_TARGETS:
    # eliminating unlabeled datapoints for this task
    df_train_filtered = df_train[df_train[task] != 9]
    df_test_filtered = df_test[df_test[task] != 9]

    y_train = df_train_filtered[task]
    y_test = df_test_filtered[task]
    if 0 not in np.unique(np.concatenate([y_train, y_test])):
        y_test -= 1
        y_train -= 1
    x_train = df_train_filtered[feature_columns]
    x_test = df_test_filtered[feature_columns]

    model = XGBClassifier(tree_method="gpu_hist", enable_categorical=True)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    print(task, accuracy_score(y_predict, y_test))
