import pandas as pd
from pandas import DataFrame

STUDY = "Dataset"
TEXT_ID = "TextID"
SENT_NO = "SentNo"
TARGET_SENTENCE = "TargetSentence"
PREVIOUS_SENTENCE = "PreviousSentence"


class StudyUtilities:

    def __init__(self, df):
        self.studies = df['Dataset'].unique()
        self.dfs_studies = {}
        for study in self.studies:
            self.dfs_studies[study] = df.loc[df['Dataset'] == study]

    def get_df_by_study(self, study) -> DataFrame:
        return self.dfs_studies[study]

    def get_stats_by_study_and_score(self, study, score):
        return self.dfs_studies[study][score].value_counts()

    def get_all_studies(self):
        return self.studies

