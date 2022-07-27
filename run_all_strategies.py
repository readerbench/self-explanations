from rb.processings.istart.new_english_se.parse_corpus import SelfExplanations

if __name__ == '__main__':
    self_explanations = SelfExplanations()
    self_explanations.parse_se_scoring_from_csv("new_english_se2.csv")
    # df_indices = self_explanations.compute_complexity_indices()
    #
    df_indices = self_explanations.load_se_with_indices()
