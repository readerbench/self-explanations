from rb.processings.istart.new_english_se.multitask_trainer import Task
from rb.processings.istart.new_english_se.parse_corpus import SelfExplanations
from rb.processings.istart.new_english_se.study_group import StudyUtilities

if __name__ == '__main__':
    se = SelfExplanations()
    se.parse_se_scoring_from_csv("new_english_se2_enhanced.csv")

    study_utilities = StudyUtilities(se.df)
    studies = study_utilities.get_all_studies()
    tasks = [Task.PARAPHRASE, Task.PR_LEXICAL_CHANGE, Task.PR_SYNTACTIC_CHANGE,
             Task.BRIDGING, Task.BR_CONTRIBUTION, Task.ELABORATION, Task.EL_LIFE_EVENT, Task.OVERALL]
    for study in studies:
        print("Study: ", study)
        print(study_utilities.dfs_studies[study].sample)
        # for task in tasks:
        #     print(task.name, " stats: ")
        #     print(study_utilities.get_stats_by_study_and_score(study, task.value))

