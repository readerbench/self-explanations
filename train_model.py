from rb.processings.istart.new_english_se.multitask_trainer import MultiTaskTrainer, Task
from rb.processings.istart.new_english_se.parse_corpus import SelfExplanations
from rb.processings.istart.new_english_se.study_group import StudyUtilities, STUDY, TEXT_ID, SENT_NO
from os.path import exists

if __name__ == '__main__':

    self_explanations = SelfExplanations()
    target_sent_enhanced = self_explanations.parse_se_scoring_from_csv("new_english_se2.csv")
    trainer = MultiTaskTrainer(target_sent_enhanced)
    trainer.load_berttweet()
    trainer.load_distibert()
    trainer.train_model(target_sent_enhanced, 100, 10)

