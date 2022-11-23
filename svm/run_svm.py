from svm.svm_trainer import SVMTrainer
from svm.zero_rules import ZeroRules

if __name__ == '__main__':
    input_file = "Study1a_CRs_Humanratings_Combined.csv"

    zeroRules = ZeroRules()
    svmTrainer = SVMTrainer()

    df = svmTrainer.readData(input_file)
    df_zero_predicted = zeroRules.predict_zeros(df)
    df_zero_predicted.to_csv("Study1a_CRs_Humanratings_Combined_0_Tagged.csv", index=False)


