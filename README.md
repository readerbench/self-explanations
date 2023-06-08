# self-explanations-mtl

In order to recreate the experiments, the following steps are required:
 
1. Parse the old and new corpus in order to generate an *.xslx file containing both the self-explanations and the source text.
For this you need to run: process_raw_se_files.py
Input files: CR_all.columns_10.17.22.xlsx, new_english_se2_enhanced.csv 
Output files: CR_all.columns_10.17.22_enhanced.xlsx

2. Process the *.xslx file containing the corpus in order to generate and attach the RB features, and do some rudimentary cleanup.
For this you need to run: input_process.py
Input files: CR_all.columns_10.17.22_enhanced.xlsx 
Output files: results_se_aggregated_dataset_clean.csv

3. Train the models on the data in `results_se_aggregated_dataset_clean.csv`. In order to train the models, run
- for BERT: mtl_bert_train.py
- for XGBoost: mtl_singletask_xgb.py

4. Evaluate trained BERT checkpoints by running mtl_bert_load_and_test
