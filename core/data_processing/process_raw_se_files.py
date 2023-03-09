import pandas

def map_entry(x, sent_dict_raw):
    return sent_dict_raw[x['TextID']][x['SentNo']]


def main():
    # File containing new data, BUT only references to texts. Needs to be enhanced with the actual texts.
    raw_file = "../../data/CR_all.columns_10.17.22.xlsx"
    df_raw = pandas.read_excel(raw_file)

    # File containing older data, with the actual text
    labeled_file = "../../data/new_english_se2_enhanced.csv"
    df = pandas.read_csv(labeled_file, delimiter=',', dtype={"SentNo": "Int64"}).dropna(how='all')

    texts = df['TextID'].unique().tolist()
    # Dict containing textID -> sentenceID -> actualSentence associations. Similar to `extra_dict`
    sent_dict = {t: {} for t in texts}

    # Extra texts, not covered in the file containing older data
    extra_dict = {
        "NIU 3 - 327": {
            5: ("The varied forces of erosion still continue to shape and reshape these landforms,as well as the entire Earth.", "These have taken thousands to millions of years to shape."),
            9: ("Given enough time, almost any exposed rock will eventually be worn away.", "Rock beds and large boulders are reduced into rock debris, soil particles, and sand."),
            14: ("These break down quite slowly.", "Rocks composed of hard minerals such as quartz crystals are resistant to weathering."),
            16: ("Such cracks provide an entry place for water, bacteria, and plant roots.", "However, if these rocks are marred with fractures and joints, they can be quite vulnerable to weathering."),
            21: ("Therefore, hot and humid climates accelerate many of the chemical reactions that lead to erosion.", "Hot and humid weather conditions can stimulate the chemical conversion of the mineral feldspar into the soft white clay known as kaolinite, which is more vulnerable to weathering."),
            22: ("As mentioned earlier, water can also accelerate the process of erosion.", "Therefore, hot and humid climates accelerate many of the chemical reactions that lead to erosion."),
            26: ("This fact explains the amazing endurance of the pyramids and other ancient monuments.", "In the absence of water, chemical weathering slows tremendously.")
        },
        "NIU 3 - 340": {
            4: ("Immediately after he was crowned, Louis repealed some of the most oppressive taxes and instituted financial and judicial reforms.", "On Louis's accession, France was impoverished and burdened with debts, and heavy taxation had resulted in widespread misery among the French people."),
            6: ("Eventually, Louis had to replace a minister, who was the central architect of the reforms that the majority of the French people needed so badly.", "Greater reforms were prevented, however, by the opposition of the upper classes and the court."),
            11: ("Louis was unsuccessful in generating the badly needed funds.", "At the same time, the public became angered by the lavish spending of the French nobility and the royal court."),
            14: ("The French government was in danger of going bankrupt.", "The French finance minister had to continue borrowing money until the limit was reached in 1786."),
            17: ("On July 14, 1789, the Parisian populace razed the Bastille, and a short time later imprisoned the King and royal family in the palace of the Tuileries.", "Once in session, the Estates-General assumed the powers of government."),
            21: ("Historians consider Louis XVI a victim of circumstance rather than a despot resembling the former French Kings Louis XIV and Louis XV.", "In 1792, when the National Convention (the assembly of elected French deputies declared France a republic, the King was tried as a traitor and condemned to death.")
        }
    }

    # populating sent_dict
    for t in texts:
        df_t = df[df['TextID'] == t]
        sents = list(df_t["SentNo"].unique())
        for s in sents:
            sample_target = df_t[df_t["SentNo"] == s].head(1)['TargetSentence'].values[0]
            sample_prev = df_t[df_t["SentNo"] == s].head(1)['PreviousSentence'].values[0]
            sent_dict[t][s] = (sample_target, sample_prev)

    # Making text IDs more uniform
    for t in sent_dict:
        if t != t.strip():
            sent_dict[t.strip()] = sent_dict[t]
            del sent_dict[t]

    texts_raw = df_raw['TextID'].unique().tolist()
    sent_dict_raw = {t: {} for t in texts_raw}

    # Merging the 2 sentence dicts
    for t in texts_raw:
        df_t = df_raw[df_raw['TextID']==t]
        sents = list(df_t["SentNo"].unique())
        for s in sents:
            sent_dict_raw[t][s] = extra_dict[t][s] if t in extra_dict else sent_dict[t][s]

    # Mapping the references in the new document to the actual texts
    df_raw['TargetSentence'] = df_raw.apply(lambda x: map_entry(x, sent_dict_raw)[0], axis=1)
    df_raw['PreviousSentence'] = df_raw.apply(lambda x: map_entry(x, sent_dict_raw)[1], axis=1)

    df_raw.to_excel("../../data/CR_all.columns_10.17.22_enhanced.xlsx")


if __name__ == '__main__':
    main()
