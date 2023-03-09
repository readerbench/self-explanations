from rb.core.document import Document
import distance


def get_lemma(word):
    if word.lemma != "-PRON-":
        return word.lemma
    return word.text


def get_pos(d: Document):
    return set([get_lemma(word) for word in d.get_words() if word.is_alpha])


def get_words(d: Document):
    return set([get_lemma(word) for word in d.get_words() if word.is_alpha])


def get_3grams(d: Document):
    list = [get_lemma(word) for word in d.get_words() if word.is_alpha]
    ngrams = ["-".join(list[i:i+3]) for i in range(len(list) - 2)]
    return set(ngrams)


def get_5grams(d: Document):
    list = [get_lemma(word) for word in d.get_words() if word.is_alpha]
    ngrams = ["-".join(list[i:i+5]) for i in range(len(list) - 4)]
    return set(ngrams)


def get_content_words(d: Document):
    return [get_lemma(word) for word in d.get_words() if word.is_alpha and word.is_content_word()]


def get_content_words_no_lemma(d: Document):
    return [word.text.lower() for word in d.get_words() if word.is_alpha and word.is_content_word()]


def get_overlap_metrics(s: Document, r: Document):
    s_words = get_words(s)
    s_c_words = get_content_words(s)
    r_words = get_words(r)
    r_c_words = get_content_words(r)

    intersection = [w for w in r_words if w in s_words]
    intersection_c = [w for w in r_c_words if w in s_c_words]
    diff_s_to_r_c = [w for w in s_c_words if not w in r_c_words]
    diff_s_to_r = [w for w in s_words if not w in r_words]
    diff_r_to_s_c = [w for w in r_c_words if not w in s_c_words]
    diff_r_to_s = [w for w in r_words if not w in s_words]

    perc_s_to_r = len(intersection) / len(r_words) if len(r_words) > 0 else 0
    perc_r_to_s = len(intersection) / len(s_words)
    perc_s_to_r_c = len(intersection_c) / len(r_c_words) if len(r_c_words) > 0 else 0
    perc_r_to_s_c = len(intersection_c) / len(s_c_words) if len(s_c_words) > 0 else 0

    return {
        "intersection": len(intersection),
        "intersection_c": len(intersection_c),
        "diff_s_to_r_c": len(diff_s_to_r_c),
        "diff_s_to_r": len(diff_s_to_r),
        "diff_r_to_s_c": len(diff_r_to_s_c),
        "diff_r_to_s": len(diff_r_to_s),
        "perc_s_to_r": perc_s_to_r,
        "perc_r_to_s": perc_r_to_s,
        "perc_s_to_r_c": perc_s_to_r_c,
        "perc_r_to_s_c": perc_r_to_s_c,
    }


def get_edit_distance_metrics(s: Document, r: Document):
    s_words = [get_lemma(word) for word in s.get_words() if word.is_alpha]
    s_c_words = [get_lemma(word) for word in s.get_words() if word.is_alpha and word.is_content_word()]
    r_words = [get_lemma(word) for word in r.get_words() if word.is_alpha]
    r_c_words = [get_lemma(word) for word in r.get_words() if word.is_alpha and word.is_content_word()]

    dist_s_to_r = distance.levenshtein(s_words, r_words)
    dist_s_to_r_c = distance.levenshtein(s_c_words, r_c_words)

    return {
        "dist_s_to_r": dist_s_to_r,
        "dist_s_to_r_c": dist_s_to_r_c
    }


def get_pos_ngram_overlap_metrics(s: Document, r: Document):
    s_words = [word.pos.value for word in s.get_words() if word.is_alpha]
    s_c_words = [word.pos.value for word in s.get_words() if word.is_alpha and word.is_content_word()]
    r_words = [word.pos.value for word in r.get_words() if word.is_alpha]
    r_c_words = [word.pos.value for word in r.get_words() if word.is_alpha and word.is_content_word()]

    intersection = [w for w in r_words if w in s_words]
    intersection_c = [w for w in r_c_words if w in s_c_words]
    diff_s_to_r_c = [w for w in s_c_words if not w in r_c_words]
    diff_s_to_r = [w for w in s_words if not w in r_words]
    diff_r_to_s_c = [w for w in r_c_words if not w in s_c_words]
    diff_r_to_s = [w for w in r_words if not w in s_words]

    perc_s_to_r = len(intersection) / len(r_words) if len(r_words) > 0 else 0
    perc_r_to_s = len(intersection) / len(s_words)
    perc_s_to_r_c = len(intersection_c) / len(r_c_words) if len(r_c_words) > 0 else 0
    perc_r_to_s_c = len(intersection_c) / len(s_c_words) if len(s_c_words) > 0 else 0

    return {
        "intersection_pos": len(intersection),
        "intersection_c_pos": len(intersection_c),
        "diff_s_to_r_c_pos": len(diff_s_to_r_c),
        "diff_s_to_r_pos": len(diff_s_to_r),
        "diff_r_to_s_c_pos": len(diff_r_to_s_c),
        "diff_r_to_s_pos": len(diff_r_to_s),
        "perc_s_to_r_pos": perc_s_to_r,
        "perc_r_to_s_pos": perc_r_to_s,
        "perc_s_to_r_c_pos": perc_s_to_r_c,
        "perc_r_to_s_c_pos": perc_r_to_s_c,
    }

