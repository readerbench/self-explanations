from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.document import Document
from rb.core.lang import Lang


def get_indexes(lang_model, doc, index_categories):
    cna_graph = CnaGraph(docs=[doc], models=[lang_model])
    try:
        compute_indices(doc=doc, cna_graph=cna_graph)
    except IndexError:
        print("Skip index error")
        print({repr(ind): doc.indices[ind] for ind in doc.indices if ind.category in index_categories and repr(ind).find("Block") == -1})

    local_relevant_indices = {repr(ind): doc.indices[ind] for ind in doc.indices if ind.category in index_categories and repr(ind).find("Block") == -1}

    return local_relevant_indices


def create_doc(text):
    text = "" if text is None else text
    return Document(lang=Lang.EN, text=text)
