import numpy as np

ENCODING = 'UTF-8'
SEP = '\t'

def load(tf_file: str, idf_file: str) -> tuple:
    index, docs = {}, []
    with open(idf_file, 'rt', encoding=ENCODING) as idf:
        for line in idf:
            term, term_idf = line.rstrip().split(SEP)
            index[term] = float(term_idf), []
    with open(tf_file, 'rt', encoding=ENCODING) as tf:
        for line in tf:
            doc_id, term, term_tf = line.rstrip().split(SEP)
            if not docs or doc_id > docs[-1]:
                docs.append(doc_id)
            assert doc_id == docs[-1]
            index[term][1].append((len(docs) - 1, float(term_tf)))
    return index, docs

def tf_idf(index: dict, docs: list) -> np.ndarray:
    tf_idf = np.zeros(shape=(len(docs), len(index)))
    word_index = 0
    for word in index:
        term_idf = index[word][0]
        for doc_id, term_tf in index[word][1]:
            tf_idf[doc_id, word_index] = term_idf * term_tf;
        word_index += 1
    return tf_idf

index, docs = load(tf_file='f2small', idf_file='f1small')
corpus_tf_idf = tf_idf(index=index, docs=docs)
print(corpus_tf_idf)
query = ['brand', 'pollyanna']
