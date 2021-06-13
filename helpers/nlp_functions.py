import pandas as pd
import numpy as np





def detect_sentences(full_text, sent_detector):
  sentences = sent_detector.tokenize(full_text)
  return sentences


def create_tdidf_doc_term_matrix(sentence_list, vectorizer):
    doc_term_matrix = vectorizer.fit_transform(sentence_list)
    doc_term_matrix_df = pd.DataFrame(doc_term_matrix.toarray(), columns=vectorizer.get_feature_names())
    doc_term_matrix_df.insert(loc=0, column="Sentence", value=sentence_list)
    doc_term_matrix_df.insert(loc=0,column="TOTAL",value=doc_term_matrix_df.sum(axis=1))
    return doc_term_matrix_df.sort_values('TOTAL',ascending=False)