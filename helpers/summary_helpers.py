import streamlit as st

from transformers import  DistilBertModel, DistilBertTokenizer
from summarizer import Summarizer

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from .nlp_functions import detect_sentences, create_tdidf_doc_term_matrix

distillBert = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)
distillBertTokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


with open("./lib/punkt/PY3/english.pickle","rb") as resource:
  sent_detector = pickle.load(resource)


def get_summary(full_text, model_name="T5_sum"):
    if "TF-IDF" in model_name:
        return summarize_tfidf(full_text)
    if "BERT" in model_name:
        return summarize_BERT_extractive(full_text)
    else:
        return summarize_tfidf(full_text) 


def summarize_hf_pipeline(abstract):
    abstract = abstract.replace('.', '.<eos>').replace('?', '?<eos>').replace('!', '!<eos>')

    max_words_in_chunk = 510
    sents = abstract.split('<eos>')
    chunk_idx = 0
    chunks = []
    for sentence in sents:
        if len(chunks) == chunk_idx + 1:
            # Check if adding current sentence's words would exceed max words limit
            if len(chunks[chunk_idx]) + len(sentence.split(' ')) <= max_words_in_chunk:
                chunks[chunk_idx].extend(sentence.split(' '))
            else:
                chunk_idx += 1
                # Append sentence words into the current chunk
                chunks.append(sentence.split(' '))
        else:
            print(chunk_idx)
            # Append sentence words into the current chunk
            chunks.append(sentence.split(' '))

    # Join words back into sentences, this is that the TF pipeline expects
    chunks = [' '.join(chunks[chunk_idx]) for chunk_idx in range(len(chunks))]

    chunk_summaries = hf_summarizer(chunks, max_length=240, min_length=120, do_sample=False)

    ## Merge chunk summaries and replace leading spaces
    summary = ' '.join([summ['summary_text'] for summ in chunk_summaries]).replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace("\\","")[1:]

    return summary

def summarize_tfidf(text,top_sents_num=4):
    sentences = detect_sentences(text,sent_detector=sent_detector)

    tfidf_vectorizer = TfidfVectorizer()

    df = create_tdidf_doc_term_matrix(sentences,tfidf_vectorizer)

    # Pick to N sents
    top_sents = df[:top_sents_num]

    # and sort them in order in which they appear in the full text, for coherence
    final_tidf_summary = top_sents.sort_index(axis=0) 

    final_tidf_summary = " ".join([row["Sentence"] for _, row in final_tidf_summary.iterrows()])

    return final_tidf_summary

def summarize_BERT_extractive(text):
    
    model = Summarizer(custom_model=distillBert, custom_tokenizer=distillBertTokenizer)
    result = model(text, ratio=0.2)
    full = ''.join(result)
    return full