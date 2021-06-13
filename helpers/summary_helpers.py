import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline, DistilBertModel,DistilBertTokenizer, DistilBertConfig

import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from summarizer import Summarizer

from .nlp_functions import detect_sentences, create_tdidf_doc_term_matrix

t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')


distillBert = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)
distillBertTokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

hf_summarizer = pipeline('summarization')

with open("./lib/punkt/PY3/english.pickle","rb") as resource:
  sent_detector = pickle.load(resource)

test_text ="""
The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.
The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.
At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.
"We'll be the comeback kids, all of us," he said. "We want to get our country back."
The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
"""

preprocess_text = test_text.strip().replace("\n","")


def get_summary(full_text, model_name="T5_sum"):
    if "T5" in model_name:
        return summarize_T5(full_text)
    if "BART" in model_name:
        return summarize_hf_pipeline(full_text)
    if "TF-IDF" in model_name:
        return summarize_tfidf(full_text)
    if "BERT" in model_name:
        return summarize_BERT_extractive(full_text)

def summarize_T5(text):

    tokenized_text = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)

    summary_ids = t5_model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=150,
                                    max_length=250,
                                    early_stopping=True)

    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


    return summary

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