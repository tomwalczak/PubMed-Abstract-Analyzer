import streamlit as st
import transformers
import torch
import numpy as np
import pandas as pd
import random

import spacy
from sklearn import decomposition
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px

# from bertopic import BERTopic

# topic_model = BERTopic.load('./saved_models/bert_topic_model')

word_embeddings_df = pd.read_csv('./helpers/data/word_embeddings.csv', index_col=0)

topic_names = set(word_embeddings_df["topic"].tolist())

nlp = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

def detect_topic(abstract):
  topics, _ = topic_model.transform([abstract])
  topic_name = get_topic_name_from_id(topics[0])
  topic_words = extract_words_from_topic(topic_name)
  return topic_name, topic_words

def get_topic_graph_df(word_embeddings_df, topic_name ):
  selected_topics = random.sample(get_list_of_topic_names(),3)
  selected_word_embeddings = word_embeddings_df.loc[word_embeddings_df['topic'].isin(selected_topics)]

  # drop words that also belong to another topic
  selected_word_embeddings = selected_word_embeddings[selected_word_embeddings["topic"]==topic_name]
  return selected_word_embeddings

def get_word_embeddings_df():
  return word_embeddings_df


  topics = topic_model.get_topics()
  ## Get all relevant words associated with topics for embedding
  topic_word_list = get_list_of_words_in_all_topics()
  ## Remove repeating words
  topic_word_list = set(topic_word_list)
  ## Get word embeddings
  spacy_embeddings = [nlp(word).vector for word in topic_word_list]

  pca = decomposition.PCA(n_components=3)
  pca.fit(spacy_embeddings)
  embeddings_pca = pca.transform(spacy_embeddings)
  words_embedded_pca= pd.DataFrame({})

  for (idx,word) in enumerate(topic_word_list):
    words_embedded_pca[word] = embeddings_pca[idx]

  words_embedded_pca = words_embedded_pca.transpose()
  words_embedded_pca.columns = ["comp1", "comp2", "comp3"]
  words_embedded_pca['word'] = topic_word_list
  words_embedded_pca['topic'] = [get_topic_name_from_id(get_topic_from_word(word,topics)) for word in topic_word_list]
  return words_embedded_pca, pca

def get_list_of_words_in_all_topics(how_many_words=3):
  topic_names = get_list_of_topic_names()
  # Get all topic words and calculate PCA
  topic_words_top = [extract_words_from_topic(topic_name)[:how_many_words] for topic_name in topic_names]
  # Remove first '-1' topic (non-topic) 
  topic_words_top = topic_words_top[1:]
  # Flatten the list
  word_list_flat = [word for sublist in topic_words_top for word in sublist]
  return word_list_flat

def get_topic_from_word(word,topics):
  for topic_id in topics:
    word_tuples_list = topics[topic_id]
    for word_tuple in word_tuples_list:
      if word_tuple[0] == word: return topic_id
    
  return ""

def get_topic_name_from_id(topic_id):
  topic_info_df = topic_model.get_topic_info()
  return topic_info_df.loc[topic_info_df['Topic']==topic_id]['Name'].tolist()[0]

def extract_words_from_topic(topic_name):
  topics = topic_model.get_topics()
  topic_id = get_topic_id_from_name(topic_name)
  topic_word_tuples = topics[topic_id]

  return [word_tuple[0] for word_tuple in topic_word_tuples]

def get_topic_id_from_name(name):
  df = topic_model.get_topic_info()
  return df.loc[df["Name"]==name]["Topic"].tolist()[0]

def get_list_of_topic_names():
  topics_df = topic_model.get_topic_info()
  topics_list = topics_df['Name'].tolist()
  # remove -1 non-topic
  return topics_list[1:]

def get_most_similar_topic_words(topic_name,numer_of_words = 3):
  top_topic_words = extract_words_from_topic(topic_name)[:numer_of_words]

  list_of_distances = []
  spacy_embeddings = [nlp(word).vector for word in top_topic_words]

  cosine_sim_matrix = cosine_similarity(spacy_embeddings)

  for idx1, word1 in enumerate(top_topic_words):
    for idx2, word2 in enumerate(top_topic_words):
      if word1 != word2:
        list_of_distances.append([word1, word2, cosine_sim_matrix[idx1][idx2]])

  df = pd.DataFrame(list_of_distances, columns=['word1', 'word2', 'cosine similarity'])
  df = df.sort_values('cosine similarity',ascending=False)
  # Similarity is pair-wise, drop one item from each pair as they are duplicates
  df = df.drop_duplicates(subset="cosine similarity")
  # Remove 0 values, for embeddings out of vocab
  df = df[df["cosine similarity"] > 0]

  list_of_words = set(df['word1'].tolist() + df['word2'].tolist())

  return list_of_words, df

def calculate_topic_centroid(topic_name, fitted_pca, include_words=3):
  topics = topic_model.get_topics()
  topic_id = get_topic_id_from_name(topic_name)
  topic_word_tuples = topics[topic_id]
  top_topic_words = extract_words_from_topic(topic_name)[:include_words]

  spacy_embeddings = [nlp(word).vector for word in top_topic_words]
  # Average the embeddings
  for idx in range(1,include_words):
    spacy_embeddings[0] += spacy_embeddings[idx]

  spacy_embeddings = spacy_embeddings[0] / include_words

  embeddings_pca = fitted_pca.transform([spacy_embeddings])
  # Return the only item, at index 0
  return embeddings_pca[0]

def plot_topic_centroid_in_context(centroid_topic_name,word_embeddings_df,pca):
  #Pick 5 topics at random
  topic_names = random.sample(get_list_of_topic_names(),7)
  
  list_of_words = []
  for topic_name in topic_names:
    topic_words, _ = get_most_similar_topic_words(topic_name,numer_of_words = 7)
    list_of_words += topic_words

  selected_word_embeddings = word_embeddings_df.loc[word_embeddings_df['word'].isin(list_of_words)]

  # Now, add topic centroid
  topic_centroid = calculate_topic_centroid(centroid_topic_name,fitted_pca=pca)
  centroid_df = pd.DataFrame({

      "comp1": topic_centroid[0],
      "comp2": topic_centroid[1],
      "comp3": topic_centroid[2],
      "word": ["THIS ABSTRACT"],
      "topic": [centroid_topic_name]
  })

   # drop words that also belong to another topic
  selected_word_embeddings = selected_word_embeddings[selected_word_embeddings["topic"]!=centroid_topic_name]

  selected_word_embeddings = selected_word_embeddings.append(centroid_df)

  fig = px.scatter_3d(selected_word_embeddings, 
                      x='comp1', y='comp2', z='comp3', 
                      text='word', 
                      range_x=[-4,4], range_y=[-4,4], range_z=[-4,4],  
                      color='topic',
                      opacity=0.7,
                      
                      
                      )
  fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

  fig.update_layout(showlegend=False)

  return fig


def plot_random_topic_words():
  #Pick a few topics at random
  # selected_topic_names = random.sample(topic_names,7)
  
  # list_of_words = []
  # for topic_name in selected_topic_names:
  #   top_topic_words = extract_words_from_topic(topic_name)[:7]
  #   topic_words, _ = get_most_similar_topic_words(topic_name,topic_model,numer_of_words = 7)
  #   list_of_words += topic_words

  random_list_of_topic_words_lists = random.sample(get_list_of_topic_words(),7)
  topic_words = []
  for topic_list in random_list_of_topic_words_lists:
    for word in topic_list:
      topic_words.append(word)

  selected_word_embeddings = word_embeddings_df.loc[word_embeddings_df['word'].isin(topic_words)]


  fig = px.scatter_3d(selected_word_embeddings, 
                      x='comp1', y='comp2', z='comp3', 
                      text='word', 
                      range_x=[-4,4], range_y=[-4,4], range_z=[-4,4],  
                      color='topic',
                      opacity=0.7,
                      size_max=18,
                      
                      )
  fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
  fig.update_layout(showlegend=False)
  return fig


def get_list_of_topic_words():
  return [['health', 'training', 'care', 'intervention', 'depression'],
 ['epidural', 'morphine', 'anesthesia', 'sedation', 'analgesia'],
 ['survival', 'chemotherapy', 'cancer', 'tumor', 'radiation'],
 ['mg', 'postmenopausal', 'testosterone', 'women', 'pregnancy'],
 ['teeth', 'bone', 'implants', 'dental', 'plaque'],
 ['diet', 'weight', 'dietary', 'cholesterol', 'intake'],
 ['eye', 'eyes', 'cataract', 'corneal', 'laser'],
 ['fasting', 'diabetes', 'insulin', 'metformin', 'glucose'],
 ['survival', 'chemotherapy', 'cancer', 'breast', 'tamoxifen'],
 ['skin', 'topical', 'cure', 'treatment', 'psoriasis'],
 ['stroke', 'infarction', 'coronary', 'pci'],
 ['methotrexate', 'arthritis', 'rheumatoid', 'remission'],
 ['stroke', 'training', 'acupuncture', 'treatment', 'motor'],
 ['viral', 'hiv', 'antiretroviral', 'cmv'],
 ['screening', 'prostate', 'prostatic', 'psa'],
 ['cessation', 'nicotine', 'smoking', 'smokers', 'quit'],
 ['artery', 'stent', 'angioplasty', 'stents'],
 ['cholesterol', 'atorvastatin', 'lipoprotein'],
 ['ckd', 'serum', 'creatinine', 'dialysis', 'renal'],
 ['consumption', 'alcohol', 'drinking', 'intervention', 'participants'],
 ['burn', 'ulcers', 'wound', 'healed', 'healing'],
 ['ulcerative', 'colonoscopy', 'bowel', 'preparation', 'colitis'],
 ['iron', 'infant', 'vitamin', 'zinc', 'infants'],
 ['pylori', 'esomeprazole', 'helicobacter', 'therapy', 'omeprazole'],
 ['budesonide', 'eosinophil', 'cough', 'asthma', 'inhaled'],
 ['urinary', 'infections', 'urethral', 'catheter'],
 ['cpap', 'melatonin', 'apnea', 'insomnia', 'sleep'],
 ['hcv', 'ifn', 'ribavirin', 'hepatitis', 'hbv'],
 ['pressure', 'hypertension', 'angiotensin'],
 ['cabg', 'cardiopulmonary', 'cpb', 'bypass'],
 ['vaccination', 'immunogenicity', 'antibody', 'vaccines', 'vaccine'],
 ['mortality', 'hf', 'heart', 'chf', 'hrv'],
 ['fibrillation', 'defibrillation', 'pacing', 'vt'],
 ['rehabilitation', 'vo2', 'pulmonary', 'chronic', 'copd'],
 ['pregabalin', 'migraine', 'headache', 'efficacy'],
 ['disability', 'lumbar', 'cervical', 'pain', 'surgery'],
 ['mineral', 'spine', 'bone'],
 ['incontinence', 'pelvic'],
 ['ventilation', 'tracheal', 'airway', 'intubation'],
 ['fractures', 'femoral', 'fixation', 'screw', 'hip'],
 ['tka', 'knee', 'pain', 'osteoarthritis', 'splint'],
 ['cocaine', 'heroin', 'abstinence', 'cannabis'],
 ['allergen', 'allergic', 'ige', 'nasal', 'allergy'],
 ['plasmodium', 'malaria', 'falciparum', 'doxycycline']]