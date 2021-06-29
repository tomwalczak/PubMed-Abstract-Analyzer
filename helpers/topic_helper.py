import streamlit as st

import numpy as np
import pandas as pd
import random

from sklearn import decomposition
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px



word_embeddings_df = pd.read_csv('./helpers/word_embeddings.csv', index_col=0)

topic_names = set(word_embeddings_df["topic"].tolist())


def get_word_embeddings_df():
  return word_embeddings_df

def plot_random_topic_words():

  # Pick a few topics at random
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