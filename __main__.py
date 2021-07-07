import streamlit as st
import pickle

import plotly.graph_objects as go

import random

from helpers.abstract_helpers import preprocess_abstracts_from_file, get_abstract_markdown, add_positon_feature_to_sentences
from helpers.remote_abstract_models import get_abstract_results_df_from_remote_model, get_remote_model_results
from helpers.summary_helpers import get_summary

from helpers.remote_claim_extraction_model import get_extracted_claims_from_remote_model

from helpers.topic_helper import plot_random_topic_words

with open("./lib/punkt/PY3/english.pickle","rb") as resource:
  sent_detector = pickle.load(resource)


def main():
  st.set_page_config(layout="wide", page_title="Abstract Analyzer", page_icon="https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/72/twitter/282/beer-mug_1f37a.png", initial_sidebar_state="expanded",)

  st.session_state.summary_models = ("🤖 Baseline TF-IDF","🤗 BERT Extractive (BERT + K-Means)")

  if 'abstracts' not in st.session_state:
    st.session_state.abstracts = preprocess_abstracts_from_file("raw_abstracts.txt")

  if 'selected_abstract' not in st.session_state:
    st.session_state.selected_abstract = st.session_state.abstracts[0]

  if 'summ_model_name' not in st.session_state:
    st.session_state.summ_model_name = st.session_state.summary_models[1]
  
  if 'summary' not in st.session_state:
    st.session_state.summary = get_summary(st.session_state.selected_abstract, model_name=st.session_state.summ_model_name)

  if 'bdown_model_name' not in st.session_state:
    st.session_state.bdown_model_name = "Conv1D"

  if 'bdown_df' not in st.session_state:
    st.session_state.bdown_df = get_abstract_results_df_from_remote_model(st.session_state.bdown_model_name, st.session_state.selected_abstract)

  if 'claims_and_probs' not in st.session_state:
    st.session_state.claims_and_probs = get_extracted_claims_from_remote_model(st.session_state.selected_abstract)

  render_sidebar()

  st.write("""# PubMed Abstract Analyzer 🍺 💊 📄 🔎 """)

  st.success("""
  *Explore how well different NLP models deal with
  summarization, classification and claim extraction, all on one page*
  """)


  st.write("# 📄 Abstract Summary ")
  st.markdown(st.session_state.summary)

  st.write("# 🧐 Claims in the abstract")
  if len(st.session_state.claims_and_probs[0]) > 0:
    for idx,  (claim, probability) in enumerate(zip(st.session_state.claims_and_probs[0],st.session_state.claims_and_probs[1])):
      st.write("{num}. 👉 {claim} (**{probability}**)".format(claim=claim,probability=probability,num=idx+1))
  else:
    st.write("### No claims found! 👀  🤷🏽‍♀️ ")
    st.write("Here's what our model predicts: ")
    st.dataframe(st.session_state.claims_and_probs[2],width=600)


  st.write("# 👩‍🔬 Abstract Breakdown")
  st.markdown(get_abstract_markdown(st.session_state.bdown_df))
  if 'y_pred' in st.session_state.bdown_df.columns:
    st.dataframe(st.session_state.bdown_df.drop(['y_pred'],axis=1),width=600)
  else: 
    st.dataframe(st.session_state.bdown_df,width=600)


  st.write("# ⛅️ Topic modeling ")
  st.markdown(""" See topic modeling experiements in: 
  [Colab Notabook](https://colab.research.google.com/drive/1zbnmjJ0LpOz7VAXoejAolgJTUW63xVw0?usp=sharing)
  """)

  st.write("A random sample of PubMed topic word embeddings:")

  fig = plot_random_topic_words()

  st.plotly_chart(fig, use_container_width=True)



def render_sidebar():

  random_abstract_btn = st.sidebar.button('🤖 Random Abstract Please! 👀')

  if random_abstract_btn:
    st.session_state.selected_abstract = random.choice(st.session_state.abstracts)
    st.session_state.summary = get_summary(st.session_state.selected_abstract, model_name=st.session_state.summ_model_name)
    st.session_state.bdown_df = get_abstract_results_df_from_remote_model(st.session_state.bdown_model_name, st.session_state.selected_abstract)
    st.session_state.claims_and_probs = get_extracted_claims_from_remote_model(st.session_state.selected_abstract)

  render_submit_abstract_form()

  render_playground()

def render_playground():
  
  st.sidebar.markdown("# NLP 🤖 Playground")

  st.sidebar.markdown("### Summarization")

  selected_summ_model = st.sidebar.selectbox(
    'Select model:',
    st.session_state.summary_models,
    index=0,
  )

  if st.session_state.summ_model_name != selected_summ_model:
    st.session_state.summ_model_name = selected_summ_model
    st.session_state.summary = get_summary(st.session_state.selected_abstract, model_name=st.session_state.summ_model_name)

  st.sidebar.markdown("### Abstract breakdown")

  selected_bdown_model_name = st.sidebar.selectbox(
    'Select model:',
    ("🤖 Naive Bayes","🤖 Conv1D w/ custom embeddings"),
    index=1,
  )

  if(st.session_state.bdown_model_name != selected_bdown_model_name):
    st.session_state.bdown_model_name = selected_bdown_model_name

    st.session_state.bdown_df = get_abstract_results_df_from_remote_model(st.session_state.bdown_model_name, st.session_state.selected_abstract)

  # Placeholder for more models to be added
  st.sidebar.markdown("### Claim Extraction")
  st.sidebar.selectbox(
  'Select model:',
  ("🤖 Fine-tuned Conv1D, 90pc accuracy","🤖 More models coming soon!"),
)


def render_submit_abstract_form():

  form = st.sidebar.form(key='my_form')

  user_abstract = form.text_area(label='Copy & paste your own!', value=st.session_state.selected_abstract, height=500)

  submit_button = form.form_submit_button(label='Submit abstract 🤖 ')

  if submit_button:
      st.session_state.selected_abstract = user_abstract
      st.session_state.summary = get_summary(st.session_state.selected_abstract, model_name=st.session_state.summ_model_name)
      st.session_state.bdown_df = get_abstract_results_df_from_remote_model(st.session_state.bdown_model_name, st.session_state.selected_abstract)
      st.session_state.claims_and_probs = get_extracted_claims_from_remote_model(st.session_state.selected_abstract)




if __name__ == "__main__":
    main()

