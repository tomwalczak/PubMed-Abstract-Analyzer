import streamlit as st
import pickle
from streamlit.hashing import _CodeHasher
from SessionState import _SessionState

import plotly.graph_objects as go

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server

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

  state = _get_state()

  state.summary_models = ("🤖 Baseline TF-IDF","🤗 BERT Extractive (BERT + K-Means)")

  state.abstracts = state.abstracts or preprocess_abstracts_from_file("raw_abstracts.txt")

  state.selected_abstract = state.selected_abstract or state.abstracts[0]

  state.summ_model_name = state.summ_model_name or state.summary_models[1]
  state.summary = state.summary or get_summary(state.selected_abstract, model_name=state.summ_model_name)

  state.bdown_model_name = state.bdown_model_name or "Conv1D"

  if state.bdown_df is None:
    state.bdown_df = get_abstract_results_df_from_remote_model(state.bdown_model_name, state.selected_abstract)

  if state.claims_and_probs is None:
    state.claims_and_probs = get_extracted_claims_from_remote_model(state.selected_abstract)

  render_sidebar(state)

  st.write("""# PubMed Abstract Analyzer 🍺 💊 📄 🔎 """)

  st.success("""
  *Explore how well different NLP models deal with
  summarization, classification and claim extraction, all on one page*
  """)


  st.write("# 📄 Abstract Summary ")
  st.markdown(state.summary)

  st.write("# 🧐 Claims in the abstract")
  if len(state.claims_and_probs[0]) > 0:
    for idx,  (claim, probability) in enumerate(zip(state.claims_and_probs[0],state.claims_and_probs[1])):
      st.write("{num}. 👉 {claim} (**{probability}**)".format(claim=claim,probability=probability,num=idx+1))
  else:
    st.write("### No claims found! 👀  🤷🏽‍♀️ ")
    st.write("Here's what our model predicts: ")
    st.dataframe(state.claims_and_probs[2],width=600)


  st.write("# 👩‍🔬 Abstract Breakdown")
  st.markdown(get_abstract_markdown(state.bdown_df))
  if 'y_pred' in state.bdown_df.columns:
    st.dataframe(state.bdown_df.drop(['y_pred'],axis=1),width=600)
  else: 
    st.dataframe(state.bdown_df,width=600)


  st.write("# ⛅️ Topic modeling ")
  st.markdown(""" See topic modeling experiements in: 
  [Colab Notabook](https://colab.research.google.com/drive/1zbnmjJ0LpOz7VAXoejAolgJTUW63xVw0?usp=sharing)
  """)

  st.write("A random sample of PubMed topic word embeddings:")

  fig = plot_random_topic_words()

  st.plotly_chart(fig, use_container_width=True)

  state.sync()

def render_sidebar(state):

  random_abstract_btn = st.sidebar.button('🤖 Random Abstract Please! 👀')

  if random_abstract_btn:
    state.selected_abstract = random.choice(state.abstracts)
    state.summary = get_summary(state.selected_abstract, model_name=state.summ_model_name)
    state.bdown_df = get_abstract_results_df_from_remote_model(state.bdown_model_name, state.selected_abstract)
    state.claims_and_probs = get_extracted_claims_from_remote_model(state.selected_abstract)

  render_submit_abstract_form(state)

  render_playground(state)

def render_playground(state):
  
  st.sidebar.markdown("# NLP 🤖 Playground")

  st.sidebar.markdown("### Summarization")

  selected_summ_model = st.sidebar.selectbox(
    'Select model:',
    state.summary_models,
    index=0,
  )

  if state.summ_model_name != selected_summ_model:
    state.summ_model_name = selected_summ_model
    state.summary = get_summary(state.selected_abstract, model_name=state.summ_model_name)

  st.sidebar.markdown("### Abstract breakdown")

  selected_bdown_model_name = st.sidebar.selectbox(
    'Select model:',
    ("🤖 Naive Bayes","🤖 Conv1D w/ custom embeddings"),
    index=1,
  )

  if(state.bdown_model_name != selected_bdown_model_name):
    state.bdown_model_name = selected_bdown_model_name

    state.bdown_df = get_abstract_results_df_from_remote_model(state.bdown_model_name, state.selected_abstract)

  # Placeholder for more models to be added
  st.sidebar.markdown("### Claim Extraction")
  st.sidebar.selectbox(
  'Select model:',
  ("🤖 Fine-tuned Conv1D, 90pc accuracy","🤖 More models coming soon!"),
)


def render_submit_abstract_form(state):

  form = st.sidebar.form(key='my_form')

  user_abstract = form.text_area(label='Copy & paste your own!', value=state.selected_abstract, height=500)

  submit_button = form.form_submit_button(label='Submit abstract 🤖 ')

  if submit_button:
      state.selected_abstract = user_abstract
      state.summary = get_summary(state.selected_abstract, model_name=state.summ_model_name)
      state.bdown_df = get_abstract_results_df_from_remote_model(state.bdown_model_name, state.selected_abstract)
      state.claims_and_probs = get_extracted_claims_from_remote_model(state.selected_abstract)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()

