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

import numpy as np
import joblib
import random

from tensorflow import keras

from helpers.abstract_helpers import preprocess_abstracts_from_file, get_abstract_markdown, get_abstract_results_df, add_positon_feature_to_sentences
from helpers.summary_helpers import get_summary

from helpers.topic_helper import detect_topic, get_word_embeddings_df, plot_topic_centroid_in_context

with open("./lib/punkt/PY3/english.pickle","rb") as resource:
  sent_detector = pickle.load(resource)


def main():
  st.set_page_config(layout="wide")

  state = _get_state()

  state.summary_models = ("🤖 Baseline TF-IDF","🤗 BERT Extractive (BERT + K-Means)")



  state.summ_class_names = np.array(['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS'])

  state.abstracts = state.abstracts or preprocess_abstracts_from_file("raw_abstracts.txt")

  state.selected_abstract = state.selected_abstract or state.abstracts[0]

  state.summ_model_name = state.summ_model_name or state.summary_models[1]
  state.summary = state.summary or get_summary(state.selected_abstract, model_name=state.summ_model_name)
  
  if state.word_embeddings_df is None:
    state.word_embeddings_df, state.fitted_pca = get_word_embeddings_df()

  # state.bdown_model_name = state.bdown_model_name or "Conv1D"
  # state.bdown_model = state.bdown_model or load_bdown_model(state.bdown_model_name)

  # if state.bdown_df is None:
  #   state.bdown_df = get_abstract_results_df(state.bdown_model, sent_detector, state.summ_class_names, state.selected_abstract)


  ####################################################################################

  render_sidebar(state)

  st.write("""# PubMed Abstract Analyzer 🍺 💊 📄 🔎 """)



  st.write("# 📄 Summary ")
  st.markdown(state.summary)



  st.write("# 🧐 Claims in the paper")
  st.markdown(""" 
  - Baclofen may represent a clinically relevant alcohol pharmacotherapy 
  - Substantial progress has recently been made in optimizing the management of cancer patients""")





  st.write("# 👩‍🔬 Abstract Breakdown - recompile models")
  # st.markdown(get_abstract_markdown(state.bdown_df))



  st.write("# ⛅️ Topics ")
  topic_name, topic_words = detect_topic(state.selected_abstract)
  st.write("Abstract topic: **{topic_name} **".format(topic_name=topic_name))
  st.write("Relevant topic words")
  st.code(' '.join(topic_words))


  st.plotly_chart(plot_topic_centroid_in_context(topic_name,state.word_embeddings_df,state.fitted_pca), use_container_width=True)

  state.sync()

def render_sidebar(state):
  
  st.sidebar.markdown("""
  *Explore how well different NLP models deal with
  summarization, classification and information extraction, all in one place.*
  """)

  random_abstract_btn = st.sidebar.button('🤖 Random Abstract Please! 👀')

  if random_abstract_btn:
    state.selected_abstract = random.choice(state.abstracts)
    state.summary = get_summary(state.selected_abstract, model_name=state.summ_model_name)
    state.bdown_df = get_abstract_results_df(state.bdown_model, sent_detector, state.summ_class_names, state.selected_abstract)

  render_submit_abstract_form(state)

  render_playground(state)

def render_playground(state):
  
  st.sidebar.markdown("# NLP 🤖 Playground")

  st.sidebar.success("""
  *Select from traditional ML models as well as state-of-the-art LSTM, BERT and T5 - see how they compare! *
  """)

  st.sidebar.markdown("### Summarization")

  selected_summ_model = st.sidebar.selectbox(
    'Select model:',
    state.summary_models,
    index=0,
  )

  if state.summ_model_name != selected_summ_model:
    state.summ_model_name = selected_summ_model
    state.summary = get_summary(state.selected_abstract, model_name=state.summ_model_name)

  # st.sidebar.markdown("### Abstract breakdown")

  # selected_bdown_model_name = st.sidebar.selectbox(
  #   'Select model:',
  #   ("🤖 Naive Bayes","🤖 Conv1D w/ custom embeddings"),
  #   index=1,
  # )

  # if(state.bdown_model_name != selected_bdown_model_name):
  #   state.bdown_model_name = selected_bdown_model_name
  #   state.bdown_df = get_abstract_results_df(state.bdown_model, sent_detector, state.summ_class_names, state.selected_abstract)

  # st.sidebar.checkbox("Use the output of Breakdown to help with summarization")  

def render_submit_abstract_form(state):

  form = st.sidebar.form(key='my_form')

  user_abstract = form.text_area(label='Copy & paste your own!', value=state.selected_abstract, height=150)


  submit_button = form.form_submit_button(label='Submit abstract 🤖 ')

  if submit_button:
      state.selected_abstract = user_abstract


## TODO: move to helpers
# @st.cache(allow_output_mutation=True)
# def load_abd_nb():
#   return joblib.load("./saved_models/abstract_highlight_model_nb_pos.sav")
# @st.cache
# def load_abd_conv1D():
#   return keras.models.load_model('./saved_models/' + 'abstract_bd_conv1d_90acc')


# def load_bdown_model(model_name):
#   # TODO! 
#   if "Naive Bayes" in model_name:
#     return load_abd_conv1D()
  
#   elif  "Conv1D" in model_name:
#     return load_abd_conv1D()

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

