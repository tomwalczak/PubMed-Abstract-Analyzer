import streamlit as st
import numpy as np
import joblib
import random

from tensorflow import keras

from helpers.abstract_helpers import get_model_preds_as_df, get_abstract_results_df_nb, preprocess_abstracts_from_file, get_abstract_markdown, get_abstract_results_df, add_positon_feature_to_sentences

st.set_page_config(layout="wide")


def load_models():
  nb_abstract = joblib.load("./saved_models/abstract_highlight_model_nb_pos.sav")

  saved_model_name = 'abstract_bd_conv1d_90acc'

  conv_abstract = keras.models.load_model('./saved_models/' + saved_model_name)

  return nb_abstract, conv_abstract


nb_abstract, conv_abstract = load_models()

class_names = np.array(['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS'])

abstracts = preprocess_abstracts_from_file("raw_abstracts.txt")



st.write("""
# PubMed Topic Transformer 🍺💊☁️🤗

""")

selected_abstract = ""

if selected_abstract == "":
  selected_abstract = random.choice(abstracts)

# form = st.form(key='my_form')
# name = form.text_input(label='Enter your name')
# submit_button = form.form_submit_button(label='Submit')

# st.form_submit_button returns True upon form submit
# if submit_button:
#     st.write(f'hello {name}')

button_state = st.button('Random Abstract Please! 🤗')

if button_state:
  selected_abstract = random.choice(abstracts)
# st.write(f'Button state is: {button_state}')





form = st.form(key='my_form')



user_abstract = form.text_area(label='Abstract text', value=selected_abstract, height=300)

submit_button = form.form_submit_button(label='Submit your own! 🤗')

if submit_button:
    selected_abstract = user_abstract


st.write("# Naive Bayes Classifier")

nb_abstract_results_df = get_abstract_results_df_nb(nb_abstract, class_names, selected_abstract)


st.markdown(get_abstract_markdown(nb_abstract_results_df))
# nb_preds = md_abstract.predict(selected_abstract)

# preds_df = pd.DataFrame({
#     'Sentence': selected_abstract,
#     'Prediction': class_names[nb_preds]
# })

# st.write(preds_df)

st.markdown("# Convolutional Classifier with Custom Embeddings")

abstract_bd_df = get_abstract_results_df(conv_abstract,class_names,selected_abstract)


st.markdown(get_abstract_markdown(abstract_bd_df))