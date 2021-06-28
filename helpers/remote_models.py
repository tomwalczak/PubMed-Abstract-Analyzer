import streamlit as st
import requests
import numpy as np

from helpers.abstract_helpers import get_model_preds_as_df

model_base_url = st.secrets["api_base_url"] 

def get_abstract_results_df_from_remote_model(model_name,full_abstract):

    results = get_remote_model_results(model_name,full_abstract)


    return get_model_preds_as_df(None,results["preds"],results["class_names"],results["sentences"])


def get_remote_model_results(model_name,full_abstract):

  if "Naive Bayes" in model_name: 
    r = requests.post(model_base_url + "/v1/nb_breakdown/predict", 
      json={ "data": full_abstract})


  elif  "Conv1D" in model_name:   
    r = requests.post(model_base_url + "/v1/conv1d_breakdown/predict", 
      json={ "data": full_abstract})

    
  return {"preds": np.array(r.json()['preds']), "sentences":r.json()['sentences'], "class_names":r.json()['class_names']}
    