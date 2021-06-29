import streamlit as st
import requests
import numpy as np
import pandas as pd

model_base_url = st.secrets["api_base_url"] 

def get_extracted_claims_from_remote_model(full_abstract):

    r = requests.post(model_base_url + "/v1/claim_extraction/predict", 
      json={ "data": full_abstract})
    r = r.json()

    df = pd.DataFrame({"sentence": r["sentences"], "preds": r["preds"]})
    df["claim_or_not"] = df["preds"] > 0.5

    claims_df = df[df["claim_or_not"]==True]
    claims = claims_df["sentence"].tolist()
    probs = [ "{p}%".format(p = round(prob*100,2)) for prob in claims_df["preds"].tolist()] 

    return claims, probs, df