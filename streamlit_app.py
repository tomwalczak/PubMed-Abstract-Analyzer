import streamlit as st

st.set_page_config(layout="wide", page_title="Abstract Analyzer", page_icon="https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/72/twitter/282/beer-mug_1f37a.png", initial_sidebar_state="expanded",)

url = 'http://tomwalczak.com/pubmed-abstract-analyzer/'

st.markdown('## I have moved the project to: [LINK]({})'.format(url))