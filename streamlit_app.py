import streamlit as st

st.write("""
# PubMed Topic Transformer 🍺💊☁️🤗

Hello World """)


form = st.form(key='my_form')
name = form.text_input(label='Enter your name')
submit_button = form.form_submit_button(label='Submit')

# st.form_submit_button returns True upon form submit
if submit_button:
    st.write(f'hello {name}')

button_state = st.button('Random Paper Please! 🤗')



st.write(f'Button state{button_state}')
