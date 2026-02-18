import streamlit as st

st.set_page_config(page_title="Hello AI World", layout="centered")

st.title("👋 Hello AI World")
st.write("This is a sample Streamlit app deployed on Azure.")

user_input = st.text_input("Ask something to AI:")

if user_input:
    st.success(f"AI says: Hello! You asked → '{user_input}'")
