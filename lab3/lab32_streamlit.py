import streamlit as st
import json
import requests
st.title("Basic Caculator App ")

# taking user imputs
option = st.selectbox('What operation You want to perform?',['Addition', 'Subtraction', 'Multiplication', 'Division'])
st.write("")
st.write("Select the numbers from slider below ")
x = st.slider("X", 0, 100, 20)
y = st.slider ("Y", 0, 130, 10)
#converting the inputs into a json format
inputs = {"operation": option,"x": x, "у": y}
# when the user clicks on button it will fetch the API
if st.button ('Calculate'):
    # res = requests.post(url = "http://127.0.0.1:8000/calculate", data = json.dumps(inputs))
    res = requests.get(url = "http://127.0.0.1:8000/")
    st.subheader (f"Response from API = {res .text}")