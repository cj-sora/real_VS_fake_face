import streamlit as st
from util import *
from PIL import Image

st.set_page_config(page_title='AI-Generated Face Detector', layout='centered')

st.title("AI-Generated Face Detector")
st.subheader("Upload your image")

user_image = st.sidebar.file_uploader("Upload your image")

if user_image:
    st.image(user_image)
    image = Image.open(user_image)
    img_array = np.array(image)
    results = get_prediction(img_array).lower()
    label2print = "The uploaded picture has a " + results + " face."
    st.subheader(label2print)
