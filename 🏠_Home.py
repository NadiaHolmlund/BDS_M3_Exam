# Imports
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

pd.set_option("display.max_columns", None)

# Setting up page configurations
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide")

st.markdown("<h1 style='text-align: center; color: grey;'>Intelligent Scouting & Player Rating</h1>", unsafe_allow_html=True)
st.write('')

# Loading images and videos only once
@st.experimental_singleton
def read_objects():
    img_1 = Image.open('/work/M3/Images/Home_img_1.png')
    img_2 = Image.open('/work/M3/Images/Home_img_2.png')
    img_3 = Image.open('/work/M3/Images/Home_img_3.png')
    video_1 = open('/work/M3/Images/Home_vid_1.mov', 'rb')
    video_2 = open('/work/M3/Images/Home_vid_2.mov', 'rb')
    video_3 = open('/work/M3/Images/Home_vid_3.mov', 'rb')
    return img_1, img_2, img_3, video_1, video_2, video_3

img_1, img_2, img_3, video_1, video_2, video_3 = read_objects()




# Displaying feature 1
col1, col2 = st.columns(2)

with col1:
    st.image(img_1)

with col2:
    st.write('')
    st.subheader("Player Rating")
    st.write("Predict a player's rating based on their position and the 10 most important features for that position. Explore the SHAP explainer to identify how the different features affect the overall rating.")
    demo_1 = st.button('Demo', key='demo_1')
if demo_1:
    video_bytes_1 = video_1.read()
    st.video(video_bytes_1)
    close_demo_1 = st.button('Close Demo', key='close_demo_1')



# Displaying feature 2
col1, col2 = st.columns(2)

with col1:
    st.image(img_2)

with col2:
    st.write('')
    st.subheader("Player Comparison")
    st.write("Compare a player to the best rated player in the Bundesliga with the same position. Easily identify shortcomings and potential areas of improvement in a spider graph.")
    demo_2 = st.button('Demo', key='demo_2')
if demo_2:
    video_bytes_2 = video_2.read()
    st.video(video_bytes_2)
    close_demo_2 = st.button('Close Demo', key='close_demo_2')



# Displaying feature 3
col1, col2 = st.columns(2)

with col1:
    st.image(img_3)

with col2:
    st.write('')
    st.subheader("Database")
    st.write("Browse through players in the Bundesliga and Superliga within a selected timeframe and examine their information and performance metrics.")
    demo_3 = st.button('Demo', key='demo_3')
if demo_3:
    video_bytes_3 = video_3.read()
    st.video(video_bytes_3)
    close_demo_3 = st.button('Close Demo', key='close_demo_3')