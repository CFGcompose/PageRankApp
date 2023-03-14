"""TODO
   Excercise: Explore the penguins pimped dataset :-) by following 
   the tasks. The data exploration would be the base on which we build 
   the streamlit app.

   Usage:
   After each task save the script and run the python script from the terminal.
   Be sure you are in the streamlit_env conda environment.
   If not activate the environment by typing in the terminal
   conda activate streamlit_env

   Mode:
   Work in group.
   
"""

### Import libraries
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64

## Data exploration 
# Let's give a title
st.title("Recommondations with PageRank")


path_to_html = "./gameofthrones.html" 

# Read file and keep in variable
with open(path_to_html,'r') as f: 
    html_data = f.read()

## Show in webpage
st.header("Graph Structure")
st.components.v1.html(html_data,height=600)


st.header("Distribution of Ratings")

st.image("Ratings_Movielense.png")





