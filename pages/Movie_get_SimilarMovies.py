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

import numpy as np

from sknetwork.data import load_netset
from sknetwork.ranking import PageRank, top_k
from sknetwork.embedding import Spectral
from sknetwork.utils import get_neighbors
from sknetwork.visualization import svg_dendrogram

## Data exploration 
# Let's give a title
st.title("Get similar Movies")

# first section Penguins Dataset
st.header("Page Rank")
st.image("movielense.jpeg")
st.write(" Hey, which movie do you like?")


dataset = load_netset('movielens')

biadjacency = dataset.biadjacency
names = dataset.names
labels = dataset.labels
names_labels = dataset.names_labels

positive = biadjacency >= 3


pagerank = PageRank()

scores = pagerank.fit_predict(positive)
sel_movie = st.selectbox("Select an movie", names[0:10])
#user = 1

st.markdown(f"Ok, you like the movie:")

sel_movie

movie_id =  list(names).index(sel_movie)

scores_ppr = pagerank.fit_predict(positive, seeds={movie_id:1})

st.header("Than these 5 movies are very similar maybe even the same")
names[top_k(scores_ppr - scores, 5)]



