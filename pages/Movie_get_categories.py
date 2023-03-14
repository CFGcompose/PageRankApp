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
st.title("Get best movies per category")

# first section Penguins Dataset
st.header("Page Rank")
st.image("movielense.jpeg")
st.write(" Hey, which category do you like?")


dataset = load_netset('movielens')

biadjacency = dataset.biadjacency
names = dataset.names
labels = dataset.labels
names_labels = dataset.names_labels


positive = biadjacency >= 3


pagerank = PageRank()

cat_name = st.selectbox("Select a category", names_labels)
#user = 1

cat = list(names_labels).index(cat_name)

st.markdown(f"For {cat_name} these 5 are the best")

# top-10 movies
scores = pagerank.fit_predict(positive)
top_10 = names[top_k(scores, 10)]


n_selection = 5

# selection

ppr = pagerank.fit_predict(positive, seeds=labels[:, cat])
scores = ppr * labels[:, cat]
selection = np.array(top_k(scores, n_selection))

# show selection  several genres
names[selection]







