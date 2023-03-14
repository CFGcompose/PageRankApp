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
st.title("Get Recommondation for user")

# first section Penguins Dataset
st.header("Page Rank")
st.image("movielense.jpeg")
st.write(" Hey, which user do you are?")


dataset = load_netset('movielens')

biadjacency = dataset.biadjacency
names = dataset.names
labels = dataset.labels
names_labels = dataset.names_labels

positive = biadjacency >= 3


pagerank = PageRank()

user = st.selectbox("Select an user", [1,2,3,4])
#user = 1

st.markdown(f"I'm user {user} and these movies I like most")
targets = get_neighbors(positive, user, transpose=True)

st.markdown(names[targets])

scores = pagerank.fit_predict(positive)

# seen movies (sample)
top_10 = names[targets][:10]

mask = np.zeros(len(names), dtype=bool)
mask[targets] = 1

scores_ppr = pagerank.fit_predict(positive, seeds=mask)

st.header("here are 5 movies you should watch")
# top-10 recommendation
names[top_k((scores_ppr - scores) * (1 - mask), 5)]








