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
st.title("Get Recommondation for Article")

# first section Penguins Dataset
st.header("Cora with Personalised Page Rank")
st.image("cora_citations.png")
st.write(" Hey, which article do you are?")


dataset = load_netset('cora')

adjacency = dataset.adjacency
names = dataset.names
labels = dataset.labels
names_labels = dataset.names_labels


pagerank = PageRank()
scores = pagerank.fit_predict(adjacency)

article = st.selectbox("Select an article", names)

article_id = list(names).index(article)


category = names_labels[labels[article_id]]


st.markdown(f"I'm article no {article} and these is my category: {category}")

targets = get_neighbors(adjacency, article_id, transpose=True)
neigh_cats = names_labels[labels[targets][:10]]

st.markdown(f"These are categories that I have citiation from")

neigh_cats

mask = np.zeros(len(names), dtype=bool)
mask[targets] = 1

scores_ppr = pagerank.fit_predict(adjacency, seeds=mask)
#recom_names = names[top_k((scores_ppr - scores)*(1-mask), 10)]
recom_labels = names_labels[labels[top_k((scores_ppr - scores)*(1-mask), 10)]]

st.header(f"here are the categories of articles that you should read:")

recom_labels







