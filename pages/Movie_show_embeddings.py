"""Here modeling"""

### Import libraries
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64

from sknetwork.data import load_netset
from sknetwork.ranking import PageRank, top_k
from sknetwork.embedding import Spectral
from sknetwork.embedding import SVD
from sknetwork.utils import WardDense, get_neighbors
from sknetwork.visualization import svg_dendrogram


## Data exploration 
# Let's give a title
st.title("Get Spectral Embedding")

dataset = load_netset('movielens')

biadjacency = dataset.biadjacency
names = dataset.names
labels = dataset.labels
names_labels = dataset.names_labels

# positive ratings
positive = biadjacency >= 3

embedding_dim = st.slider('Select Embedding Dimension', 2,15,10)
n_cluster= st.slider('Select n_cluster', 2,8,6)

pagerank = PageRank()
scores = pagerank.fit_predict(positive)

spectral = Spectral(embedding_dim)
embedding = spectral.fit_transform(positive)    
   
        
ward = WardDense()# top-100 movies
scores = pagerank.fit_predict(positive)
index = top_k(scores, 100)
dendrogram = ward.fit_transform(embedding[index])

image = svg_dendrogram(dendrogram, names=names[index], rotate=True, width=200, height=1000, n_clusters=n_cluster)
st.image(image)   
        
        
