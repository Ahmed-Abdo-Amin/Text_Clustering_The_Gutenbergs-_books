
# Text Clustering (The Gutenbergs' books)

Text Clustering is one of the major tasks of AI in general, NLP in particular. Having five Gutenberg books, this report discusses the methodologies and models with different transformation techniques that have been applied to reach the best accuracy that the champion model achieves by correctly classifying unseen text to the corresponding book.

## Introduction
Text can be a rich source of information, however extracting insights from it is not an easy task due to its unstructured nature. The overall objective is to produce classification predictions and compare them; analyze the pros and cons of algorithms and generate and communicate the insights so, the implementation is needed to check the accuracy of each model going to be used and select the champion model. the best language to be used in such problems is python with its vast libraries.

## Dataset
The Gutenberg dataset represents a corpus of over 60,000 book texts, their authors and titles. The data has been scraped from the Project Gutenberg website using a custom script to parse all bookshelves. we have taken five different samples of Gutenberg digital books that are of five different authors, that we think are of the criminal same genre and are semantically the same. The books are Criminal Sociology by Enrico Ferri. Crime: Its Cause and Treatment by Clarence Darrow. The Pirates' Who's Who by Philip Gosse. Celebrated Crimes by Alexandre Dumas and Buccaneers and Pirates of Our Coasts by Frank Richard. 

## Installation 
import nltk
import pandas as pd
import re
import numpy as np
nltk.download("gutenberg")
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,5)
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import cohen_kappa_score
from sklearn. preprocessing import StandardScaler
from sklearn. cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

#  Solution (G10_TextClustering.ipynb)
### This contains a series of steps to create a model that give some answers as ("who is author that writes this sentence?"):
       1. Use 5 books of different types ("Fiction | Poetry | Mystery | Drama | Children")
	   1. Make Preprocessing and Data Cleaning
			- Have 1000 rows each row has 100 words.
			- Use label encoder "on Author".
			- Clean the data using regex.
			- Remove the stop words too.
       2. Use Feature Engineering ("Feature selection" and "Features Reduction")
			- Use "SelectPercentile" for the most 20% important features.
			- Use "BOW", "N-Gram", and "TFiDF" to create new features.
       3. Train 3 algorithms with 3 transformer
	   4. Evaluate each model "cv ,mean accuracy and std
	   5. Display bais and variance for All models to choose the champion model.
	   6. Select The champion model ("Expectation maximization (EM) With LDA-BOW")
	   7. Apply Error Analysis on The champion model.
	       -Identify where the word belongs to each topic and this led to the misclassification of some of the words
## Thank You
Ahmed Abdo Amin Abdo
