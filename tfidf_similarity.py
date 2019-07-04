# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:56:22 2019

@author: luanpo1234
"""

import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(text):
    #Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    #Text to lower
    text = text.lower()
    return text

def vectorize_sim_search(query, data, stop_words="english"):
    """
    query: `str`
    data: `list` of strings
    stop_words: `list` of stop words (default is "english" from SKlearn)
    
    Returns:
    cos_sim: first element from `cosine_similarity` array
    tfidf_vectorizer: `TfidfVectorizer object`
    data: updated `data` with `query` in position 0
    """
    #Inserting query in list position 0
    data = [query] + data
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    cos_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
    print(type(cos_sim))
    return cos_sim[0], tfidf_vectorizer, data

def get_most_similar(data, cos_sim, n):
    """
    data: `list` of strings (updated in `vectorize_sim_search()`)
    cos_sim: first element from `cosine_similarity` array
    n: `int` with number of elements to return
    
    Returns:
    most_similar: `list` of tuples with `n` most similar documents arranged from most similar to less similar.
    Each `tuple` contains one similarity value `float` and one `str` document.
    """
    sorted_indexes = []
    index = 0
    for el in cos_sim:
        sorted_indexes.append((index, el))
        index += 1
    #índices de `data`, organizando do mais similar para o menos similar
    sorted_indexes = sorted(sorted_indexes, key = lambda x: x[1], reverse = True)
    most_similar = []
    for n in range(n):
        most_similar.append((sorted_indexes[n][1], data[sorted_indexes[n][0]]))
    return most_similar