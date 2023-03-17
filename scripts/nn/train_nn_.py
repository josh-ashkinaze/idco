"""
Author: Joshua Ashkinaze
Date: 03/17/2023

Description: This script trains a Word2Vec model for each subreddit in the dataset. It outputs the models to the models directory.
"""

import json
from gensim.models import Word2Vec
import os
import logging
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('brown')
nltk.download('wordnet')


def preprocess_brown_corpus():
    lemmatizer = WordNetLemmatizer()
    brown_lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in brown.words()]
    brown_corpus = ' '.join(brown_lemmatized_words)
    return brown_corpus


def calculate_tfidf(sub_comments, brown_corpus):
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer on the Brown corpus
    vectorizer.fit([brown_corpus])

    # Transform the subreddit comments using the fitted vectorizer
    tfidf_matrix = vectorizer.transform([' '.join(comment) for comment in sub_comments])
    feature_names = vectorizer.get_feature_names()
    tfidf_dict = dict(zip(feature_names, np.mean(tfidf_matrix, axis=0).tolist()[0]))
    return tfidf_dict

def train_word2vec_model(comments, size=100, window=5, min_count=5, workers=4, sg=1):
    model = Word2Vec(comments, vector_size=size, window=window, min_count=min_count, workers=workers, sg=sg)
    return model

def main():
    log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', format='%(asctime)s %(message)s')
    brown_corpus = preprocess_brown_corpus()

    with open('../../data/processed_comments.json', 'r') as infile:
        comments_by_sub = json.load(infile)

    if not os.path.exists('../../data/models'):
        os.makedirs('../../data/models')

    if not os.path.exists('../../data/models'):
        os.makedirs('../../data/tfidf')

    n_subs = len(comments_by_sub)
    sub_counter = 0
    for sub, comments in comments_by_sub.items():
        logging.info(f'Training Word2Vec model for r/{sub}, which is {sub_counter} of {n_subs}...')
        model = train_word2vec_model(comments)
        model.save(f'../../data/models/w2v_{sub}.model')
        logging.info(f'Model for r/{sub} saved.')
        logging.info(f'Getting tfidf scores for r/{sub}, which is {sub_counter} of {n_subs}...')
        tfidf_dict = calculate_tfidf(comments, brown_corpus)
        with open(f'../../data/tfidf/tfidf_{sub}.json', 'w') as outfile:
            json.dump(tfidf_dict, outfile)
        logging.info(f'tfidf for r/{sub} saved.')
        sub_counter +=1
    logging.info("w2v + tfidf saved")

if __name__ == "__main__":
    main()
