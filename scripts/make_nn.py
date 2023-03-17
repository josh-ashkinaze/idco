"""
Author: Joshua Ashkinaze
Date: 03/17/2023

Description:
* This script trains the Word2Vec model and calculates the tf-idf scores for each subreddit.
* It outputs the models and tf-idf scores to the models and tfidf folders, respectively.
"""

import json
from gensim.models import Word2Vec
import os
import logging
from wordfreq import word_frequency

def calculate_tfidf(sub_comments):
    # Calculate term frequencies in the subreddit corpus
    term_frequencies = {}
    total_words = 0
    for comment in sub_comments:
        for word in comment:
            term_frequencies[word] = term_frequencies.get(word, 0) + 1
            total_words += 1

    # Calculate tf-idf scores
    tfidf_dict = {}
    for word, count in term_frequencies.items():
        tf = count / total_words
        word_freq = word_frequency(word, 'en', wordlist='best', minimum=0.0)
        idf = 1 / word_freq if word_freq > 0 else 1
        tfidf = tf * idf
        tfidf_dict[word] = tfidf

    return tfidf_dict


def train_word2vec_model(comments, size=100, window=5, min_count=5, workers=4, sg=1):
    model = Word2Vec(comments, vector_size=size, window=window, min_count=min_count, workers=workers, sg=sg)
    return model

def main():
    log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', format='%(asctime)s %(message)s')

    with open('../data/processed_comments.json', 'r') as infile:
        comments_by_sub = json.load(infile)

    if not os.path.exists('../data/models'):
        os.makedirs('../data/models')

    if not os.path.exists('../data/tfidf'):
        os.makedirs('../data/tfidf')

    n_subs = len(comments_by_sub)
    sub_counter = 0
    for sub, comments in comments_by_sub.items():
        logging.info(f'Training Word2Vec model for r/{sub}, which is {sub_counter} of {n_subs}...')
        model = train_word2vec_model(comments)
        model.save(f'../../data/models/w2v_{sub}.model')
        logging.info(f'Model for r/{sub} saved.')
        logging.info(f'Getting tfidf scores for r/{sub}, which is {sub_counter} of {n_subs}...')
        tfidf_dict = calculate_tfidf(comments)
        with open(f'../../data/tfidf/tfidf_{sub}.json', 'w') as outfile:
            json.dump(tfidf_dict, outfile)
        logging.info(f'tfidf for r/{sub} saved.')
        sub_counter +=1
    logging.info("w2v + tfidf saved")

if __name__ == "__main__":
    main()
