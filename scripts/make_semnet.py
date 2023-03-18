"""
Author: Joshua Ashkinaze
Date: 03/17/2023

Description: Creates graphs from w2v models
"""

import json
import os
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import logging
def load_word2vec(subreddit):
    model_path = f'../data/models/w2v_{subreddit}.model'
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
        return model
    else:
        print(f"No model found for subreddit {subreddit}. Skipping...")
        return None

def cosine_similarity_matrix(model):
    """Creates a matrix of cosine similarities between all words in the model"""
    word_vectors = [model.wv[word] for word in model.wv.index_to_key]
    if len(word_vectors) < 2:
        return None
    sim_matrix = cosine_similarity(word_vectors)
    return sim_matrix

def create_graph(words, sim_matrix):
    """Creates a graph from a list of words and a matrix of cosine similarities"""
    G = nx.Graph()
    for i in range(len(words)):
        G.add_node(words[i])
        for j in range(i + 1, len(words)):
            similarity = sim_matrix[i][j]
            if similarity > 0:
                G.add_edge(words[i], words[j], weight=similarity)
    return G


def subgraph_by_edge_weight(G, threshold):
    """Returns a subgraph of G that only contains edges with weight above threshold"""
    subgraph_edges = [(u, v) for u, v, w in G.edges(data=True) if w['weight'] >= threshold]
    subgraph = G.edge_subgraph(subgraph_edges)
    return subgraph


def main():
    log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', format='%(asctime)s %(message)s')

    if not os.path.exists('../data/semnets'):
        os.makedirs('../data/semnets')

    with open('../data/processed_comments.json', 'r') as infile:
        subreddits = json.load(infile)

    for subreddit in subreddits:
        logging.info(f"Processing subreddit {subreddit}...")
        model = load_word2vec(subreddit)
        if model is not None:
            words = model.wv.index_to_key
            sim_matrix = cosine_similarity_matrix(model)
            if sim_matrix is not None:
                G = create_graph(words, sim_matrix)
                nx.write_gpickle(G, f'../data/semnets/semnet_{subreddit}.gpickle')
                logging.info(f"Graph for subreddit {subreddit} saved")
                logging.info(f"Making subgraph with 0.25 for {subreddit} saved")
                sub = subgraph_by_edge_weight(G, 0.25)
                nx.write_gpickle(G, f'../data/semnets/semnet_25_{subreddit}.gpickle')

    logging.info("Done")
if __name__ == '__main__':
    main()
