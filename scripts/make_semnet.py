import json
import numpy as np
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
    word_vectors = [model.wv[word] for word in model.wv.index_to_key]
    if len(word_vectors) < 2:
        return None
    sim_matrix = cosine_similarity(word_vectors)
    return sim_matrix

def create_graph(words, sim_matrix):
    G = nx.Graph()
    for i in range(len(words)):
        G.add_node(words[i])
        for j in range(i + 1, len(words)):
            similarity = sim_matrix[i][j]
            if similarity > 0:
                G.add_edge(words[i], words[j], weight=similarity)
    return G

def main():
    log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', format='%(asctime)s %(message)s')

    if not os.path.exists('../data/semnets'):
        os.makedirs('../data/semnets')

    with open('../data/processed_comments.json', 'r') as infile:
        subreddits = json.load(infile)

    for subreddit in subreddits:
        print(f"Processing subreddit {subreddit}...")
        model = load_word2vec(subreddit)
        if model is not None:
            words = model.wv.index_to_key
            sim_matrix = cosine_similarity_matrix(model)
            if sim_matrix is not None:
                G = create_graph(words, sim_matrix)
                nx.write_gpickle(G, f'../data/semnets/semnet_{subreddit}.gpickle')
                print(f"Graph for subreddit {subreddit} saved")

if __name__ == '__main__':
    main()
