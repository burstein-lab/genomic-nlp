from gensim.models import word2vec as w2v
import pickle
import pandas as pd


def load_embeddings(embedding_mdl):
    """
    load the existing embeddings from trained model
    :param embedding_mdl: the path to a trained w2v model
    :return: w2v object, with embeddings for each word in the corpus
    """
    return w2v.Word2Vec.load(embedding_mdl)

def get_2d_mapping(embedding_2d_path):
    """
    get the 2d coordinates as obtained by umap for each gene in the vocabulary.
    :param embedding_2d_path: a path to pickle file containing the 2d coordinates for each gene
    :return: a dataframe with a word and coordinates <x,y>
    """
    with open(embedding_2d_path, "rb") as handle:
        embedding_2d = pickle.load(handle)
    return embedding_2d

def get_functional_prediction(predicted_hypo_path):
    """
    get a table with all prediction made by the functional model
    :param predicted_hypo_path: a path to the pickle file with the hypothetical proteins
    :return: data frame with predictions for every hypothetical word
    """
    return pd.read_pickle(predicted_hypo_path, 'rb')
