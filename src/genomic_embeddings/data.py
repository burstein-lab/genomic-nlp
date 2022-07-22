import pandas as pd
import numpy as np
import re

# word 2 vec
from gensim.models import word2vec as w2v


class Embedding(object):
    def __init__(self, mdl, metadata, labels=None):
        self.mdl = w2v.Word2Vec.load(mdl)
        self.metadata = pd.read_csv(metadata)
        self.labels = labels
        self.embedding = self.mdl.wv.vectors.astype('float64')
        self.known_embeddings = None
        self.word2index = None
        self.effective_words = None
        self.train_words = None
        self.data = None
        self.unknown_embeddings = None
        self.unknown_word2index = None
        self.data_with_words = None

    def extract_known_words(self, unknown="hypo.clst"):
        idxs = [self.mdl.wv.vocab[word].index for word in self.mdl.wv.vocab if unknown not in word]
        known_mat = self.embedding[idxs]
        known_word2index = {self.mdl.wv.index2word[word]: i for i, word in enumerate(idxs)}

        self.known_embeddings = known_mat
        self.word2index = known_word2index

    def extract_effective_words(self, label='label'):
        metadata = self.metadata
        metadata[label] = metadata[label].apply(lambda x: re.split('(.)\[|\(|,', x)[0].strip()) #remove redundant
        eff_words = pd.DataFrame(self.word2index.items(), columns=["word","index"])
        eff_words["KO"] = eff_words["word"].apply(lambda x: x.rsplit(".")[0])
        eff_words = eff_words.merge(metadata, on=["KO"], how='left')[["word","index",label]].dropna()
        self.effective_words = eff_words

    def filter_effective_words(self, q=0.96, label='label'):
        eff_words = self.effective_words
        labels = self.labels
        if labels is None:
            labels_count = eff_words.groupby(label).size().reset_index(name="size").\
                sort_values(by="size", ascending=False)
            labels_to_keep = labels_count[labels_count["size"] >= np.quantile(labels_count["size"], q)]
            labels_to_keep = labels_to_keep[
                ~labels_to_keep[label].isin(["Function unknown [99997]", "Enzymes with EC numbers [99980]"])]
            labels = labels_to_keep[label].values

        eff_words = eff_words[eff_words[label].isin(labels)]
        self.train_words = eff_words

        if self.labels is None:
            self.labels = eff_words[label].unique()

    def add_other_class(self, label='label', sample_size=12, min_points=30):
        eff_words = self.effective_words
        eff_words[label] = eff_words[label].apply(lambda x: re.split('(.)\[|\(|,', x)[0].strip())  # remove redundant

        label_sizes = eff_words.groupby('label').size().reset_index(name='size')
        labels_to_keep = self.labels

        sample_from = label_sizes[~label_sizes["label"].isin(labels_to_keep)].sort_values(by='size', ascending=False)
        labels_to_sample_from = sample_from[sample_from['size'] > min_points]['label']

        other_class = eff_words[eff_words['label'].isin(labels_to_sample_from)].groupby("label").sample(n=sample_size,
                                                                                             random_state=42)
        other_class['label'] = 'Other'
        data_with_other = pd.concat([self.train_words, other_class])
        self.train_words = data_with_other


    def cleanup_train_data(self, add_other=False):
        df = pd.DataFrame(self.known_embeddings)
        if add_other:
            self.add_other_class()
        df = df.reset_index().merge(self.train_words, on="index", how="right")
        self.data_with_words = df
        self.data = df.drop(columns=["index", "word"])


    def process_unknown_words(self, labels2filter, label='label'):
        meta = self.metadata
        meta['label'] = meta['label'].apply(lambda x: re.split('(.)\[|\(|,', x)[0].strip())
        if labels2filter is None:
            labels2filter = meta[label].unique()
        train_words = meta[meta[label].isin(labels2filter)]['KO'].values
        test_embeddings_idx = {word: self.mdl.wv.vocab[word].index for word in self.mdl.wv.vocab if
                               word not in train_words}
        unknown_embs = self.embedding[[*test_embeddings_idx.values()]]

        self.unknown_embeddings = unknown_embs
        self.unknown_word2index = test_embeddings_idx

    def process_data_pipeline(self, label, q, add_other=False):
        self.extract_known_words()
        self.extract_effective_words(label=label)
        self.filter_effective_words(q=q, label=label)
        self.cleanup_train_data(add_other=add_other)





