# imports
import codecs
import glob
import logging
import multiprocessing
from tqdm import tqdm
import argparse
import os
import pickle
import sys
import sklearn.manifold
import pandas as pd
from datetime import datetime
import umap

#word 2 vec
from gensim.models import word2vec as w2v


class Corpus(object):
    def __init__(self, dir_path):
        self.dirs = dir_path
        self.len = None
        self.corpus = None
        self.sentences = None
        self.token_count = None

    def load_corpus(self):
        # initialize rawunicode , all text goes here
        corpus_raw = u""
        files = glob.glob(self.dirs)
        files.sort()
        print(f"Number of files in corpus: {len(files)}")
        for f in tqdm(files):
            with codecs.open(f, "r", "utf-8") as book_file:
                corpus_raw += book_file.read()

        # set current corpus
        self.corpus = corpus_raw

    def make_sentences(self, delim=". "):
        # create sentences from corpus
        if self.corpus == None:
            print("Error: no corpus object found, use load_corpus function to generate corpus object")
            return
        raw_sentences = self.corpus.split(delim)
        sentences = []
        for raw_sentence in tqdm(raw_sentences):
            if len(raw_sentence) > 0:
                sentences.append(raw_sentence.split())
        self.sentences = sentences

        # update number of tokens in corpus
        self.token_count = sum([len(sentence) for sentence in sentences])

def main(args):
    # configure logger -
    out_dir = os.path.join(args.output, args.alias)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filename=os.path.join(out_dir, f"{args.alias}.log"), level=logging.INFO)

    corpus = Corpus(args.input)
    corpus.load_corpus()
    corpus.make_sentences()
    # Seed for the RNG, to make the results reproducible.
    seed = 1
    if args.workers == None:
        args.workers = multiprocessing.cpu_count()

    # build model
    gene2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=args.workers,
        size=args.size,
        min_count=args.minTF,
        window=args.window,
        sample=args.sample
    )
    gene2vec.build_vocab(corpus.sentences)
    print("Gene2Vec vocabulary length:", len(gene2vec.wv.vocab))
    gene2vec.train(corpus.sentences,
                     total_examples=gene2vec.corpus_count, epochs=args.epochs)
    # save model
    gene2vec.save(os.path.join(out_dir, f"{args.alias}_{datetime.today().strftime('%Y-%m-%d')}.w2v"))

    mapper = umap.UMAP(n_neighbors=15,min_dist=0.0, n_components=2)
    #train umap
    all_word_vectors_matrix_2d = mapper.fit_transform(gene2vec.wv.vectors.astype(
        'float64'))
    points = pd.DataFrame([(word, coords[0], coords[1])
        for word, coords in [(word, all_word_vectors_matrix_2d[gene2vec.wv.vocab[word].index])
            for word in gene2vec.wv.vocab]],
    columns=["word", "x", "y"])
    with open(os.path.join(out_dir, f"words_umap_{datetime.today().strftime('%Y-%m-%d')}"), 'wb') as o:
        pickle.dump(points, o)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--window', default=5, type=int, help='window size')
    argparse.add_argument('--size', default=300, type=int, help='vector size')
    argparse.add_argument('--workers', required=False, type=int, help='number of processes')
    argparse.add_argument('--epochs', default=5, type=int, help='number of epochs')
    argparse.add_argument('--minTF', default=4, type=int, help='minimum term frequency')
    argparse.add_argument('--sample', default=1e-3, type=int, help='down sampling setting for frequent words')
    argparse.add_argument('--model', required=False, type=str, help='model file if exists')
    argparse.add_argument('--input', default='../data/*', type=str, help='dir to learn from, as a regex for file generation')
    argparse.add_argument('--output', default='outputs/', type=str, help='output folder for results')
    argparse.add_argument('--alias', default='G2V', type=str, help='model running alias that will be used for model tracking')
    params = argparse.parse_args()

    main(params)
