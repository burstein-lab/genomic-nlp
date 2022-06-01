import glob
import shutil
import os
import codecs
import argparse
import pickle

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

def cp_files(list_of_files, dest):
    for f in list_of_files:
        if os.path.isfile(f):
            shutil.copy(f, dest)

def extract_known_words(list_of_files, unknown='hypo.clst'):
    corpus_raw = u""
    for f in tqdm(list_of_files):
        with codecs.open(f, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
    raw_sentences = corpus_raw.split('. ')
    words = []
    for raw_sentence in tqdm(raw_sentences):
        if len(raw_sentence) > 0:
            words.extend([w for w in raw_sentence.split() if unknown not in w])
    return set(words)

class CorpusCV():
    def __init__(self, corpus_dir, output_dir, folds, name, folds_mapping=None):
        self.corpus_dir = corpus_dir
        self.output_dir = output_dir
        self.nfolds = folds
        self.name = name
        self.folds_mapping = folds_mapping

    def create_output_dir(self):
        os.makedirs(os.path.join(self.output_dir, self.name),exist_ok=True)
        self.output_dir = os.path.join(self.output_dir, self.name)

    def corpus_kfold(self):
        corpus_files = np.array(glob.glob(self.corpus_dir))
        cv = KFold(n_splits=self.nfolds)
        fold = 1
        for train_idx, test_idx in cv.split(corpus_files):
            train_files = corpus_files[train_idx]
            test_files = corpus_files[train_idx]

            test_words = extract_known_words(test_files)

            # create the fold directory
            fold_dir = os.path.join(self.output_dir, f'fold_{fold}')
            train_dir = os.path.join(fold_dir, 'corpus')
            os.makedirs(fold_dir, exist_ok=True)
            os.makedirs(train_dir, exist_ok=True)

            cp_files(train_files, train_dir)
            np.save(os.path.join(fold_dir, 'test_words.npy'), test_words)

            fold += 1

    def corpusLOPOCV(self):
        with open(self.folds_mapping, 'rb') as handle:
            lopocv_mapper = pickle.load(handle)

        for phylum in lopocv_mapper:
            train_files = lopocv_mapper[phylum]['train_files']
            test_files = lopocv_mapper[phylum]['test_files']

            test_words = extract_known_words(test_files)

            # create the fold directory
            fold_dir = os.path.join(self.output_dir, f'fold_{phylum}')
            train_dir = os.path.join(fold_dir, 'corpus')
            os.makedirs(fold_dir, exist_ok=True)
            os.makedirs(train_dir, exist_ok=True)

            cp_files(train_files, train_dir)
            np.save(os.path.join(fold_dir, 'test_words.npy'), test_words)

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--input_dir', required=True, type=str, help='input directory with the full corpus files')
    argparse.add_argument('--output_dir', required=True, type=str, help='output directory for cross validation')
    argparse.add_argument('--lopocv', default=None, type=str, help='folds mapping of corpus files')
    argparse.add_argument('--name', default='5foldcv', type=str, help='cv identifier')
    argparse.add_argument('--folds', default=5, type=int, help='number of folds in cv')

    params = argparse.parse_args()

    corpus_cv = CorpusCV(params.input_dir, params.output_dir, params.folds, params.name, params.lopocv)
    corpus_cv.create_output_dir()
    if corpus_cv.folds_mapping is None:
        corpus_cv.corpus_kfold()
    else:
        corpus_cv.corpusLOPOCV()