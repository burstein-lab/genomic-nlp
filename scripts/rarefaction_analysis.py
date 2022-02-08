import pandas as pd
import numpy as np
import os
import argparse

# word 2 vec
from gensim.models import word2vec as w2v

class EmpiricalRarefaction(object):
    def __init__(self, mdl, function):
        self.mdl = w2v.Word2Vec.load(mdl)
        self.preds =  pd.read_pickle(os.path.join(os.path.dirname(mdl),
                                                                    'predictions/hypothetical_predictions.pkl'))
        self.function = function
        self.t_weighted = 0.9
        self.t_unweighted = 0.99
        self.out_dir = os.path.join(os.path.dirname(mdl),'predictions')

    def set_preds_to_function(self):
        preds = self.preds
        preds = preds[(preds['predicted_class'] == self.function) &
                      ((preds['weighted_prediction_score'] > self.t_weighted) |
                       (preds['predicted_class_score'] > self.t_unweighted))]
        self.preds = preds

    def bootstrap_samples(self, n_bootstraps, n_genes_min, n_genes_max, step, alpha=0.05):
        preds = self.preds
        mdl = self.mdl
        preds['word_count'] = preds['word'].apply(lambda w: mdl.wv.vocab[w].count)
        preds = preds[['word', 'word_count']]
        preds['words_by_count'] = preds.apply(lambda row: [row["word"]] * row['word_count'], axis=1)

        gene_words = preds['words_by_count'].explode().values

        res = []
        for n_genes in np.arange(n_genes_min, n_genes_max, step):
            n_genes = min(n_genes, len(gene_words))
            X = np.random.choice(gene_words, size=[n_bootstraps, n_genes])
            X_sorted = np.sort(X, axis=1)
            uniq_genes_dist = (X_sorted[:,1:] != X_sorted[:,:-1]).sum(axis=1)+1
            uniq_genes_mean = np.mean(uniq_genes_dist)
            upper_std = uniq_genes_mean + np.std(uniq_genes_dist)
            lower_std = uniq_genes_mean - np.std(uniq_genes_dist)
            upper_q = 2*uniq_genes_mean - np.quantile(uniq_genes_dist, alpha/2)
            lower_q = 2*uniq_genes_mean - np.quantile(uniq_genes_dist, 1 - alpha / 2)
            res.append((uniq_genes_mean, upper_std, lower_std, upper_q, lower_q, self.function, n_genes))
        df = pd.DataFrame(res, columns=['uniq_genes_mean', 'upper_std', 'lower_std',
                                        'upper_q', 'lower_q', 'function', 'n_genes'])

        df.to_pickle(os.path.join(self.out_dir, f'{self.function}_bootstrap.pkl'))
        return df




if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--model', required=True, type=str, help='model file path')
    argparse.add_argument('--function', required=True, type=str, help='functional group')
    argparse.add_argument('--min_genes', default=100, type=int, help='min number of genes to sample')
    argparse.add_argument('--max_genes', default=500000, type=int, help='max number of genes to sample')
    argparse.add_argument('--bootstrap', default=10000, type=int, help='mumber of bootstraps')
    argparse.add_argument('--step', default=1000, type=int, help='step size')
    params = argparse.parse_args()
    
    rarefaction = EmpiricalRarefaction(mdl=params.model, function=params.function)
    rarefaction.set_preds_to_function()
    rarefaction.bootstrap_samples(n_bootstraps=params.bootstrap, n_genes_min=params.min_genes, n_genes_max=params.max_genes, step=params.step, alpha=0.05)
    
