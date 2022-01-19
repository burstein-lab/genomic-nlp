from collections import Counter
import statsmodels.stats.multitest as multi
from matplotlib.backends.backend_pdf import PdfPages
import os

import pandas as pd
import umap
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import stats, entropy
import gensim
from gensim.models import word2vec as w2v
import re
from tqdm import tqdm

# taken from https://umap-learn.readthedocs.io/en/latest/parameters.html
def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=data)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=data)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=data, s=100)
    plt.title(title, fontsize=18)

def reducer(matrix, mdl, n_neighbors=20, min_dist=0.1, n_components=2, metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(matrix)

    points = pd.DataFrame([tuple([word] + [coord for coord in coords])
                           for word, coords in [(word, u[mdl.wv.vocab[word].index])
                                                for word in mdl.wv.vocab]],
                          columns=["word"] + [str(i) for i in range(n_components)])
    return points

def add_metadata(metadata_path, mdl_folder):
    """
    add metadata for 2D words
    :param metadata_path: a path to a ko table with annotations per KO
    :param mdl_folder: a folder containing g2v model outputs
    :return: a merged dataframe
    """
    ko_table = pd.read_table(metadata_path)
    # read files from input folder
    cur_files = glob.glob(f"{mdl_folder}/*")
    with open([c for c in cur_files if "tsne" in c][0], 'rb') as o:
        words = pickle.load(o)
    words["hypothetical"] = words["word"].apply(lambda x: "YES" if "Cluster" in x else "NO")
    words["KO"] = words["word"]
    # merge files and save as pickle to reduce space
    merged = words.merge(ko_table, on=["KO"], how="left").fillna("unkown")
    return merged

def cluster_data(merged, cluster_obj):
    """
    cluster words data using a clustering algorithm
    :param merged: words dataframe having "x","y" columnsrr
    :param cluster_obj: abject for clustering, need to have a fit_predict function
    :return: 2d clustered dataframe
    """

    fp = getattr(cluster_obj, "fit_predict", None)
    if not callable(fp):
        raise Exception("cluster object provided do not have a fit_predict function")

    cluster_labels = cluster_obj.fit_predict(merged[["x","y"]])
    merged["cluster"] = cluster_labels
    merged = merged.sort_values(by="cluster")
    merged["cluster"] = merged["cluster"].astype(str)
    return merged


def get_kegg_enrichments(merged):
    """
    calculate all naive enrichments of kegg ko_lvl_3 annotations using a fisher exact test
    :param merged: a data frame of words, must be merged with KO data
    :return: a pair of dataframes- (merged df with enrichments, only enrichments)
    """
    res = {}
    dfs = []
    for cluster in merged["cluster"].unique():
        clust_df = merged[merged["cluster"] == cluster]
        split_by = "KO_lvl_3"
        splitted = clust_df[split_by].apply(lambda x: x.split(';')).explode()
        annotations = [s.strip() for s in splitted]
        res[cluster] = Counter(annotations)
        d = pd.DataFrame.from_dict(res[cluster], orient='index').reset_index().rename(
            columns={'index': 'annotation', 0: 'count'})
        d["cluster"] = cluster
        dfs.append(d)
    df = pd.concat(dfs)

    annot_enrich = []
    for cluster in merged["cluster"].unique():
        clust_df = df[df["cluster"] == cluster]
        not_clust_df = df[df["cluster"] != cluster]

        annotations = clust_df[clust_df["count"] > 1]["annotation"].unique()
        for annot in annotations:
            cluster_annot = clust_df[clust_df["annotation"] == annot]["count"].sum()
            not_cluster_annot = not_clust_df[not_clust_df["annotation"] == annot]["count"].sum()
            cluster_not_annot = clust_df[clust_df["annotation"] != annot]["count"].sum()
            not_cluster_not_annot = not_clust_df[not_clust_df["annotation"] != annot]["count"].sum()

            oddsratio, pvalue = stats.fisher_exact(
                [[cluster_annot, cluster_not_annot], [not_cluster_annot, not_cluster_not_annot]])
            annot_enrich.append((oddsratio, pvalue, cluster, annot))
    data = pd.DataFrame(annot_enrich, columns=["odds", "pvalue", "cluster", "annotation"])
    data['corrected_pvalue'] = multi.fdrcorrection(data['pvalue'])[1]
    data["enriched"] = data["corrected_pvalue"].apply(lambda x: "yes" if x < 0.05 else "no")

    enriched_df = data[(data["enriched"] == "yes") & (data["annotation"] != "unkown")]
    enriched_df = enriched_df.sort_values(by=["cluster", "corrected_pvalue"])
    res_df = merged.merge(data, on="cluster", how="left")

    return res_df, enriched_df


def cluster_entropy(merged, mode="collapsed"):
    """
    get the entropy of each cluster
    :param merged: a dataframe with words, kos and clusters
    :return: a dict - cluster: (score, cluster size, unknown size)
    """
    res = {}
    for cluster in merged["cluster"].unique():
        clust_df = merged[merged["cluster"] == cluster]
        split_by = "KO_lvl_3"
        if mode != "collapsed":
            splitted = clust_df[split_by].apply(lambda x: x.split(';')).explode()
            annotations = [s.strip() for s in splitted]
        else:
            annotations = clust_df[split_by]
        res[cluster] = Counter(annotations)

    cluster_scores = {}
    for cluster in res:
        vals = [v for key, v in res[cluster].items() if key!= "unkown"]
        n = sum(vals)
        score = entropy(vals, base=n)
        n_unknown = res[cluster]["unkown"]
        size = sum(res[cluster].values())
        cluster_scores[cluster] = (score, size, n_unknown)
    return cluster_scores


def process_word_statistics(mdl, outdir, hypo_word = 'hypo.clst'):
    with PdfPages(os.path.join(outdir, f'HypoDistribution.pdf')) as pdf:
        w2c_known = dict()
        w2c_unknown = dict()
        for item in mdl.wv.vocab:
            if hypo_word in item:
                w2c_unknown[item] = mdl.wv.vocab[item].count
            else:
                w2c_known[item] = mdl.wv.vocab[item].count

        fig, ax = plt.subplots(1, 2, figsize=(14, 4))

        ax[0].hist(w2c_known.values(), color='#7FB7E5', bins=20)
        ax[0].set_title(
            f"Known\nAVG: {round(np.mean(list(w2c_known.values())), 2)}, MED: {round(np.median(list(w2c_known.values())), 2)} MAX: {max(w2c_known.values())}")
        ax[0].set_yscale("log")
        ax[0].grid(True)

        ax[1].hist(w2c_unknown.values(), color='#DC3D13', alpha=0.8)
        ax[1].set_title(
            f"Unknown\nAVG: {round(np.mean(list(w2c_unknown.values())), 2)}, MED: {round(np.median(list(w2c_unknown.values())), 2)} MAX: {max(w2c_unknown.values())}")
        ax[1].set_yscale("log")
        ax[1].grid(True)

        pdf.savefig(transparent=True, bbox_inches="tight")
        plt.close()

def summeraize_mdls(w2v_mdl, word2metadata="words2metadata.pkl"):

    with open(word2metadata, 'rb') as o:
        words = pickle.load(o)

    res = []
    for mdl in tqdm(w2v_mdl):
        g2v = w2v.Word2Vec.load(mdl)
        corpus_type = "annotation" if "extended" not in mdl else "annotation_extended"
        batch_type = mdl.split("/")[6]
        mintf = int(re.findall(r"tf(\d*)_annotation", mdl)[-1])
        vocab_size = sum([v.count for k,v in g2v.wv.vocab.items()])
        unique_tokens = len(g2v.wv.vocab)
        unique_hypo = len([k for k in g2v.wv.vocab if 'hypo.clst.' in k])
        unique_kegg = unique_tokens - unique_hypo
        hypo_count = sum([v.count for k,v in g2v.wv.vocab.items() if 'hypo.clst.' in k])
        diamond_hypo = sum([words[k][0] for k in g2v.wv.vocab if 'hypo.clst.' in k and k in words])
        diamond_known_hypo = len([words[k][0] for k in g2v.wv.vocab if 'hypo.clst.' in k and k in words]) - diamond_hypo
        diamond_nf = unique_hypo - diamond_hypo - diamond_known_hypo

        res.append((corpus_type, batch_type, mintf, vocab_size, unique_tokens, unique_kegg, unique_hypo, hypo_count, diamond_hypo, diamond_known_hypo, diamond_nf))
    

    df = pd.DataFrame(res, columns=['corpus_type', 'batch_type', 'mintf', 'vocab_size', 'unique_tokens', 'unique_kegg', 'unique_hypo', 'hypo_count', 'diamond_hypo','diamond_known_hypo', 'diamond_not_found'])
    df["kegg_count"] = df["vocab_size"] - df["hypo_count"]
    df["per_kegg"] = df["kegg_count"] / df["vocab_size"]
    df["per_hypo"] = df["hypo_count"] / df["vocab_size"]
    df["per_unique_tokens"] = df["unique_tokens"] / df["vocab_size"]
    df["per_unique_kegg"] = df["unique_kegg"] / df["unique_tokens"]
    df["per_unique_hypo"] = df["unique_hypo"] / df["unique_tokens"]

    df["per_diamond_hypo"] = df["diamond_hypo"] / df["unique_hypo"]
    df["per_diamond_known_hypo"] = df["diamond_known_hypo"] / df["unique_hypo"]
    df["per_diamond_not_found"] = df["diamond_not_found"] / df["unique_hypo"]
    df = df.sort_values(by=["corpus_type", "mintf"])
    return df
