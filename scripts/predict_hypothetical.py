import pandas as pd
import re
import numpy as np
import os
import glob

from src.genomic_embeddings.models import NNClf
from src.genomic_embeddings.data import Embedding
import argparse
import pickle

argparse = argparse.ArgumentParser()
argparse.add_argument('--model', required=True, type=str, help='model file')
argparse.add_argument('--output',
                          default='predictions',
                          type=str, help='predictions output dir')
argparse.add_argument('--metadata',
                          default='metadata.csv',
                          type=str, help='metadata csv file path')
params = argparse.parse_args()

MODEL = params.model
METADATA = params.metadata
OUTPUT_DIR = params.output

label='label'


with open(glob.glob(os.path.join(os.path.dirname(MODEL), "predictions/label/FOLD_NO-PUMPS-CURATED-LABELS/*.pkl"))[0], 'rb') as o:
    class_2_aupr = pickle.load(o)
class_2_aupr = {k:v for k, v in class_2_aupr.items() if k != 'ALL'}
curated_labels = [k for k in class_2_aupr if k != 'ALL' and k != 'Other']

emb = Embedding(mdl=MODEL, metadata=METADATA, labels=curated_labels)
emb.process_data_pipeline(label=label, q=0, add_other=True)
data = emb.data
X, y = data.drop(columns=[label]).values, data[label].values

meta = emb.metadata
meta['label'] = meta['label'].apply(lambda x: re.split('(.)\[|\(|,', x)[0].strip())
test_embeddings_idx = {word:emb.mdl.wv.vocab[word].index for word in emb.mdl.wv.vocab
                       if word not in emb.train_words["word"] and "hypo.clst" in word}
X_test = emb.embedding[[*test_embeddings_idx.values()]]

trainer = NNClf(X=X, y=y, out_dir=None)
predicted, predicted_prob = trainer.model_fit(X, X_test, y)

test_df = pd.DataFrame.from_dict(test_embeddings_idx, orient="index").reset_index().rename(
    columns={0:"index", 'index':"word"})
dic_y_mapping = {n: label for n, label in enumerate(np.unique(y))}

for key, value in dic_y_mapping.items():
    test_df[value] = predicted_prob[:,key]

test_df['predicted_class'] = predicted
test_df['predicted_class_score'] = predicted_prob.max(axis=1)

test_df["weighted_total_score"] = test_df.apply(lambda row: sum([row[k]*class_2_aupr[k] for k in class_2_aupr]), axis=1)
test_df["weighted_prediction_score"] = test_df.apply(lambda row:
                                                     class_2_aupr[row["predicted_class"]] *
                                                     row["predicted_class_score"], axis=1)
test_df = test_df.sort_values(by="weighted_total_score", ascending=False)

test_df.to_pickle(os.path.join(OUTPUT_DIR,  "hypothetical_predictions.pkl"))

