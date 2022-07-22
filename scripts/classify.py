import logging
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.genomic_embeddings.models import MLClf, MLClfFolds, NNClf, NNClfFolds
from src.genomic_embeddings.data import Embedding
from src.genomic_embeddings.plot import ModelPlots, FoldModelPlots
import argparse


argparse = argparse.ArgumentParser()
argparse.add_argument('--model', required=True, type=str, help='model file')
argparse.add_argument('--output',
                          default='/predictions',
                          type=str, help='predictions output dir')
argparse.add_argument('--metadata',
                          default='metadata.csv',
                          type=str, help='metadata csv file path')
params = argparse.parse_args()

MODEL = params.model
METADATA = params.metadata
OUTPUT_DIR = params.output

# configure logger
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    filename=os.path.join(OUTPUT_DIR, f"Validation.log"), level=logging.INFO)

# top predicted LABEL
top_labels = ['Amino sugar and nucleotide sugar metabolism', 'Benzoate degradation', 'Cell growth', 'Energy metabolism',
              'Methane metabolism', 'Oxidative phosphorylation', 'Prokaryotic defense system', 'Ribosome',
              'Secretion system', 'Transporters', 'Glycosyltransferases']

curated_labels = ['Amino sugar and nucleotide sugar metabolism',
                  'Benzoate degradation',
                  'Energy metabolism',
                  'Oxidative phosphorylation',
                  'Porphyrin and chlorophyll metabolism',
                  'Prokaryotic defense system',
                  'Purine metabolism',
                  'Ribosome',
                  'Secretion system',
                  'Transporters',
                  'Two-component system']

curated_labels_no_pumps = ['Amino sugar and nucleotide sugar metabolism',
                  'Benzoate degradation',
                  'Energy metabolism',
                  'Oxidative phosphorylation',
                  'Porphyrin and chlorophyll metabolism',
                  'Prokaryotic defense system',
                  'Ribosome',
                  'Secretion system',
                  'Two-component system']

labels = [(top_labels, 'TOPLABELS'), (curated_labels, 'CURATED-LABELS'),
          (curated_labels_no_pumps, 'NO-PUMPS-CURATED-LABELS')]
LABEL = 'label'
q = ''

for label, label_alias in labels:
        alias = label_alias
        logging.info(f"=== Extract embedding for label = {LABEL}, Q= {q}")
        emb = Embedding(mdl=MODEL, metadata=METADATA, labels=label)
        emb.process_data_pipeline(label=LABEL, q=q, add_other=True)
        logging.info(f"Number of effective words: {emb.effective_words.shape[0]}\n")


        data = emb.data
        X, y = data.drop(columns=[LABEL]).values, data[LABEL].values
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        logging.info(f"Matrix shape is: {X.shape}, 20% used for testing in 5 fold CV\n"
                     f"Train size: {X.shape[0] * 0.8}, Test size: {X.shape[0] * 0.2}\n"
                     f"Number of unique classes: {pd.Series(y).nunique()}")

        MDLS = [(MLClf(X=X, y=y, out_dir=OUTPUT_DIR), alias),
                (NNClf(X=X, y=y, out_dir=OUTPUT_DIR),alias),
                (MLClfFolds(X=X, y=y, cv=cv, out_dir=OUTPUT_DIR),'FOLD_' + alias),
                (NNClfFolds(X=X, y=y, cv=cv, out_dir=OUTPUT_DIR), 'FOLD_' + alias)]


        for mdl, name in MDLS:
            mdl.classification_pipeline(LABEL, alias=name)
            if 'FOLD' in name:
                plotter = FoldModelPlots(mdl=mdl)
                plotter.plot_single_aupr_with_ci()
                plotter.plot_single_roc_with_ci()
            else:
                plotter = ModelPlots(mdl=mdl)
            plotter.plot_precision_recall()
            plotter.plot_roc()






