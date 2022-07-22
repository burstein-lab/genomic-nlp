import logging
import os
import sys
from src.genomic_embeddings.models import NNClfCVFolds, RFClfCVFolds, XGBClfCVFolds, SVMClfCVFolds
from src.genomic_embeddings.plot import FoldModelPlots
import argparse
import pickle


argparse = argparse.ArgumentParser()
argparse.add_argument('--cv', default='LOPOCV', type=str, help='name of the cross validation')
argparse.add_argument('--output',
                      default='/predictions',
                      type=str, help='predictions output dir')
argparse.add_argument('--metadata',
                      default='metadata.csv',
                      type=str, help='metadata csv file path')
argparse.add_argument('--fold2data',
                      default='fold2data.pkl',
                      type=str, help='mapping of LOPOCV train and test')

params = argparse.parse_args()

CV = params.cv
METADATA = params.metadata
OUTPUT_DIR = params.output
FOLD2DATA = params.fold2data

# configure logger
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    filename=os.path.join(OUTPUT_DIR, f"Validation.log"), level=logging.INFO)

# top predicted LABEL
curated_labels_no_pumps = ['Amino sugar and nucleotide sugar metabolism',
                           'Benzoate degradation',
                           'Energy metabolism',
                           'Oxidative phosphorylation',
                           'Porphyrin and chlorophyll metabolism',
                           'Prokaryotic defense system',
                           'Ribosome',
                           'Secretion system',
                           'Two-component system']

labels = [(curated_labels_no_pumps, 'NO-PUMPS-CURATED-LABELS')]
LABEL = 'label'
q = ''

with open(FOLD2DATA, 'rb') as o:
    fold2data = pickle.load(o)

for label, label_alias in labels:
    alias = label_alias
    MDLS = [(NNClfCVFolds(X=1, y=1, cv=5, out_dir=OUTPUT_DIR, fold2data=fold2data[CV], fold_type=CV), 'CVFOLD_' + alias),
            (XGBClfCVFolds(X=1, y=1, cv=5, out_dir=OUTPUT_DIR, fold2data=fold2data[CV], fold_type=CV), 'CVFOLD_' + alias),
            (RFClfCVFolds(X=1, y=1, cv=5, out_dir=OUTPUT_DIR, fold2data=fold2data[CV], fold_type=CV), 'CVFOLD_' + alias),
            (SVMClfCVFolds(X=1, y=1, cv=5, out_dir=OUTPUT_DIR, fold2data=fold2data[CV], fold_type=CV), 'CVFOLD_' + alias)]


    for mdl, name in MDLS:
        mdl.classification_pipeline(LABEL, alias=name)
        plotter = FoldModelPlots(mdl=mdl)
        plotter.plot_single_aupr_with_ci()
        plotter.plot_single_roc_with_ci()
        plotter.plot_precision_recall()
        plotter.plot_roc()
        plotter.plot_precision_recall_by_fold()
        plotter.plot_roc_by_fold()