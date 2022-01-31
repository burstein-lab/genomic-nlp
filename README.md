# genomic-nlp

This repository contains the code used for compiling and analyzing the "biological corpus" presented in the paper:

**Deciphering microbial gene function using natural language processing**



## Getting the data
Start by downloading the data files from Zenodo database.

1. Click on the Zenodo link at the top of the repository or use [this link](https://www.Zenodo.org) to download the data zip file
2. Alternatively, use the command line as follows: 
```
mkdir data
cd data

wget https://zenodo.org/record/XXX/files/models_and_data.tar.gz?download=1
tar -zxvf models_and_data.tar.gz
mv data/* ./
rm models_and_data.tar.gz
```

## Setting up the working environment
First, set up python environment and dependencies. 
#### using pip
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
#### using conda
```
conda env create -f environment.yml
conda activate g2v-env
```

### Trained gene annotation embedding
The trained word2vec model on the entire genomic corpus are available in `models_and_data` as a gensim model.
To farther use them for downstream analysis set up your working environment and load the model.

In python run:
```
from gensim.models import word2vec as w2v

MODEL = model_and_data/gene2vec_w5_v300_tf24_annotation_extended/gene2vec_w5_v300_tf24_annotation_extended_2021-10-03.w2v
mdl = w2v.Word2Vec.load(MODEL)
```

from here you may use [gensim api](https://radimrehurek.com/gensim/models/word2vec.html) to extract words embeddings, calculate distances between words and more 
For example:
```
mdl.wv.vocab["K09140.2"]
```
will obtain the embedding of the word `K09140.2`, a sub-cluster of the KO identifier `K09140` in KEGG.

### Two-dimensional embedding space
Gene embeddings after dimension reduction using UMAP are available as a pickl file.

In python run:
```
import pickle

with open("model_and_data/gene2vec_w5_v300_tf24_annotation_extended/words_umap_2021-10-03", "rb") as hanle:
    embbedings_2d = pickle.load(handle)
```

###  Re-training word embeddings using the corpus
Re-training word embeddings with different parameters can be executed using the following script:
1. First, go to `models_and_data` folder and extract the corpus files
```
cd models_and_data 
tar -zxvf corpus.tar.gz
```
2. Train the model
```
python src/gene2vec.py --input 'corpus/*.txt'
```
To change specific parameters of the algorithm run
`python src/gene2vec.py --help` and configure accordingly. 

### Function classification model validation
Function classification validations are available in:
`models_and_data/gene2vec_w5_v300_tf24_annotation_extended/predictions`.
To re-run validations and generate AUC and AUPR graphs run the following script:
```
python scripts/classify.py --model PATH_TO_W2V_MDL --output PATH_TO_OUT_DIR --metadata metadata.csv
```
The csv file `metadata.csv` can be found in `models_and_data`.
Running this script will produce all data found under the folder:
`models_and_data/gene2vec_w5_v300_tf24_annotation_extended/predictions`

### Function classification of all hypothetical proteins
All prediction of hypothetical proteins in the corpus can be found here:
`models_and_data/gene2vec_w5_v300_tf24_annotation_extended/predictions/hypothetical_predictions.pkl`

To load the file as table, run in python:
```
import pandas as pd
table = pd.read_pickle("models_and_data/gene2vec_w5_v300_tf24_annotation_extended/predictions/hypothetical_predictions.pkl")
```
To regenerate the model predictions run:
```
cd models_and_data/gene2vec_w5_v300_tf24_annotation_extended/
python scripts/predict_hypothetical.py --model PATH_TO_W2V_MDL --output PATH_TO_OUT_DIR --metadata ../metadata.csv
```




