# genomic-nlp

This repository contains the code used for compiling and analyzing the "biological corpus" presented in the paper:

**Deciphering microbial gene function using natural language processing**


## Getting the data
*Note:* The data will be deposited to the Zenodo database and assigned a permanent DOI.
meanwhile, do not use the download instructions below and go to [this link](http://tiny.cc/eb6ouz).

Start by downloading the data files from the Zenodo database.  

1. Click on the Zenodo link at the top of the repository or use [this link](http://tiny.cc/eb6ouz) to download the data zip file
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

The setup was tested on Python 3.7.
Versions of all required programs appear in `requirements.txt` (for pip) and `environment.yml` (for conda).

### code availability
The source code used to train the word2vec model, extract its embedding and functional classifier can be
downloaded using pip:

```
pip install genomic-embeddings
```

### Trained gene annotation embedding
The trained word2vec model on the entire genomic corpus are available in `models_and_data` as a gensim model.
To farther use them for downstream analysis set up your working environment and load the model.

In python:
```
from genomic-embeddings import Embeddings

model_path = model_and_data/gene2vec_w5_v300_tf24_annotation_extended/gene2vec_w5_v300_tf24_annotation_extended_2021-10-03.w2v
gene_embeddings = Embeddings.load_embeddings(model_path)
```

from here you may use [gensim api](https://radimrehurek.com/gensim/models/word2vec.html) to extract words embeddings, 
calculate distances between words and more 
For example:
```
gene_embeddings.wv["K09140.2"]
```
will obtain the embedding of the word `K09140.2`, a sub-cluster of the KO identifier `K09140` in KEGG.

### Two-dimensional embedding space
Gene embeddings after dimension reduction using UMAP are available as a pickle file.

In python:
```
from genomic-embeddings import Embeddings

embeddings_2d_rep_path = "model_and_data/gene2vec_w5_v300_tf24_annotation_extended/words_umap_2021-10-03"
embeddings_2d = Embeddings.get_2d_mapping(embeddings_2d_rep_path)
```

### Functional classifier
To re-train all function classifier\generate performance plots:

```
from genomic_embeddings.models import NNClf
from genomic_embeddings.data import Embedding
from genomic_embeddings.plot import ModelPlots

metadata_path = '/models_and_data/metadata.csv'
labels = ['Prokaryotic defense system','Ribosome','Secretion system'] # example labels

# load embedding
emb = Embedding(mdl=model_path, metadata=metadata_path, labels=labels)
emb.process_data_pipeline(label='label', q='', add_other=True)
X, y = emb.data.drop(columns=['label']).values, emb.data['label'].values

# classify
clf = NNClf(X=X, y=y, out_dir='./')
clf.classification_pipeline('labe', alias='DNN')

# plot 
plotter = ModelPlots(mdl=clf)
plotter.plot_precision_recall()
plotter.plot_roc()
```
### Function classification model validation
Function classification validations are available in:
`models_and_data/gene2vec_w5_v300_tf24_annotation_extended/predictions`.   
To re-run validations and generate AUC and AUPR graphs run the following script:
```
python scripts/classify.py --model PATH_TO_W2V_MDL --output PATH_TO_OUT_DIR --metadata PATH_TO_METADATA
```
The csv file `metadata.csv` can be found in `models_and_data`.  
Running this script will produce all data found under the folder:  
`models_and_data/gene2vec_w5_v300_tf24_annotation_extended/predictions`

### Function classification of all hypothetical proteins
All predictions of hypothetical proteins in the corpus can be found here:
`models_and_data/gene2vec_w5_v300_tf24_annotation_extended/predictions/hypothetical_predictions.pkl`

To load the file as table, run in python:
```
import 
preds_path = "models_and_data/gene2vec_w5_v300_tf24_annotation_extended/predictions/hypothetical_predictions.pkl"
preds = get_functional_prediction(preds_path)
```
or the alternative  
```
import pandas as pd
table = pd.read_pickle("models_and_data/gene2vec_w5_v300_tf24_annotation_extended/predictions/hypothetical_predictions.pkl")
```
To **regenerate** the model predictions run:
```
cd models_and_data/gene2vec_w5_v300_tf24_annotation_extended/
python scripts/predict_hypothetical.py --model PATH_TO_W2V_MDL --output PATH_TO_OUT_DIR --metadata ../metadata.csv
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


### Running times
Model loading, result generation and analysis script are anticipated to run from few seconds up to 4-5 min.\
re-training of language model, and dimensionality reduction can take up to 10h with 20 CPUs.


### Paper figures reproducibility
All paper figures (excluding illustrations) are available as a jupyter notebook.  
To run the notebook on your computer, go to `figures/` and type `jupyter notebook` in your command line.  
The notebook `paper_figures.ipynb` will be available on your local machine.  

*Note:* running the notebook requires the `models_and_data` folder, configure paths accordingly.





