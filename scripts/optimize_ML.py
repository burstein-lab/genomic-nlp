import pickle
import xgboost as xgb
from sklearn.model_selection import  StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os

import argparse


def optimize(clf, X, y, parameters, alias):
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = ["precision", "recall"]

    results = []

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        mdl = GridSearchCV(estimator=clf, param_grid=parameters, cv=cv, scoring="%s_weighted" % score)
        mdl.fit(X, y)

        res = pd.DataFrame(mdl.cv_results_)
        res['score_function'] = score
        results.append(res)

    return pd.cocnat(results)



def main(args):

    with open(args.fold2data, 'rb') as o:
        fold2data = pickle.load(o)

    name = args.clf_name
    fold = args.fold_name
    lopocv = fold2data[args.cv_name]

    X_train, y_train = lopocv[fold]['X_train'], lopocv[fold]['y_train']

    names_mappings = { 'SVC' : {'CLF':SVC(),
                                'params': {'kernel':('linear', 'rbf'), 'C':[0.01, 1, 10]}},
                       'RF': {'CLF': RandomForestClassifier(n_jobs=10),
                              'params': {'max_depth': [6, 10, 50, 100, None], 'max_features': ['auto', 'sqrt'],
                                         'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10],
                                         'n_estimators': [100, 500, 1000]}},
                       'XGB' : {'CLF':xgb.XGBClassifier(n_jobs=10),
                                'params': {'n_estimators': [100, 500, 800, 1000], 'max_depth': [6, 10, 50, 100, None],
                                           'learning_rate': [0.001, 0.05, None]}}
    }

    clf, param_grid  = names_mappings[name]['CLF'], names_mappings[name]['params']
    results = optimize(clf, X_train, y_train, param_grid, name)
    results['fold_name'] = fold
    results['model'] = name

    results.to_csv(os.path.join(args.outdir, f'opt_{name}_{fold}'), index=False)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--fold_name', required=True, type=str, help='name of the fold use for train')
    argparse.add_argument('--cv_name', default='LOPOCV', type=str, help='name of the CV use for train')
    argparse.add_argument('--clf_name', default='SVC', type=str, help='name of the CV use for train')
    argparse.add_argument('--outdir', default='./',
                          type=str, help='output dir to save files')
    argparse.add_argument('--fold2data',
                          default='fold2data.pkl',
                          type=str, help='mapping of LOPOCV train and test')

    args = argparse.parse_args()

    main(args)