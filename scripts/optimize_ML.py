import pickle
import xgboost as xgb
from sklearn.model_selection import  StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import argparse


def optimize(clf, X, y, parameters, alias):
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = ["precision", "recall"]
    print("========" +  alias + "========")
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        mdl = GridSearchCV(estimator=clf, param_grid=parameters, cv=cv, scoring="%s_weighted" % score)
        mdl.fit(X, y)

        print("Best parameters set found on development set:")
        print()
        print(mdl.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = mdl.cv_results_["mean_test_score"]
        stds = mdl.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, mdl.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


def main(args):

    with open(args.fold2data, 'rb') as o:
        fold2data = pickle.load(o)
    fold = args.fold_name
    lopocv = fold2data[args.cv_name]

    X_train, y_train = lopocv[fold]['X_train'], lopocv[fold]['y_train']
    clfs = [SVC(),
            RandomForestClassifier(n_jobs=10),
            xgb.XGBClassifier(n_jobs=10)]
    names = ['SVC', 'RF', 'XGB']
    params = [{'kernel':('linear', 'rbf'), 'C':[0.01, 1, 10]},
              {'max_depth': [6, 10, 50, 100, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [100, 500, 1000]},
              {'n_estimators': [100, 500, 1000],
               'max_depth': [6, 10, 50, 100, None],
               'learning_rate': [0.001, 0.05, None]},
              ]

    for clf, name, param_grid in zip(*[clfs, names, params]):
        optimize(clf, X_train, y_train, param_grid, name)




if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--fold_name', required=True, type=str, help='name of the fold use for train')
    argparse.add_argument('--cv_name', default='LOPOCV', type=str, help='name of the CV use for train')
    argparse.add_argument('--fold2data',
                          default='fold2data.pkl',
                          type=str, help='mapping of LOPOCV train and test')

    args = argparse.parse_args()

    main(args)