import pandas as pd
import numpy as np
import os
import pickle

# ML packages
from sklearn import metrics
from sklearn import model_selection
import xgboost as xgb
from sklearn.model_selection import  StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# DL packages
import tensorflow as tf


##### Models interface ######

class Model(object):
    def __init__(self, X, y, out_dir, clf=None):
        self.X = X
        self.y = y
        self.clf = clf
        self.out_dir = out_dir
        self.name = "MDL"
        self.report = None
        self.confusion_matrix = None
        self.pr = None
        self.roc = None
        self.auc = None
        self.ap = None
        self.history=None


    def set_alias(self, alias='_TOPLABELS'):
        self.name = self.name + alias

    def model_fit(self, X_train, X_test, y_train):
        clf = self.clf
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        predicted_prob = clf.predict_proba(X_test)
        return predicted, predicted_prob


    def summarize_accuracy(self, y_test, predicted, predicted_prob, fold):
        report = metrics.classification_report(y_test, predicted, output_dict=True)
        classes = np.unique(y_test)
        y_test_array = pd.get_dummies(y_test, drop_first=False).values

        pr_dfs = []
        roc_dfs = []
        for i in range(len(classes)):
            precision, recall, thresholds = metrics.precision_recall_curve(
                y_test_array[:, i], predicted_prob[:, i])
            report[classes[i]]["aupr"] = metrics.auc(recall, precision)

            fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i],
                                                     predicted_prob[:, i])
            report[classes[i]]["auc"] = metrics.auc(fpr, tpr)
            cur_df_pr = pd.DataFrame({"precision": precision, "recall": recall, "fold": fold, "class": classes[i]})
            cur_df_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "fold": fold, "class": classes[i]})
            pr_dfs.append(cur_df_pr)
            roc_dfs.append(cur_df_roc)
        try:
            precision, recall, _ = metrics.precision_recall_curve(y_test_array.ravel(),
                                                                  predicted_prob.ravel())
            fpr, tpr, thresholds = metrics.roc_curve(y_test_array.ravel(), predicted_prob.ravel())
            micro_pr = pd.DataFrame({"precision": precision, "recall": recall, "fold": "micro", "class": "ALL"})
            micro_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "fold": "micro", "class": "ALL"})
            pr_dfs.append(micro_pr)
            roc_dfs.append(micro_roc)
            auc = round(metrics.roc_auc_score(y_test, predicted_prob,
                                              multi_class="ovr", average='weighted'), 2)
            ap = round(metrics.average_precision_score(y_test_array, predicted_prob,average="micro"), 2)
            self.auc = auc
            self.ap = ap
        except:
            print("Cannot calculate accuracy metrics")

        report_df = pd.DataFrame(report).T
        report_df["fold"] = fold

        pr = pd.concat(pr_dfs)
        roc = pd.concat(roc_dfs)

        return report_df, pr, roc


    def split_and_classify(self):
        X = self.X
        y = self.y
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42,
                                                                            stratify=y)
        predicted, predicted_prob = self.model_fit(X_train, X_test, y_train)
        report, precision_report, roc_report = self.summarize_accuracy(y_test, predicted, predicted_prob, fold='NO-CV')

        cm = metrics.confusion_matrix(y_test, predicted)
        classes = np.unique(y_test)
        cm = pd.DataFrame(cm, columns=classes, index=classes)

        self.pr = precision_report
        self.roc = roc_report
        self.report = report
        self.confusion_matrix = cm

        return report, precision_report, roc_report

    def wrap_up(self, label, alias):
        label_dir = os.path.join(self.out_dir, label)
        q_dir = os.path.join(label_dir, alias)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        if not os.path.exists(q_dir):
            os.makedirs(q_dir)

        report_path = os.path.join(q_dir, f"{self.name}_report.csv")
        cm_path = os.path.join(q_dir, f"{self.name}_confusion_matrix.csv")
        history_path = os.path.join(q_dir, f"{self.name}_history.pickle")

        self.out_dir = q_dir
        report = self.report.reset_index().rename(columns={"index": label})
        report.to_csv(report_path, index=False)
        cm = self.confusion_matrix
        history = self.history
        if cm is not None:
            cm.to_csv(cm_path)
        if history is not None:
            with open(history_path, 'wb') as handle:
                pickle.dump(history.history, handle)

    def classification_pipeline(self, label, alias='_TOPLABELS'):
        self.set_alias(alias)
        self.split_and_classify()
        self.wrap_up(label, alias)

class FoldsModel(Model):
    def __init__(self, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), **kwargs):
        super().__init__(**kwargs)
        self.name = self.name + 'Folds'
        self.cv = cv
        self.folds_pr = None
        self.folds_roc = None
        self.mean_pr = None

    def calc_ovelall_pr_by_folds(self, y_test_list, predicted_prob_list):
        res = []
        for fold in range(len(y_test_list)):
            classes = np.unique(y_test_list[fold])
            y_test_array = pd.get_dummies(y_test_list[fold], drop_first=False).values
            predicted_prob = predicted_prob_list[fold]

            for i in range(len(classes)):
                fold_by_class_pr = pd.DataFrame({'class':classes[i],'y_test':y_test_array[:, i],
                                                 'predicted_prob':predicted_prob[:, i]})
                res.append(fold_by_class_pr)
        df = pd.concat(res)

        all_classes = []
        for cl in df['class'].unique():
            cl_df = df[df['class'] == cl]
            precision, recall, thresh = metrics.precision_recall_curve(cl_df['y_test'], cl_df['predicted_prob'])
            res = pd.DataFrame({'class':cl, 'precision':precision, 'recall':recall, 'thresh': np.insert(thresh, 0,0)})
            res = res[~((res['precision'] == 0) & (res['recall'] == 0))].sort_values(by='thresh')
            all_classes.append(res)
        self.mean_pr = pd.concat(all_classes)

    def split_and_classify(self):
        reports, prs, rocs = [], [], []
        fold = 1
        X = self.X
        y = self.y

        y_real = []
        y_proba = []

        for train_index, test_index in self.cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            predicted, predicted_prob = self.model_fit(X_train, X_test, y_train)
            fold_report, fold_pr, fold_roc = self.summarize_accuracy(y_test, predicted, predicted_prob, fold)
            reports.append(fold_report)
            prs.append(fold_pr)
            rocs.append(fold_roc)
            y_real.append(y_test)
            y_proba.append(predicted_prob)
            fold += 1

        self.pr = pd.concat(prs)
        self.roc = pd.concat(rocs)
        self.report = pd.concat(reports)

        self.calc_ovelall_pr_by_folds(y_real, y_proba)

        return self.report, self.pr, self.roc

    def merge_folds(self):
        pr = self.pr
        roc = self.roc
        overall_by_cl = self.mean_pr

        mean_precision = np.linspace(0, 1, 100)
        mean_fpr = np.linspace(0, 1, 100)

        pr_grp = pr.groupby(["class", "fold"]).agg({'precision': list, 'recall': list}).reset_index()
        pr_grp["interp"] = pr_grp.apply(lambda row: np.interp(np.linspace(0,1,max(100, overall_by_cl[overall_by_cl['class'] == row['class']].shape[0])), row['precision'], row['recall']), axis=1)
        pr_grp["auc"] = pr_grp.apply(lambda row: metrics.auc(sorted(row["precision"]),
                                                             sorted(row["recall"], reverse=True)), axis=1)
        pr_data = pr_grp.groupby("class").agg({"interp": list, "auc": list}).reset_index()

        roc_grp = roc.groupby(["class", "fold"]).agg({'fpr': list, 'tpr': list}).reset_index()
        roc_grp["interp"] = roc_grp.apply(lambda row: np.interp(mean_fpr, row['fpr'], row['tpr']), axis=1)
        roc_grp["auc"] = roc_grp.apply(lambda row: metrics.auc(sorted(row["fpr"]),
                                                               sorted(row["tpr"], reverse=True)), axis=1)
        roc_data = roc_grp.groupby("class").agg({"interp": list, "auc": list}).reset_index()

        self.folds_roc = roc_data
        self.folds_pr = pr_data


    def merge_folds_reports(self):
        res = self.report
        res = res[res['fold'] != 'micro']
        avg = res.drop(columns=['fold']).groupby(res.index).mean()
        avg["fold"] = 'AVG'
        res = pd.concat([res, avg], axis=0)
        self.report = res
        return res

    def classification_pipeline(self, label, alias='_TOPLABELS'):
        self.set_alias(alias)
        self.split_and_classify()
        self.merge_folds_reports()
        self.merge_folds()
        self.wrap_up(label, alias)

class CVFoldsModel(FoldsModel):
    def __init__(self, fold2data, fold_type, **kwargs):
        super().__init__(**kwargs)
        self.name = self.name + fold_type + 'Folds'
        self.fold2data = fold2data

    def split_and_classify(self):
        reports, prs, rocs = [], [], []
        fold2data = self.fold2data

        y_real = []
        y_proba = []

        for fold in fold2data:
            X_train, X_test = fold2data[fold]['X_train'], fold2data[fold]['X_test']
            y_train, y_test = fold2data[fold]['y_train'], fold2data[fold]['y_test']

            predicted, predicted_prob = self.model_fit(X_train, X_test, y_train)
            fold_report, fold_pr, fold_roc = self.summarize_accuracy(y_test, predicted, predicted_prob, fold)
            reports.append(fold_report)
            prs.append(fold_pr)
            rocs.append(fold_roc)
            y_real.append(y_test)
            y_proba.append(predicted_prob)

        self.pr = pd.concat(prs)
        self.roc = pd.concat(rocs)
        self.report = pd.concat(reports)

        self.calc_ovelall_pr_by_folds(y_real, y_proba)

        return self.report, self.pr, self.roc



##### Models ######
class MLClf(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clf = xgb.XGBClassifier(n_estimators=100, max_depth=5)
        self.name = "XGB"

class NNClf(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DNN"


    def set_clf(self, n):
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                            tf.keras.layers.Dropout(0.2),
                                            tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                            tf.keras.layers.Dropout(0.2),
                                            tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                            tf.keras.layers.Dense(n, activation=tf.nn.softmax)])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.clf = model

    def model_fit(self, X_train, X_test, y_train):
        self.set_clf(pd.Series(y_train).nunique())
        clf = self.clf
        dic_y_mapping = {n: label for n, label in
                         enumerate(np.unique(y_train))}
        inverse_dic = {v: k for k, v in dic_y_mapping.items()}
        y_train_tag = np.array([inverse_dic[y] for y in y_train])

        history = clf.fit(x=X_train, y=y_train_tag, batch_size=256,
                          epochs=20, shuffle=True, verbose=0)
        self.history = history

        predicted_prob = self.clf.predict(X_test, workers=5)
        predicted = [dic_y_mapping[np.argmax(pred)] for pred in
                     predicted_prob]
        return predicted, predicted_prob


class MLClfFolds(FoldsModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clf = xgb.XGBClassifier(n_estimators=100, max_depth=5)
        self.name = "XGBFolds"


class NNClfFolds(FoldsModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DNNFolds"

    def set_clf(self, n):
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                            tf.keras.layers.Dropout(0.2),
                                            tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                            tf.keras.layers.Dropout(0.2),
                                            tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                            tf.keras.layers.Dense(n, activation=tf.nn.softmax)])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.clf = model

    def model_fit(self, X_train, X_test, y_train):
        self.set_clf(pd.Series(y_train).nunique())
        clf = self.clf
        dic_y_mapping = {n: label for n, label in
                         enumerate(np.unique(y_train))}
        inverse_dic = {v: k for k, v in dic_y_mapping.items()}
        y_train_tag = np.array([inverse_dic[y] for y in y_train])

        history = clf.fit(x=X_train, y=y_train_tag, batch_size=256,
                          epochs=20, shuffle=True, verbose=0)

        self.history = history

        predicted_prob = self.clf.predict(X_test)
        predicted = [dic_y_mapping[np.argmax(pred)] for pred in
                     predicted_prob]
        return predicted, predicted_prob

class NNClfCVFolds(CVFoldsModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "DNN" + self.name

    def set_clf(self, n):
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                            tf.keras.layers.Dropout(0.2),
                                            tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                            tf.keras.layers.Dropout(0.2),
                                            tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                            tf.keras.layers.Dense(n, activation=tf.nn.softmax)])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.clf = model

    def model_fit(self, X_train, X_test, y_train):
        self.set_clf(pd.Series(y_train).nunique())
        clf = self.clf
        dic_y_mapping = {n: label for n, label in
                         enumerate(np.unique(y_train))}
        inverse_dic = {v: k for k, v in dic_y_mapping.items()}
        y_train_tag = np.array([inverse_dic[y] for y in y_train])

        history = clf.fit(x=X_train, y=y_train_tag, batch_size=256,
                          epochs=20, shuffle=True, verbose=0)

        self.history = history

        predicted_prob = self.clf.predict(X_test)
        predicted = [dic_y_mapping[np.argmax(pred)] for pred in
                     predicted_prob]
        return predicted, predicted_prob

class XGBClfCVFolds(CVFoldsModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clf = xgb.XGBClassifier()
        self.name = "XGB" + self.name

        params = {"learning_rate":0.05, "max_depth":6, "n_estimators":800}
        self.clf.set_params(**params)

class RFClfCVFolds(CVFoldsModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clf = RandomForestClassifier(max_depth=50, min_samples_split=2, min_samples_leaf=1, n_estimators=1000)
        self.name = "RF" + self.name

class SVMClfCVFolds(CVFoldsModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clf = SVC(kernel="rbf", C=1, gamma='auto', probability=True)
        self.name = "SVM" + self.name