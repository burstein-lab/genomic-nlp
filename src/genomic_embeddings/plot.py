import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle
import math
import os
from sklearn import metrics
import numpy as np
import pickle

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class ModelPlots(object):
    def __init__(self, mdl):
        self.mdl = mdl
        self.roc_data = mdl.roc
        self.precision_data = mdl.pr
        self.mdl_report = mdl.report
        self.out_dir = mdl.out_dir
        self.name = mdl.name

    def plot_roc(self):
        roc_df = self.roc_data
        res = self.mdl_report

        colors = ['pink', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'gold', 'olive','tomato', 'deeppink']

        with PdfPages(os.path.join(self.out_dir, f'{self.name}_ROC.pdf')) as pdf:
            fig, ax = plt.subplots(figsize=(5, 4))

            for c, cl in zip(cycle(colors), roc_df["class"].unique()):
                if cl == "ALL":
                    c = "k"
                    score = metrics.auc(roc_df[roc_df["class"] == cl]["fpr"], roc_df[roc_df["class"] == cl]["tpr"])
                else:
                    score = res.groupby(res.index)['auc'].mean().loc[cl]

                ax.plot(roc_df[roc_df["class"] == cl]["fpr"], roc_df[roc_df["class"] == cl]["tpr"], lw=3,
                        label="class {0} ({1:0.2f})".format(cl, score),
                        color=c)
            ax.plot([0, 1], [0, 1], color='grey', lw=3, linestyle='--', alpha=0.2)
            ax.set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], title=f"ROC",
                   xlabel="False Positive Rate", ylabel="True Positive rate")
            ax.grid(True)
            ax.text(.7, 0.05, f"AUC:{self.mdl.auc}", fontsize=12)
            plt.legend(bbox_to_anchor=(1.01, 1))
            pdf.savefig(transparent=True, bbox_inches="tight")
            plt.close()

    def plot_precision_recall(self):
        pr_df = self.precision_data
        res = self.mdl_report

        colors = ['pink', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'gold', 'olive','tomato', 'deeppink']
        with PdfPages(os.path.join(self.out_dir, f'{self.name}_AUPR.pdf')) as pdf:
            fig, ax = plt.subplots(figsize=(5, 4))
            for c, cl in zip(cycle(colors), pr_df["class"].unique()):

                if cl == "ALL":
                    c = "k"
                    score = metrics.auc(pr_df[pr_df["class"] == cl]["recall"], pr_df[pr_df["class"] == cl]["precision"])
                else:
                    score = res.groupby(res.index)['aupr'].mean().loc[cl]

                ax.plot(pr_df[pr_df["class"] == cl]["recall"], pr_df[pr_df["class"] == cl]["precision"], lw=3,
                        label="class {0} ({1:0.2f})".format(cl, score),
                        color=c)
            ax.set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], title=f"AUPR", xlabel="Recall", ylabel="Precision")
            ax.grid(True)
            ax.text(0.01, 0.05, "F1:{0:0.2f}".format(res.groupby(res.index)['f1-score'].mean().loc['weighted avg']),
                    fontsize=12)
            plt.legend(bbox_to_anchor=(1.01, 1))
            pdf.savefig(transparent=True, bbox_inches="tight")
            plt.close()


class FoldModelPlots(object):
    def __init__(self, mdl):
        self.mdl = mdl
        self.roc_data = mdl.roc
        self.precision_data = mdl.pr
        self.mdl_report = mdl.report
        self.outdir = mdl.out_dir
        self.name = mdl.name

    def plot_roc(self):
        roc_df = self.roc_data
        res = self.mdl_report

        roc_df = roc_df[roc_df['class'] != 'ALL']

        n_classes = roc_df["class"].nunique()
        classes = list(roc_df["class"].unique())
        COLS = 4
        ROWS = math.ceil(n_classes / COLS)
        pages = math.ceil(ROWS / 4)

        colors = ['pink', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'gold', 'olive','tomato', 'deeppink']

        with PdfPages(os.path.join(self.outdir, f'{self.name}_ROC.pdf')) as pdf:
            for page in range(pages):
                fig, ax = plt.subplots(COLS, COLS, figsize=(20,16))

                i, j = 0, 0
                for cl in classes[page*COLS*COLS: page*COLS*COLS +COLS*COLS]:
                    d = roc_df[roc_df["class"] == cl]

                    for c, fold in zip(colors, d["fold"].unique()):
                        ax[i][j].plot(d[d["fold"] == fold]["fpr"], d[d["fold"] == fold]["tpr"],
                                      lw=3, label=f"fold: {fold}", color=c)
                        ax[i][j].plot([0, 1], [0, 1], color='grey', lw=3, linestyle='--', alpha=0.2)
                        ax[i][j].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], title=f"{cl}")
                        ax[i][j].grid(True)
                        ax[i][j].text(.6, 0.05, "AUC:{0:0.2f}".format(res.groupby(res.index)['auc'].mean().loc[cl]),
                                      fontsize=12)
                        if j == 0:
                            ax[i][j].set(ylabel="True Positive Rate")
                        if i == COLS - 1:
                            ax[i][j].set(xlabel="False Positive Rate")
                    if j == (COLS-1):
                        j = 0
                        i += 1
                    else:
                        j += 1
                plt.subplots_adjust(hspace=.3, wspace=.3)
                _ = [ax[k][q].axis("off") for k in range(COLS) for q in range(COLS) if not ax[k][q].lines]
                pdf.savefig(transparent=True, bbox_inches="tight")
                plt.close()

    def plot_roc_by_fold(self):
        roc_df = self.roc_data
        res = self.mdl_report

        roc_df = roc_df[roc_df['class'] != 'ALL']
        roc_df['fold'] = roc_df['fold'].apply(lambda x: x.split('_')[-1])

        n_folds = roc_df["fold"].nunique()
        folds = list(roc_df["fold"].unique())
        COLS = 3
        ROWS = math.ceil(n_folds / COLS)
        pages = math.ceil(ROWS / 3)

        class2color = {'Amino sugar and nucleotide sugar metabolism':'DarkOrchid',
                       'Benzoate degradation':'darkorange', 'Energy metabolism':'cornflowerblue', 'Other':'grey',
                       'Oxidative phosphorylation':'gold',
                       'Porphyrin and chlorophyll metabolism':'teal',
                       'Prokaryotic defense system':'tomato', 'Ribosome':'deeppink', 'Secretion system':'pink',
                       'Two-component system':'turquoise', 'ALL':'k'}

        with PdfPages(os.path.join(self.outdir, f'{self.name}_ROC_BY_FOLD.pdf')) as pdf:
            for page in range(pages):
                fig, ax = plt.subplots(COLS, COLS, figsize=(20,16))

                i, j = 0, 0
                for fold in folds[page*COLS*COLS: page*COLS*COLS +COLS*COLS]:
                    d = roc_df[roc_df["fold"] == fold]

                    for cl in d["class"].unique():
                        ax[i][j].plot(d[d["class"] == cl]["fpr"], d[d["class"] == cl]["tpr"],
                                      lw=3, label=cl, color=class2color[cl])
                        ax[i][j].plot([0, 1], [0, 1], color='grey', lw=3, linestyle='--', alpha=0.2)
                        ax[i][j].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], title=f"{fold}")
                        ax[i][j].grid(True)
                        ax[i][j].text(.6, 0.05, "AUC:{0:0.2f}".format(res.groupby('fold')['auc'].mean().loc[f'fold_{fold}']),
                                      fontsize=12)
                        if j == 0:
                            ax[i][j].set(ylabel="True Positive Rate")
                        if i == COLS - 1:
                            ax[i][j].set(xlabel="False Positive Rate")
                    if j == (COLS-1):
                        j = 0
                        i += 1
                    else:
                        j += 1
                plt.subplots_adjust(hspace=.3, wspace=.3)
                _ = [ax[k][q].axis("off") for k in range(COLS) for q in range(COLS) if not ax[k][q].lines]
                pdf.savefig(transparent=True, bbox_inches="tight")
                plt.close()


    def plot_precision_recall(self):
        pr_df = self.precision_data
        res = self.mdl_report

        pr_df = pr_df[pr_df['class'] != 'ALL']

        n_classes = pr_df["class"].nunique()
        classes = list(pr_df["class"].unique())
        COLS = 4
        ROWS = math.ceil(n_classes / COLS)
        pages = math.ceil(ROWS / 4)

        colors = ['pink', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'gold', 'olive','tomato', 'deeppink']

        with PdfPages(os.path.join(self.outdir, f'{self.name}_AUPR.pdf')) as pdf:
            for page in range(pages):
                fig, ax = plt.subplots(COLS, COLS, figsize=(20,16))

                i, j = 0, 0
                for cl in classes[page*COLS*COLS : COLS*COLS*(page +1)]:

                    d = pr_df[pr_df["class"] == cl]

                    for c, fold in zip(colors, d["fold"].unique()):
                        ax[i][j].plot(d[d["fold"] == fold]["recall"], d[d["fold"] == fold]["precision"], lw=3,
                                      label=f"fold: {fold}", color=c
                                      )
                        ax[i][j].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], title=f"{cl}")
                        ax[i][j].grid(True)
                        ax[i][j].text(0.01, 0.05, "AUPR:{0:0.2f}".format(res.groupby(res.index)['aupr'].mean().loc[cl]),
                                      fontsize=12)
                        if j == 0:
                            ax[i][j].set(ylabel="Precision")
                        if i == COLS - 1:
                            ax[i][j].set(xlabel="Recall")

                    if j == (COLS-1):
                        j = 0
                        i += 1
                    else:
                        j += 1
                plt.subplots_adjust(hspace=.3, wspace=.3)
                _ = [ax[k][q].axis("off") for k in range(COLS) for q in range(COLS) if not ax[k][q].lines]
                pdf.savefig(transparent=True, bbox_inches="tight")
                plt.close()


    def plot_precision_recall_by_fold(self):
        pr_df = self.precision_data
        res = self.mdl_report

        pr_df = pr_df[pr_df['class'] != 'ALL']
        pr_df['fold'] = pr_df['fold'].apply(lambda x: x.split('_')[-1])

        n_folds = pr_df["fold"].nunique()
        folds = list(pr_df["fold"].unique())
        COLS = 3
        ROWS = math.ceil(n_folds / COLS)
        pages = math.ceil(ROWS / 3)

        class2color = {'Amino sugar and nucleotide sugar metabolism':'DarkOrchid',
                       'Benzoate degradation':'darkorange', 'Energy metabolism':'cornflowerblue', 'Other':'grey',
                       'Oxidative phosphorylation':'gold',
                       'Porphyrin and chlorophyll metabolism':'teal',
                       'Prokaryotic defense system':'tomato', 'Ribosome':'deeppink', 'Secretion system':'pink',
                       'Two-component system':'turquoise', 'ALL':'k'}


        with PdfPages(os.path.join(self.outdir, f'{self.name}_AUPR_BY_FOLD.pdf')) as pdf:
            for page in range(pages):
                fig, ax = plt.subplots(COLS, COLS, figsize=(20,16))

                i, j = 0, 0
                for fold in folds[page*COLS*COLS : COLS*COLS*(page +1)]:

                    d = pr_df[pr_df["fold"] == fold]
                    d = d[~((d['precision'] == 0) & (d['recall'] == 0))]

                    for cl in d["class"].unique():
                        ax[i][j].plot(d[d["class"] == cl]["recall"], d[d["class"] == cl]["precision"], lw=3,
                                      label=cl, color=class2color[cl]
                                      )
                        ax[i][j].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], title=f"{fold}")
                        ax[i][j].grid(True)
                        ax[i][j].text(0.01, 0.05, "AUPR:{0:0.2f}".format(res.groupby('fold')['aupr'].mean().loc[f'fold_{fold}']),
                                      fontsize=12)
                        if j == 0:
                            ax[i][j].set(ylabel="Precision")
                        if i == COLS - 1:
                            ax[i][j].set(xlabel="Recall")

                    if j == (COLS-1):
                        j = 0
                        i += 1
                    else:
                        j += 1
                plt.subplots_adjust(hspace=.3, wspace=.3)
                _ = [ax[k][q].axis("off") for k in range(COLS) for q in range(COLS) if not ax[k][q].lines]
                pdf.savefig(transparent=True, bbox_inches="tight")
                plt.close()


    def plot_single_aupr_with_ci(self):
        data = self.mdl.folds_pr
        overall_data = self.mdl.mean_pr
        overall_data = overall_data[~((overall_data['precision'] == 0) & (overall_data['recall'] == 0))]
        class2aupr = {}
        class2color = {'Amino sugar and nucleotide sugar metabolism':'DarkOrchid',
                       'Benzoate degradation':'darkorange', 'Energy metabolism':'cornflowerblue', 'Other':'grey',
                       'Oxidative phosphorylation':'gold',
                       'Porphyrin and chlorophyll metabolism':'teal',
                       'Prokaryotic defense system':'tomato', 'Ribosome':'deeppink', 'Secretion system':'pink',
                       'Two-component system':'turquoise', 'ALL':'k'}


        with PdfPages(os.path.join(self.outdir, f'{self.name}_AUPR_CI.pdf')) as pdf:
            fig, ax = plt.subplots(figsize=(5, 4))
            for cl in data["class"].unique():

                class_data = data[data["class"] == cl]
                tprs = class_data['interp'].tolist()[0]
                mean_precision = np.mean(tprs, axis=0)
                mean_recall = np.linspace(0, 1, mean_precision.shape[0])
                mean_auc = metrics.auc(mean_recall, mean_precision)

                if cl == 'ALL':
                    ax.plot(mean_recall, mean_precision, color=class2color[cl], label="{0} ({1:0.2f})".format(cl, mean_auc), lw=3, alpha=.8)
                else:
                    pr_data = overall_data[overall_data['class'] == cl]
                    mean_auc = metrics.auc(pr_data['recall'], pr_data['precision'])
                    ax.plot(pr_data['recall'], pr_data['precision'], color=class2color[cl],label="{0} ({1:0.2f})".format(cl, mean_auc), lw=3, alpha=.8)

                    mean_precision = np.mean(tprs)
                    std_precision = np.std(tprs, axis=0)
                    ax.fill_between(pr_data['recall'], pr_data['precision'] + std_precision, pr_data['precision'] - std_precision, color=class2color[cl], alpha=.1)

                class2aupr[cl] = mean_auc
            ax.set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], title=f"AUPR", xlabel="Recall", ylabel="Precision")
            ax.grid(True)
            plt.legend(bbox_to_anchor=(1.01, 1))
            pdf.savefig(transparent=True, bbox_inches="tight")
            plt.close()
        with open(os.path.join(self.outdir, f'{self.name}_AUPR_CI.pkl'), 'wb') as o:
            pickle.dump(class2aupr, o)

    def plot_single_roc_with_ci(self):
        data = self.mdl.folds_roc

        class2color = {'Amino sugar and nucleotide sugar metabolism':'DarkOrchid',
                       'Benzoate degradation':'darkorange', 'Energy metabolism':'cornflowerblue', 'Other':'grey',
                       'Oxidative phosphorylation':'gold',
                       'Porphyrin and chlorophyll metabolism':'teal',
                       'Prokaryotic defense system':'tomato', 'Ribosome':'deeppink', 'Secretion system':'pink',
                       'Two-component system':'turquoise', 'ALL':'k'}

        with PdfPages(os.path.join(self.outdir, f'{self.name}_ROC_CI.pdf')) as pdf:
            fig, ax = plt.subplots(figsize=(5, 4))
            for cl in data["class"].unique():

                class_data = data[data["class"] == cl]
                tprs = class_data['interp'].tolist()[0]
                mean_tpr = np.mean(tprs, axis=0)
                mean_fpr = np.linspace(0, 1, 100)
                mean_auc = metrics.auc(mean_fpr, mean_tpr)

                ax.plot(mean_fpr, mean_tpr, color=class2color[cl],
                        label="{0} ({1:0.2f})".format(cl, mean_auc), lw=3, alpha=.8)

                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=class2color[cl], alpha=.1)

            ax.set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], title=f"ROC", xlabel="True Positive Rate",
                   ylabel="False Positive Rate")
            ax.grid(True)
            plt.legend(bbox_to_anchor=(1.01, 1))
            pdf.savefig(transparent=True, bbox_inches="tight")
            plt.close()