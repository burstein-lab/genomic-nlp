import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import glob
from Bio import SeqIO
import socket
import subprocess
import hashlib
from BCBio import GFF

class CorpusGenerator(object):
    def __init__(self, gff, by="annotation", include='include.txt',
                 word_mapping='word_mapping.pkl'):
        self.gff = gff
        self.name = gff.name
        self.include = include
        self.annotation = by
        self.text_df = None
        self.word_mapping = word_mapping

    def __repr__(self):
        return self.name

    def get_gff(self):
        return self.gff

    def validate(self):
        """ validate params before calling to make sentences"""
        gff = self.get_gff()
        gff.set_name()
        try:
            gff.set_gff_table()
        except:
            print("Cannot load KEGG table")
            return
        try:   
            gff.set_hypothetical()
        except:
            print("Cannot load hypothetical table")
            return
        return gff


    def make_sentences_df(self):
        gff = self.validate()
        if gff is None:
            print(f"Validation failed for GFF instance {self.gff.name}.")
            return
        hypo2rep = gff.hypothetical
        sample = gff.gff_table
        annotation = self.annotation

        with open(self.include, 'r') as o:
            includes = [l.replace('\n', '') for l in o.readlines()]

        sample["word"] = sample.apply(
            lambda row: hypo2rep[row["contig_id"]] if row["contig_id"] in hypo2rep else row[annotation],
            axis=1)
        sample['ctg'] = sample['contig_id'].apply(lambda x: x.rsplit('_', 1)[0])
        sample = sample[sample['ctg'].isin(includes)]
        sample['orf'] = sample['contig_id'].apply(lambda x: int(x.rsplit('_', 1)[-1]))

        data_to_text = sample.sort_values(["ctg", "orf"]).groupby(["ctg"])["word"].apply(
            list).reset_index()

        self.text_df = data_to_text

        return data_to_text

    def compile_text(self, path):
        """ create a text file contains the """
        text_df = self.text_df
        if text_df is None:
            print("no text df found, Exiting....")
            return

        text_df["text"] = text_df["word"].apply(lambda x: ' '.join(x))
        text = '. '.join(text_df["text"].values)
        with open(path, "w") as handle:
            handle.write(text)
        return text

    def generate(self, path):
        text_df = self.make_sentences_df()
        if text_df is None:
            return
        self.compile_text(path)








