import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import glob
from Bio import SeqIO
import pickle
from BCBio import GFF

class Gff(object):
    def __init__(self, gff_path=None, hypothetical_folder='/hypothetical_mapping/',
                 ko_path="metadata.csv"):
        self.gff = gff_path
        self.ko_path = ko_path
        self.hypothetical = None
        self.keggRun = False
        self.hypotheticalRun = False
        self.clusterRun = False
        self.name = None
        self.gff_table = None

        self.fasta = self.gff.replace('.kg.05_21.gff', '.fa')
        self.proteins_fasta = self.gff.replace('.kg.05_21.gff', '.proteins.faa')
        self.hypothetical_path = os.path.join(hypothetical_folder, f"{os.path.basename(gff_path).split('.contig')[0]}.pkl")

    def __repr__(self):
        return self.name


    def set_name(self):
        # get the name of the file as the name of the gff object (includes some previous folder hierarchy)
        protein_fasta_path = self.proteins_fasta
        self.name = os.path.basename(protein_fasta_path).split(".contig")[0]

    def set_hypothetical(self):
        hypothetical_path = self.hypothetical_path
        if not os.path.exists(hypothetical_path):
            raise FileNotFoundError
        with open(hypothetical_path, 'rb') as hanlde:
            hypothetical_mapping = pickle.load(hanlde)
        self.hypothetical = hypothetical_mapping


    def set_gff_table(self):
        gff_table_path = f"{self.gff}.parsed.tsv"
        if not os.path.exists(gff_table_path):
            raise FileNotFoundError
        gff_table = pd.read_table(gff_table_path)
        self.gff_table = gff_table


    def parse_gff(self):
        f = self.gff
        if not os.path.exists(f):
            raise Exception("No GFF is currently available, need to run_kegg_qprokka first")
        if os.path.exists(f.replace(".gff", ".gff.parsed.tsv")):
            return
        
        records = []
        in_handle = open(f)
        for rec in tqdm(GFF.parse(in_handle)):
            for feature in rec.features:
                strand = feature.strand
                start = int(feature.location.start)
                end = int(feature.location.end)
                if feature.id == '' and 'locus_tag' in feature.qualifiers:
                    feature.id = feature.qualifiers['locus_tag'][0]#tmp addition due to curr gff file formats
                if "inference" not in feature.qualifiers:
                    feature.qualifiers["inference"] = "no inference record"

                if "product" in feature.qualifiers:
                    records.append((feature.id, feature.qualifiers['product'][0], feature.qualifiers['inference'][-1], feature.type, strand, start, end))
                else:
                    records.append((feature.id, feature.type, feature.type, feature.type, strand, start, end))
        in_handle.close()
        df = pd.DataFrame(records, columns=["contig_id", "product", "inference", "type", "strand", "start", "end"])
        df["annotation"] = df.apply(lambda row: _annotate(row["product"], row["inference"], row["type"]), axis=1)
        df["annotation_extended"] = df.apply(lambda row: _annotate_extended(row["product"], row["inference"],
                                                                            row["type"]), axis=1)

        output_path = f.replace(".gff", ".gff.parsed.tsv")
        df.to_csv(output_path, index=False, sep='\t')

    def extract_hypothetical(self):
        """ extract all hypothetical proteins for sequence-based clustering """
        protein_fasta_path = self.proteins_fasta
        hypothetical_fasta_path = protein_fasta_path.replace(".faa", ".hypothetical.faa")
        gff_table_path = self.gff.replace(".gff", ".gff.parsed.tsv")

        if self.hypotheticalRun or os.path.exists(hypothetical_fasta_path):
            self.hypotheticalRun = True
            return

        if not os.path.exists(gff_table_path):
            raise Exception("no parsed gff table was found - please run parse_gff to obtain the table")

        annotations = pd.read_table(gff_table_path)
        #adjust to lower case:
        annotations["annotation"] = annotations["annotation"].apply(lambda x: x.lower())
        hypothetical_ids = annotations[annotations["annotation"].isin(["hypothetical protein", "putative protein"])][
            "contig_id"].values
        hypothetical_records = []
        for rec in SeqIO.parse(protein_fasta_path, 'fasta'):
            if rec.name in hypothetical_ids:
                hypothetical_records.append(rec)
        with open(hypothetical_fasta_path, 'w') as handle:
            SeqIO.write(hypothetical_records, handle, "fasta")

        self.hypotheticalRun = True

    def extract_hypothetical_and_prokka(self):
        """ extract all hypothetical proteins for sequence-based clustering """
        protein_fasta_path = self.proteins_fasta
        hypothetical_fasta_path = protein_fasta_path.replace(".faa", ".hypothetical.prokka.faa")
        gff_table_path = self.gff.replace(".gff", ".gff.parsed.tsv")
        if os.path.exists(hypothetical_fasta_path):
            self.hypotheticalRun = True
            return

        if not os.path.exists(gff_table_path):
            raise Exception("no parsed gff table was found - please run parse_gff to obtain the table")

        annotations = pd.read_table(gff_table_path)
        ko_table = pd.read_table(self.ko_path)
        filter_ids = \
        annotations[(~annotations["annotation"].isin(ko_table['KO'])) & (annotations["type"] == 'CDS')][
            "contig_id"].values.tolist()

        hypothetical_and_prokka_records = []
        for rec in SeqIO.parse(protein_fasta_path, 'fasta'):
            if rec.name in filter_ids:
                hypothetical_and_prokka_records.append(rec)
        with open(hypothetical_fasta_path, 'w') as handle:
            SeqIO.write(hypothetical_and_prokka_records, handle, "fasta")
        self.hypotheticalRun = True
    # for in house use
    def cluster_hypothetical(self, queue="dudulight", threads=4):
        """ run mmseq cluster to generate a table of clustered """
        hypothetical_path = self.proteins_fasta.replace(".faa", ".hypothetical.faa")
        os.system(f"python /davidb/daniellemiller/bioutils/scripts/mmseq_cluster_runner.py --query {hypothetical_path} "
                  f"--queue {queue} --threads {threads}")
        self.hypotheticalRun = True

    def cluster_hypothetical_and_prokka(self, queue="dudulight", threads=4):
        """ run mmseq cluster to generate a table of clustered """
        hypothetical_path = self.proteins_fasta.replace(".faa", ".hypothetical.prokka.faa")
        os.system(f"python /davidb/daniellemiller/bioutils/scripts/mmseq_cluster_runner.py --query {hypothetical_path} "
                  f"--queue {queue} --threads {threads}")
        self.hypotheticalRun = True

    def assign_clusters(self):
        clustering_path = self.hypothetical_path
        if not os.path.exists(clustering_path):
            print("Clustering tsv do not exists, check whether cluster_hypothetical was previously run")
            return
        output_path = clustering_path.replace(".tsv", ".assigned.tsv")
        if os.path.exists(output_path):
            return pd.read_table(output_path)
        table = pd.read_table(clustering_path, names=["id1","id2"]) #id1 is the cluster representative, id2 is the matched orf
        table["cluster_id"] = pd.factorize(table["id1"])[0]
        table["cluster_id"] = table["cluster_id"].apply(
            lambda x: f"{os.path.basename(clustering_path).split('.contig')[0].replace('.', '_')}_Cluster_{x}")
        table.to_csv(output_path, index=False, sep='\t')
        return table


def _annotate(product, inference, gff_type):
    if "kg.05_21.ren4prok" in inference:
        return inference.split(":")[-1].split('.')[0]
    elif gff_type != "CDS":
        return gff_type
    else:
        return product

def _annotate_extended(product, inference, gff_type):
    if "kg.05_21.ren4prok" in inference:
        return inference.split(":")[-1]
    elif gff_type != "CDS":
        return gff_type
    else:
        return product











