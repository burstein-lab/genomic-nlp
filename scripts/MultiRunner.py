from src.genomic_embeddings import Gff, corpus
import os
import argparse


def main(args):
    gff_file = args.input
    hypothetical = args.hypothetical

    gff = Gff.Gff(gff_path=gff_file)
    gff.set_name()
    gff.parse_gff()
    gff.extract_hypothetical_and_prokka()
    
    if args.build_corpus:
        gff = Gff.Gff(gff_path=gff_file, hypothetical_folder=hypothetical)
        gff.set_name()
        gff_corpus = corpus.CorpusGenerator(gff=gff, by=args.annotation)
        gff_corpus.generate(os.path.join(args.output, f"{gff_corpus.gff.name}.txt"))



if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--input', required=True, type=str, help='gff input file')
    argparse.add_argument('--output', default='/output/',
                          type=str, help='the path to restore output txt files')
    argparse.add_argument('--hypothetical',
                          default='pkl_by_sample',
                          type=str, help='hypothetical pkl per sample folder')
    argparse.add_argument('--alias', default='G2V', type=str, help='model running alias that will be used for model tracking')
    argparse.add_argument('--cluster', dest='cluster', action='store_true', help="run on cluster flag, default")
    argparse.add_argument('--local', dest='cluster', action='store_false', help="run locally flag")
    argparse.add_argument('--annotation', default='annotation', type=str, help='annotation level, can be annotation or annotation_extended [default: annotation]')
    argparse.add_argument('--build_corpus', dest='build_corpus', action='store_true', help="build corpus from parsed GFFs flag")
    argparse.set_defaults(cluster=True, build_corpus=False)
    params = argparse.parse_args()

    main(params)


