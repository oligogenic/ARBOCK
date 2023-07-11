from ..utils.csv_utils import *
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)

def parse_gene_pair_file(input_file, kg, input_gene_id_format="Ensembl", delimiter="\t", header_gene_cols=("gene_1", "gene_2"), header_weight_col="weight"):
    '''
    Parse a file containing a list of gene pairs with optional weight
    :param input_file: TSV file formatted as: gene_1 | gene_2 [| weight], where the genes are given according to gene_id_format.
    :param kg: the knowledge graph object
    :param input_gene_id_format: either Ensembl or HGNC (i.e official gene name)
    :return: a dictionary gene_pair -> weight ; where gene pair is a tuple of ensembl ids, oriented by RVIS. Weight is set to 1.0 if absent.
    '''
    gene_pair_to_weight = {}
    unresolved_gene_pairs = set()
    has_header = True if header_gene_cols else False
    with CSVFileReader(input_file, has_header=has_header, delimiter=delimiter) as f:
        for row in f:
            if not row:
                continue
            if len(row) < 2:
                raise Exception(f"The provided file should contain at least 2 columns separated by {delimiter}")
            if has_header:
                genes = [row[col] for col in header_gene_cols]
            else:
                genes = row[:2]
            resolved_gene_pair = format_gene_pair(genes[0], genes[1], kg, input_gene_id_format)
            if has_header and header_weight_col in row:
                weight = float(row[header_weight_col])
            elif len(row) > 2:
                weight = float(row[2])
            else:
                weight = 1.0
            if resolved_gene_pair is None:
                unresolved_gene_pairs.add(tuple(genes))
            else:
                gene_pair_to_weight[resolved_gene_pair] = weight
    return gene_pair_to_weight, unresolved_gene_pairs


def parse_gene_pair(gene_pair_str, kg, input_gene_format="Ensembl", separator=","):
    gene_pair = gene_pair_str.split(separator)
    return format_gene_pair(gene_pair[0], gene_pair[1], kg, input_gene_format)

def check_gene_pair(kg_index, gene_1, gene_2):
    if gene_1 in kg_index and gene_2 in kg_index:
        return kg_index[gene_1], kg_index[gene_2]
    else:
        return None

def format_gene_pair(gene_1, gene_2, kg, input_gene_format="Ensembl"):
    if input_gene_format == "Ensembl":
        node_pair = check_gene_pair(kg.index["id"], gene_1, gene_2)
    elif input_gene_format == "HGNC":
        node_pair = check_gene_pair(kg.index["geneName"], gene_1, gene_2)
    else:
        raise Exception("Unsupported gene id format")
    if node_pair is None:
        return None
    else:
        oriented_node_pair = kg.orient_pair(node_pair)
        gene_pair = tuple([kg.get_node_property(n, "id") for n in oriented_node_pair])
        return gene_pair
