from .olida_combination_filters import OLIDAEvidenceFilter, select_most_recent_non_redundant_combinations
from .sample_parser import parse_gene_pair_file
from ..utils.dict_utils import *
from ..utils.csv_utils import CSVFileWriter
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


def get_trainset_and_holdout(oligoKG, negative_pair_source, holdout_positive_size=10, pos_min_conf_level=1):
    positives_to_weight, holdout_positives_to_weight, gene_pairs_to_disease_ids = get_positive_pairs_from_kg(oligoKG, pos_min_conf_level, holdout_positive_size)
    negatives_to_weight = get_negative_file_ensg_pairs(oligoKG, negative_pair_source)

    logger.info(f"Training data sizes: {len(positives_to_weight)} positives | {len(negatives_to_weight)} negatives | {len(holdout_positives_to_weight)} holdout positives")
    sample_to_class = {}
    sample_to_weight = {}
    for sample, w in positives_to_weight.items():
        sample_to_class[sample] = 1
        sample_to_weight[sample] = w
    for sample, w in holdout_positives_to_weight.items():
        sample_to_class[sample] = 1
        sample_to_weight[sample] = w
    for sample, w in negatives_to_weight.items():
        sample_to_class[sample] = 0
        sample_to_weight[sample] = w
    return list(positives_to_weight.keys()), list(negatives_to_weight.keys()), list(holdout_positives_to_weight.keys()), sample_to_weight, sample_to_class


def get_positive_pairs_from_kg(kg, min_confidence_level=1, holdout_positive_size=15, holdout_min_confidence_level=2):
    gene_pairs_to_criteria = defaultdict(list)
    gene_pairs_to_disease_ids = defaultdict(set)
    for olida_id, properties in kg.index["olida"].items():
        if "gene_pair" in properties:
            node_pair = properties["gene_pair"]
            ensg_pair = tuple([kg.get_node_property(n, "id") for n in node_pair])
            weight = OLIDAEvidenceFilter.get_confidence_level(properties["evidences"])
            timestamp = properties["timestamp"]
            gene_pairs_to_criteria[ensg_pair].append((weight, timestamp))
            disease_nodes = properties.get("disease_nodes", [])
            disease_ids = [kg.get_node_property(d_n, "id") for d_n in disease_nodes]
            gene_pairs_to_disease_ids[ensg_pair].update(disease_ids)
    filtered_gene_pair_to_weight = {}
    filtered_gene_pair_to_timestamp = {}
    for gene_pair, criterias in gene_pairs_to_criteria.items():
        max_weight = max([w for w,t in criterias])
        min_timestamp = min([t for w,t in criterias])
        if max_weight >= min_confidence_level:
            filtered_gene_pair_to_weight[gene_pair] = max_weight
            if max_weight >= holdout_min_confidence_level:
                filtered_gene_pair_to_timestamp[gene_pair] = min_timestamp

    holdout_set = select_most_recent_non_redundant_combinations(filtered_gene_pair_to_timestamp, holdout_positive_size)

    training_pos_w = max_scaling_dict_vals({gp:w for gp,w in filtered_gene_pair_to_weight.items() if gp not in holdout_set}, 3)
    holdout_pos_w = max_scaling_dict_vals({gp:w for gp,w in filtered_gene_pair_to_weight.items() if gp in holdout_set}, 3)

    return training_pos_w, holdout_pos_w, gene_pairs_to_disease_ids


def get_negative_file_ensg_pairs(kg, negative_pair_source):
    ensg_pair_to_weight, unresolved_gene_pairs = parse_gene_pair_file(negative_pair_source, kg, input_gene_id_format="Ensembl")
    assert(len(unresolved_gene_pairs) == 0)
    return max_scaling_dict_vals(ensg_pair_to_weight)


def print_out_held_out_combinations(kg, output_file_prefix, format="HGNC", include_weight=False, include_header=False, min_confidence_level=1, holdout_positive_size=15, holdout_min_confidence_level=2):
    with CSVFileWriter(f"{output_file_prefix}.csv", delimiter=",") as out:
        if include_header:
            header = ["gene_1", "gene_2"]
            if include_weight:
                header += ["weight"]
            out.write_row(header)
        training_pos_w, holdout_pos_w, gene_pairs_to_disease_ids = get_positive_pairs_from_kg(kg, min_confidence_level, holdout_positive_size, holdout_min_confidence_level)
        for holdout_pos, w in holdout_pos_w.items():
            ensg_A, ensg_B = holdout_pos
            if format == "HGNC":
                A = kg.get_node_property(kg.index["id"][ensg_A], "name")
                B = kg.get_node_property(kg.index["id"][ensg_B], "name")
            else:
                A, B = holdout_pos
            row = [A, B]
            logger.debug(row)
            out.write_row(row)