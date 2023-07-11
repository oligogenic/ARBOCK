from ..kg.bock import *
from ..rule_mining.rule_querying import MetapathRuleMatcher
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


def write_explanation_subgraph(rules, gene_ensg_pair, metapath_dict, kg, prediction_output_folder):
    '''
    Write the explanation subgraph for a given gene pair.
    :param rules: the rules to write
    :param gene_ensg_pair: the gene pair
    :param metapath_dict: the metapath dictionary
    :param kg: the knowledge graph
    :param prediction_output_folder: the output folder
    '''
    gene_pair = "-".join(kg.convert_to_gene_name(gene_ensg_pair))
    rules_and_probas = sorted(rules.items(), key=lambda x: x[1], reverse=True)
    print(f"Matching {len(rules)} rules")
    for rule, probability in rules_and_probas:
        print(f" p(+)={'%0.3f' % probability} >> R{rule.id}: {rule.antecedent}")
    logger.info(f"Writing explanation subgraphs for {gene_ensg_pair} in {prediction_output_folder}")
    last_proba = -1
    rank = 0
    for rule, proba in rules_and_probas:
        nodes, edges, direction_to_edge_paths = MetapathRuleMatcher().retrieve_subgraph_and_paths(rule, gene_ensg_pair,
                                                                            metapath_dict[gene_ensg_pair], kg)
        subgraph = Graph(kg.get_subgraph(nodes, edges), prune=True)
        path_count = len(direction_to_edge_paths[1])
        if proba != last_proba:
            rank += 1
            last_proba = proba
        subgraph.save(f"{prediction_output_folder}/{gene_pair}-{rank}-{'%0.2f' % proba}-{path_count}.graphml",
                      fmt="graphml")