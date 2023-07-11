from ..utils.dict_utils import default_to_regular
from ..utils.cache_utils import Cache
from .rule_utils import directs_metapath
from collections import defaultdict
from tqdm import tqdm
import time
import abc
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


class RuleMatcher(metaclass=abc.ABCMeta):
    '''
    Abstract class for rule matching.
    '''

    @abc.abstractmethod
    def evaluate_rule(self, rule, sample_data):
        pass


class MetapathRuleMatcher(RuleMatcher):
    '''
    A rule matcher that evaluate rules based on metapaths, thresholds and unification conditions.
    '''

    def __init__(self, algo_params=None):
        super().__init__()
        self.algo_params = algo_params or {}
        self.orient_gene_pairs = self.algo_params.get("orient_gene_pairs", True)

    def run(self, rules, metapath_dict, sample_name, update_cache=False, pproc=None, partitions_count=100):
        '''
        Evaluate a list of rules over all samples in metapath_dict (with caching).
        :param rules: A list of rules to evaluate.
        :param metapath_dict: A dictionary of samples and their metapaths.
        :param sample_name: The name of the sample for caching.
        :param update_cache: Whether to update the cache.
        :param pproc: A parallel processing object.
        :param partitions_count: The number of partitions to use for parallel processing.
        :return: A dictionary mapping rules to their positive matches.
        '''
        logger.info("Running rule querying ...")
        output_name = Cache.generate_cache_file_name("rule_querying", sample_name, self.algo_params, 'path_cutoff', 'excl_node_types', 'minsup_ratio', 'max_rule_length', 'orient_gene_pairs', 'compute_unifications', 'optimize_metapath_thresholds')
        storage = Cache(output_name, update_cache, single_file=True)
        return storage.get_or_store("", lambda x: self._run(rules, metapath_dict, pproc, partitions_count))

    def _run(self, rules, metapath_dict, pproc=None, partitions_count=None):
        start = time.process_time()
        rule_to_index = {r:i for i,r in enumerate(rules)}

        if pproc:
            match_rules = pproc.map_collect(self.evaluate_match, metapath_dict.items(), partitions_count,
                                            {"rule_to_index": rule_to_index})
        else:
            match_rules = [self.evaluate_match(pair_and_metapaths, rule_to_index) for pair_and_metapaths in tqdm(metapath_dict.items())]

        query_result = {pair: matching_rule_indices for pair, matching_rule_indices in match_rules}
        elapsed_time = time.process_time() - start
        return query_result, elapsed_time

    def evaluate_rule(self, rule, sample_metapath_data):
        '''
        Evaluate a rule against a sample.
        :param rule: A rule to evaluate.
        :param sample_metapath_data: A dictionary of metapaths for a sample.
        :return: True if the rule is satisfied, False otherwise.
        '''
        directions = [1] if self.orient_gene_pairs else [1, -1]
        for direction in directions:
            if MetapathRuleMatcher._evaluate_rule_directed(rule, sample_metapath_data, direction):
                return True
        return False

    def evaluate_match(self, pair_and_metapath_dict, rule_to_index):
        '''
        Evaluate a sample against a list of rules.
        :param pair_and_metapath_dict: A tuple of (entity pair, metapath dictionary).
        :param rule_to_index: A dictionary mapping rules to their indices.
        :return: A tuple of (entity pair, matching rule indices).
        '''
        pair, metapath_dict = pair_and_metapath_dict
        matching_rules_idx = set()
        for rule, rule_idx in rule_to_index.items():
            if self.evaluate_rule(rule, metapath_dict):
                matching_rules_idx.add(rule_idx)
        return pair, matching_rules_idx

    @staticmethod
    def evaluate_to_get_paths(rule, pair_metapath_dict, orient_entity_pairs):
        '''
        (Internal use) Query the rule against entity pair KG data and return its paths
        :param rule: a Rule or KGPattern object
        :param pair_metapath_dict: a dictionary of precalculated metapath with the associated path information for a single entity pair (pair)
        :param orient_entity_pairs: boolean: True if the pair should be matched following the RVIS orientation ; False if pair should be matched both ways
        :return: A dictionary in the form direction -> unification -> metapath -> path -> path_score
        Note that the metapath is expressed in the rule order and you can use the direction to reorder it
        The paths are given in the order found in the matching instance subgraph (no need to reorder it)
        '''
        directions = [1] if orient_entity_pairs else [1, -1]
        pattern = rule.antecedent if hasattr(rule, 'antecedent') else rule
        rule_metapaths, unifications, path_thresholds = pattern.metapaths, pattern.unification, pattern.path_thresholds
        orientation_to_matching_paths = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        if path_thresholds is None:
            path_thresholds = [0] * len(rule_metapaths)

        for direction in directions:

            matching_metapaths = defaultdict(dict)
            for rule_metapath, path_threshold in zip(rule_metapaths, path_thresholds):
                directed_metapath = directs_metapath(rule_metapath, direction)
                if directed_metapath in pair_metapath_dict:
                    for path, path_score in pair_metapath_dict[directed_metapath].items():
                        path = path[::direction]
                        if path_score >= path_threshold:
                            matching_metapaths[rule_metapath][path] = path_score
                else:
                    break
            if len(matching_metapaths) != len(rule_metapaths):
                continue
            if len(rule_metapaths) == 1:
                for path, score in matching_metapaths[rule_metapaths[0]].items():
                    orientation_to_matching_paths[direction][None][rule_metapaths[0]][path] = score
            else:
                if unifications:
                    MetapathRuleMatcher.evaluate_to_get_unification_paths(unifications, orientation_to_matching_paths,
                                                                          matching_metapaths, direction)

                else:
                    for mp, paths in matching_metapaths.items():
                        for path, path_score in paths.items():
                            orientation_to_matching_paths[direction][None][mp][path] = path_score

        return default_to_regular(orientation_to_matching_paths)

    @staticmethod
    def evaluate_to_get_unification_paths(unifications, orientation_to_matching_paths, matching_metapaths, direction):
        unification_path_dict = defaultdict(lambda: defaultdict(dict))
        path_count = 0
        for mp, unified_var_pos in unifications:
            for path, path_score in matching_metapaths[mp].items():
                path_count += 1
                unified_node = path[unified_var_pos - 1]
                unification_path_dict[unified_node][mp][path] = path_score
        matching_unified_mps = set()
        for unified_node, mps_to_paths in unification_path_dict.items():
            if len(mps_to_paths) == len(unifications):
                for mp, paths in mps_to_paths.items():
                    for path, path_score in paths.items():
                        orientation_to_matching_paths[direction][unified_node][mp][path] = path_score
                        matching_unified_mps.add(mp)
        if matching_unified_mps:
            for matching_metapath, path_idx_to_score in matching_metapaths.items():
                if matching_metapath not in matching_unified_mps:
                    for path, path_score in path_idx_to_score.items():
                        orientation_to_matching_paths[direction][None][matching_metapath][path] = path_score

    @staticmethod
    def retrieve_subgraph_and_paths(rule, entity_pair, pair_metapath_dict, oligoKG, orient_entity_pairs=True):
        '''
        :param rule: a Rule or KGPattern object
        :param entity_pair: a entity pair tuple in the form (ENSGxxx, ENSGxxx)
        :param pair_metapath_dict:  a dictionary of precalculated metapath with the associated path information for a single entity pair (pair)
        :param oligoKG: the knowledge graph object
        :param orient_entity_pairs: boolean: True if the pair should be matched following the RVIS orientation ; False if pair should be matched both ways
        :return: Subgraph nodes & edges, and the paths indexed per direction
        '''
        entity_node_pairs = [oligoKG.index["id"][ensg] for ensg in entity_pair]
        orientation_to_matching_paths = MetapathRuleMatcher.evaluate_to_get_paths(rule.antecedent, pair_metapath_dict, orient_entity_pairs)

        edges = set()
        nodes = set()
        nodes.update(entity_node_pairs)
        direction_to_paths = defaultdict(dict)
        for direction, unification_to_paths in orientation_to_matching_paths.items():
            directed_entity_node_pairs = entity_node_pairs[::direction]
            for unification, mp_to_paths in unification_to_paths.items():
                for mp, path_to_score in mp_to_paths.items():
                    edge_types, node_types, edge_directions = mp
                    for path, path_score in path_to_score.items():
                        path_edges = []
                        intermediate_nodes = path
                        path_nodes = [directed_entity_node_pairs[0]] + list(intermediate_nodes) + [directed_entity_node_pairs[1]]
                        nodes.update(intermediate_nodes)

                        for i in range(len(path_nodes)-1):
                            e_label = edge_types[i]
                            n1 = path_nodes[i]
                            n2 = path_nodes[i+1]

                            for edge in oligoKG.g.edge(n1, n2, all_edges=True):
                                if e_label == oligoKG.get_edge_label(edge):
                                    edges.add(edge)
                                    path_edges.append(edge)
                        direction_to_paths[direction][tuple(path_edges)] = path_score

        return nodes, edges, direction_to_paths

    @staticmethod
    def _evaluate_rule_directed(rule, pair_metapath_dict, direction):
        '''
        :param rule: a Rule or KGPattern object
        :param pair_metapath_dict:  a dictionary of precalculated metapath with the associated path information for a single entity pair
        :param direction: 1 or -1
        :return: a dictionary of metapaths to paths
        '''
        matching_metapaths = defaultdict(set)
        pattern = rule.antecedent if hasattr(rule, 'antecedent') else rule
        rule_metapaths, unifications, path_thresholds = pattern.metapaths, pattern.unification, pattern.path_thresholds
        if path_thresholds is None:
            path_thresholds = [0] * len(rule_metapaths)
        for rule_metapath, path_threshold in zip(rule_metapaths, path_thresholds):
            directed_metapath = directs_metapath(rule_metapath, direction)
            if directed_metapath in pair_metapath_dict:
                for path, path_score in pair_metapath_dict[directed_metapath].items():
                    path = path[::direction]
                    if path_score >= path_threshold:
                        matching_metapaths[rule_metapath].add(path)
            else:
                break
        if len(matching_metapaths) != len(rule_metapaths):
            return False
        if len(rule_metapaths) == 1:
            return True
        else:
            if unifications:
                unification_check_dict = defaultdict(set)
                path_count = 0
                for mp, unified_var_pos in unifications:
                    for path in matching_metapaths[mp]:
                        path_count += 1
                        unified_node = path[unified_var_pos - 1]
                        unification_check_dict[unified_node].add(mp)
                has_valid_unification = False
                for unified_node, mps in unification_check_dict.items():
                    if len(mps) == len(unifications):
                        has_valid_unification = True
                        break
                return has_valid_unification
            else:
                return True






