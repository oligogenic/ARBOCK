from .rule import KGPattern
from .rule_utils import directs_metapath, valid_support, get_absolute_minsup
from ..utils.cache_utils import Cache
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations, product
import time
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


class MetapathPatternUnificationMiner:
    '''
    Class to mine unifications of metapath patterns.
    Metapath unifications aim to represent cases where paths of different types (metapaths) share common intermediate nodes.
    This adds a refinement to the pattern, making it more specific and limiting the number of paths that match it (increased interpretability).
    Note that the minsup criterion is also applied to consider a refined pattern as frequent (and therefore a valid one).
    A unification pattern is encoded as a list of tuples of the size of the unified metapaths.
    Each tuple consists in (the unified metapath, the index (0-indexed) of the common node type).

    Example:
    Consider the following metapaths composing a rule:
    - MP1: (('i', 'i'), ('A', 'B', 'Z'), (1, -1))        # edge types, node types, edge directions
    - MP2: (('j', 'i', 'i'), ('A', 'C', 'B', 'Z'), (0, 1, -1))
    - MP3: (('j'), ('A', 'Z'), (0))

    Then, the unification pattern could be:
    - ((MP1, 1), (MP2, 2))   # -> meaning, unify MP1 and MP2 on B (node type at index 1 & node type at index 2)
    '''

    def __init__(self, algo_params):
        self.algo_params = algo_params

    def run(self, pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, sample_name, update_cache=False, pproc=None, partitions_count=None):
        '''
        Run the unification mining algorithm.
        :param pattern_to_positive_matches: the patterns to mine unifications for
        :param metapath_dict_positives: the metapaths of the positive samples
        :param sample_to_weight: the weight of each sample
        :param sample_name: the name of the sample
        :param update_cache: whether to update the cache or not
        :param pproc: the parallel processing object
        :param partitions_count: the number of partitions
        :return: the updated patterns and the elapsed time
        '''
        compute_unifications = self.algo_params["compute_unifications"]
        if compute_unifications:
            minsup_ratio = self.algo_params["minsup_ratio"]
            orient_gene_pairs = self.algo_params["orient_gene_pairs"]
            base_rule_size = len(pattern_to_positive_matches)
            cache_name = Cache.generate_cache_file_name("unification_mining", sample_name, self.algo_params, 'path_cutoff', 'excl_node_types', 'minsup_ratio', 'max_rule_length', 'orient_gene_pairs', 'compute_unifications')
            storage = Cache(cache_name, update_cache, single_file=True)
            updated_pattern_to_matches, elapsed_time = storage.get_or_store("", lambda x: self._run(pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, minsup_ratio, orient_gene_pairs, pproc, partitions_count))
            logger.info(f"... [Unifications] - {len(updated_pattern_to_matches) - base_rule_size} additional patterns generated (now {len(updated_pattern_to_matches)}).")
            return updated_pattern_to_matches, elapsed_time
        else:
            logger.info(f"Skipping unification mining (param compute_unifications={compute_unifications})")
            return pattern_to_positive_matches, 0

    def _run(self, pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, minsup_ratio, orient_gene_pairs, pproc=None, partitions_count=None):
        start = time.process_time()
        min_support_count = get_absolute_minsup(metapath_dict_positives, sample_to_weight, minsup_ratio)
        unified_pattern_to_pos_matches = MetapathPatternUnificationMiner.add_unifications(pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, min_support_count, orient_gene_pairs, pproc, partitions_count)
        elapsed_time = time.process_time() - start
        return unified_pattern_to_pos_matches, elapsed_time

    @staticmethod
    def add_unifications(pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, minsup, orient_gene_pairs, pproc=None, partitions_count=None):
        '''
        Add unifications to the patterns.
        :param pattern_to_positive_matches: the patterns to mine unifications for
        :param metapath_dict_positives: the metapaths of the positive samples
        :param sample_to_weight: the weight of each sample
        :param minsup: the minimum support count
        :param orient_gene_pairs: whether to orient gene pairs or not
        :param pproc: the parallel processing object
        :param partitions_count: the number of partitions
        :return: the updated patterns
        '''
        logger.info("Running unification mining...")

        rule_matches = []
        for pattern, matches in pattern_to_positive_matches.items():
            metapaths = pattern.metapaths
            if MetapathPatternUnificationMiner.is_unifiable(metapaths):
                rule_matches.append((metapaths, matches))

        if pproc:
            unifications = pproc.map_collect(MetapathPatternUnificationMiner.add_unification, rule_matches, partitions_count,
                                             {"metapath_dict_positives": metapath_dict_positives,
                                           "orient_gene_pairs": orient_gene_pairs})
        else:
            unifications = [MetapathPatternUnificationMiner.add_unification(r, metapath_dict_positives, orient_gene_pairs) for r in tqdm(rule_matches)]

        for rule_metapaths, unifications_to_matches in unifications:
            for unification, matches in unifications_to_matches.items():
                if valid_support(matches, minsup, sample_to_weight):
                    pattern = KGPattern(rule_metapaths, unification)
                    pattern_to_positive_matches[pattern] = matches

        return pattern_to_positive_matches

    @staticmethod
    def add_unification(metapath_set_and_matches, metapath_dict_positives, orient_gene_pairs):
        '''
        Add unifications to a pattern.
        :param metapath_set_and_matches: the metapaths and matches of the pattern
        :param metapath_dict_positives: the metapaths of the positive samples
        :param orient_gene_pairs: whether to orient gene pairs or not
        :return: the updated pattern and the unifications
        '''
        directions = [1] if orient_gene_pairs else [1, -1]

        unification_clause = defaultdict(set)
        metapath_set, matches = metapath_set_and_matches
        for match in matches:
            match_metapaths = metapath_dict_positives.get(match)
            for direction in directions:
                MetapathPatternUnificationMiner.add_match_to_unification(match, unification_clause, match_metapaths,
                                                                         metapath_set, direction)

        return metapath_set, unification_clause

    @staticmethod
    def add_match_to_unification(match, unification_clause, match_metapaths, metapath_set, direction):
        '''
        Add a match to a unification.
        :param match: the match to add
        :param unification_clause: the unification to add the match to
        :param match_metapaths: the metapaths of the match
        :param metapath_set: the metapaths of the pattern
        :param direction: the direction to add the match in
        '''
        node_to_mps = defaultdict(lambda: defaultdict(set))
        has_all_mps = True
        for mp in metapath_set:
            paths = match_metapaths.get(directs_metapath(mp, direction))
            if paths:
                for path in paths:
                    path = path[::direction]
                    node_position = 1
                    for node in path:
                        node_to_mps[node][mp].add(node_position)
                        node_position += 1
            else:
                has_all_mps = False
                break
        if has_all_mps:
            for node, mp_and_positions in node_to_mps.items():
                mp_and_positions_list = mp_and_positions.items()
                if len(mp_and_positions_list) >= 2:
                    for i in range(2, len(mp_and_positions_list) + 1):
                        for mp_comb in combinations(mp_and_positions_list, i):
                            unified_mps = [e[0] for e in mp_comb]
                            unified_positions = [e[1] for e in mp_comb]
                            for prod in product(*unified_positions):
                                unification = []
                                for unification_var in zip(unified_mps, prod):
                                    unification.append(unification_var)
                                unification_clause[tuple(unification)].add(match)

    @staticmethod
    def is_unifiable(metapath_set):
        '''
        Check if a set of metapaths is unifiable (i.e having some common intermediate node types).
        :param metapath_set: the set of metapaths
        :return: True if the set is unifiable, False otherwise
        '''
        if len(metapath_set) == 1:
            return False
        node_type_to_mp_count = defaultdict(int)
        for metapath in metapath_set:
            edge_types, node_types, edge_directions = metapath
            unique_intermediate_node_types = set(node_types[1:-1])
            for node_type in unique_intermediate_node_types:
                node_type_to_mp_count[node_type] += 1
        return any(i >= 2 for i in node_type_to_mp_count.values())









