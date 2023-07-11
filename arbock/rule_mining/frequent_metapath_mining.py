from ..utils.cache_utils import Cache
from ..utils.dict_utils import default_to_regular
from ..rule_mining.rule_utils import get_absolute_minsup, valid_support, directs_metapath
from ..rule_mining.rule import KGPattern
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
import time
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


class FrequentMetapathSetMiner:
    '''
    This class is responsible for mining frequent metapath sets (i.e frequent itemsets where the items are metapaths)
    using the Apriori algorithm (Agrawal and Srikant. Fast algorithms for mining association rules. Proc. 20th int. conf. very large data bases, VLDB. Vol. 1215. 1994).
    It takes the associated metapaths for each samples and returns the frequent metapath sets
    passing the predefined minimum support ratio cutoff and maximum rule length (metapath set size) cutoff.
    Depending on the parameter orient_gene_pairs, the metapaths are either oriented or not.

    Faster implementation using transaction id trick from  https://arxiv.org/pdf/1403.3948.pdf (implementation credit: https://stackoverflow.com/questions/61621532)
    Further speedup using parallelisation during the candidate generation and support counting steps.
    '''

    def __init__(self, algo_params):
        self.algo_params = algo_params

    def run(self, metapath_dict, entity_pair_to_weight, sample_name, update_cache=False, pproc=None, partitions_count=None):
        '''
        Mine frequent metapaths from a knowledge graph (with caching).
        Requires the parameters 'minsup_ratio',  'max_rule_length' and 'orient_gene_pairs' in algo_params
        :param metapath_dict: A dictionary of metapaths.
        :param entity_pair_to_weight: A dictionary of entity pair weights.
        :param sample_name: The name of the sample for caching.
        :param update_cache: Whether to update the cache.
        :param pproc: A parallel processing context (None for no parallelization).
        :param partitions_count: The number of partitions to chunk data into for parallel computing (None for auto).
        '''
        minsup_ratio = self.algo_params["minsup_ratio"]
        max_rule_length = self.algo_params["max_rule_length"]
        orient_gene_pairs = self.algo_params["orient_gene_pairs"]
        logger.info(f"Running frequent metapath set mining. [minsup_ratio={minsup_ratio}, max_rule_length={max_rule_length}, orient_gene_pairs={orient_gene_pairs}]")
        cache_name = Cache.generate_cache_file_name("frequent_metapath_mining", sample_name, self.algo_params, 'path_cutoff', 'excl_node_types', 'minsup_ratio', 'max_rule_length', 'orient_gene_pairs')
        storage = Cache(cache_name, update_cache, single_file=True)
        pattern_to_matches, elapsed_time = storage.get_or_store("", lambda x: self._run(metapath_dict, entity_pair_to_weight, minsup_ratio, max_rule_length, orient_gene_pairs, pproc, partitions_count))
        logger.info(f"{len(pattern_to_matches)} frequent patterns generated.")
        return pattern_to_matches, elapsed_time

    def _run(self, metapath_dict_positive, sample_to_weight, minsup_ratio, max_pattern_size, orient_gene_pairs, pproc=None, partitions_count=None):
        start = time.process_time()
        min_support_count = get_absolute_minsup(metapath_dict_positive, sample_to_weight, minsup_ratio)
        pattern_to_pos_matches = self.apriori(metapath_dict_positive, sample_to_weight, min_support_count, max_pattern_size,
                                              orient_gene_pairs, pproc, partitions_count)
        elapsed_time = time.process_time() - start
        return pattern_to_pos_matches, elapsed_time

    def apriori(self, metapath_dict, sample_to_weight, min_support, max_pattern_size, orient_gene_pairs, pproc=None, partitions_count=None):
        '''
        The Apriori algorithm for mining frequent metapath sets.
        :param metapath_dict: A dictionary of metapaths.
        :param sample_to_weight: A dictionary of sample weights.
        :param min_support: The minimum support count.
        :param max_pattern_size: The maximum pattern size.
        :param orient_gene_pairs: Whether to orient gene pairs.
        :param pproc: A parallel processing context (None for no parallelization).
        :param partitions_count: The number of partitions to chunk data into for parallel computing (None for auto).
        :return: A dictionary of frequent metapath sets.
        '''

        directions = [1] if orient_gene_pairs else [1, -1]

        pattern_to_paths = defaultdict(lambda: defaultdict(set))

        for sample, metapath_to_paths in metapath_dict.items():
            for metapath in metapath_to_paths:
                for direction in directions:
                    pattern_to_paths[sample][direction].add(directs_metapath(metapath, direction))

        pattern_to_paths = default_to_regular(pattern_to_paths)

        L1 = FrequentMetapathSetMiner.generate_first_level_patterns(pattern_to_paths, min_support, sample_to_weight)
        L = {1: L1}
        for k in range(2, max_pattern_size + 1):
            logger.info(f"<Running> Generating patterns of size: {k}")
            if len(L[k - 1]) < 2:
                break
            Ck = FrequentMetapathSetMiner.generate_all_candidate_patterns(L[k - 1], pproc, partitions_count)
            L[k] = FrequentMetapathSetMiner.get_all_valid_patterns(Ck, min_support, pattern_to_paths, sample_to_weight, pproc, partitions_count)
            logger.info(f"... [{k}] - {len(L[k])} patterns generated.")

        rule_to_positive_matches = {}
        for pattern_size, Lk in L.items():
            if Lk and pattern_size != 1:
                for pattern, matching_sample_ids in Lk.items():
                    pattern = KGPattern(pattern)
                    rule_to_positive_matches[pattern] = set(matching_sample_ids)
        return rule_to_positive_matches

    @staticmethod
    def generate_first_level_patterns(pattern_to_paths, min_support, sample_to_weight=None):
        '''
        Generate the first level of patterns.
        :param pattern_to_paths: A dictionary of patterns to paths.
        :param min_support: The minimum support count.
        :param sample_to_weight: A dictionary of sample weights.
        :return: A dictionary of patterns to samples.
        '''
        level_1 = defaultdict(set)
        for sample_id, metapath_info in pattern_to_paths.items():
            for metapath_direction, metapath_paths in metapath_info.items():
                for metapath in metapath_paths:
                    level_1[(metapath,)].add(sample_id)
        return {pattern: samples for pattern, samples in level_1.items()
                if valid_support(samples, min_support, sample_to_weight)}

    @staticmethod
    def generate_all_candidate_patterns(previous_level_patterns, pproc=None, partitions_count=None):
        '''
        Generate all candidate metapath patterns from the previous level.
        :param previous_level_patterns: A dictionary of patterns to samples.
        :param pproc: A parallel processing context (None for no parallelization).
        :param partitions_count: The number of partitions to chunk data into for parallel computing (None for auto).
        :return: A dictionary of patterns to samples.
        '''
        all_candidate_patterns = {}
        if pproc:
            results = pproc.map_collect(FrequentMetapathSetMiner.generate_candidate_patterns, list(previous_level_patterns.items()), partitions_count,
                                        {"previous_level_patterns": previous_level_patterns})
        else:
            results = [FrequentMetapathSetMiner.generate_candidate_patterns(patterns, previous_level_patterns) for patterns in tqdm(previous_level_patterns.items())]
        for pattern_to_samples in results:
            if pattern_to_samples:
                all_candidate_patterns.update(pattern_to_samples)
        return all_candidate_patterns

    @staticmethod
    def generate_candidate_patterns(seed_pattern, previous_level_patterns):
        '''
        Generate metapath candidate patterns from a seed pattern.
        :param seed_pattern: A seed pattern.
        :param previous_level_patterns: A dictionary of patterns to samples.
        :return: A dictionary of patterns to samples.
        '''
        pattern_1, p1_samples = seed_pattern
        candidate_patterns = {}
        for pattern_2, p2_samples in previous_level_patterns.items():
            if pattern_1[:-1] == pattern_2[:-1] and pattern_1[-1] < pattern_2[-1]:
                new_pattern = tuple([*pattern_1, pattern_2[-1]])
                if not FrequentMetapathSetMiner.has_infrequent_subset(new_pattern, previous_level_patterns):
                    candidate_patterns[new_pattern] = p1_samples.intersection(p2_samples)
        return candidate_patterns

    @staticmethod
    def get_all_valid_patterns(candidate_patterns, min_support, pattern_to_paths, sample_to_weight, pproc=None, partitions_count=None):
        '''
        Get all valid metapath patterns from a set of candidate patterns.
        :param candidate_patterns: A dictionary of candidate patterns to samples.
        :param min_support: The minimum support count.
        :param pattern_to_paths: A dictionary of patterns to paths.
        :param sample_to_weight: A dictionary of sample weights.
        :param pproc: A parallel processing context (None for no parallelization).
        :param partitions_count: The number of partitions to chunk data into for parallel computing (None for auto).
        :return: A dictionary of patterns to samples.
        '''
        all_valid_patterns = {}
        if pproc:
            results = pproc.map_collect(FrequentMetapathSetMiner.get_valid_pattern, candidate_patterns.items(), partitions_count,
                                        {"min_support": min_support, "pattern_to_paths": pattern_to_paths, "sample_to_weight": sample_to_weight})
        else:
            results = [FrequentMetapathSetMiner.get_valid_pattern((candidate, newTIDs), min_support, pattern_to_paths, sample_to_weight) for candidate, newTIDs in tqdm(candidate_patterns.items())]
        for result in results:
            if result:
                pattern, sample_ids = result
                all_valid_patterns[pattern] = sample_ids
        return all_valid_patterns

    @staticmethod
    def get_valid_pattern(candidate_new_samples, min_support, pattern_to_paths, sample_to_weight=None):
        '''
        Get a valid metapath pattern from a candidate pattern.
        :param candidate_new_samples: A candidate pattern and its new samples.
        :param min_support: The minimum support count.
        :param pattern_to_paths: A dictionary of patterns to paths.
        :param sample_to_weight: A dictionary of sample weights.
        :return: A valid pattern and its samples.
        '''
        candidate, new_samples = candidate_new_samples
        valid_sample_ids = set()
        for sample_id in new_samples:
            for metapath_direction, metapath_to_paths in pattern_to_paths[sample_id].items():
                matching_metapath_count = 0
                for metapath in metapath_to_paths:
                    if metapath in candidate:
                        matching_metapath_count += 1
                if matching_metapath_count == len(candidate):
                    valid_sample_ids.add(sample_id)
        if not valid_support(valid_sample_ids, min_support, sample_to_weight):
            return None

        return candidate, valid_sample_ids

    @staticmethod
    def has_infrequent_subset(candidate_pattern, previous_level_patterns):
        '''
        Check if a candidate pattern has an infrequent subset.
        :param candidate_pattern: A candidate pattern.
        :param previous_level_patterns: A dictionary of patterns to samples.
        :return: True if the candidate pattern has an infrequent subset, False otherwise.
        '''
        return any(subset not in previous_level_patterns for subset in combinations(candidate_pattern, len(candidate_pattern) - 1))
















