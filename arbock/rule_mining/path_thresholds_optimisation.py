from .rule_querying import MetapathRuleMatcher
from .rule_utils import get_absolute_minsup
from ..utils.cache_utils import Cache
from packaging import version
from tqdm import tqdm
import random
from random import choices
import scipy
from scipy import optimize
import numpy as np
import time
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


class PathThresholdOptimiser:
    '''
    This class is designed to optimize path reliability thresholds that are associated with the metapaths of a particular rule.
    The primary objective is to identify an optimal set of thresholds (one threshold per metapath) that will minimize a cost function.

    The cost function (C_r) of a rule given a threshold (t), bound in the interval [0,1], is based on:
    - S: The support of the rule (how many positive training examples matches the rule)
    - P: The number of valid paths on average for all positive training examples matching the rule
    S and P varies based on the choice of thresholds (_t) compared to no thresholding (_init).

    We can define the cost function as:

    C = ((1 - (S_t / S_init)) + (P_t / P_init)) / 2 if S_t > minsup else 1

    Optimizing this cost function lead to reducing the number of paths per positive training example while maintaining a high support.

    Two distinct optimization strategies have been implemented, that can be used based on algo_params["optimization_strategy"]:

    1. Greedy Optimization (optimization_strategy=GREEDY):
        This strategy iteratively selects and optimizes each variable based on a locally optimal choice, aiming to minimize the cost function.
        It is a heuristic approach that can provide a sufficiently optimal solution in a relatively quick time frame.
    2. Differential Evolution (optimization_strategy=DE): A more global optimization method that leverages evolutionary algorithms.
    It manipulates a population of candidate solutions to explore the search-space and can potentially find a globally optimal solution, albeit at a higher computational cost.
    '''

    np.seterr(invalid='ignore')

    _decimal_precision = 100

    _DE_best_hyperparameters = {
        "pop_size": 50,
        "recombination_rate": 0.7,
        "mutation_rate": (0.5, 1),
        "strategy": "best1bin",
        "tol": 0.01,
        "maxiter": 1000
    }

    def __init__(self, algo_params):
        self.algo_params = algo_params

    def run(self, pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, sample_name, update_cache=False, pproc=None, partitions_count=None):
        '''
        Run the metapath threshold optimization process.
        :param pattern_to_positive_matches: A dictionary mapping a pattern to the positive training examples that matches it.
        :param metapath_dict_positives: A dictionary mapping a metapath to the positive training examples that matches it.
        :param sample_to_weight: A dictionary mapping a sample to its weight.
        :param sample_name: The name of the sample (used for caching).
        :param update_cache: Whether to update the cache or not.
        :param pproc: The parallel processing object (multiprocessing or spark)
        :param partitions_count: The number of partitions to use for parallel processing.
        :return: A dictionary mapping a pattern to its optimal thresholds.
        '''
        optimize_metapath_thresholds = self.algo_params["optimize_metapath_thresholds"]
        if optimize_metapath_thresholds:
            minsup_ratio = self.algo_params["minsup_ratio"]
            max_rule_length = self.algo_params["max_rule_length"]
            orient_gene_pairs = self.algo_params["orient_gene_pairs"]
            optimize_metapath_thresholds = self.algo_params["optimize_metapath_thresholds"]

            logger.info(f"Running rule path thresholds optimisation. [Optimisation method={optimize_metapath_thresholds}]")

            if optimize_metapath_thresholds.upper() not in ["DE", "GREEDY"]:
                raise Exception(f"Unsupported parameter algo_params['optimize_metapath_thresholds']={optimize_metapath_thresholds}")
            output_name = Cache.generate_cache_file_name("path_thresholds_optimisation", sample_name, self.algo_params, 'path_cutoff', 'excl_node_types', 'minsup_ratio', 'max_rule_length', 'orient_gene_pairs', 'compute_unifications', 'optimize_metapath_thresholds')
            storage = Cache(output_name, update_cache, single_file=True)
            thresholded_pattern_to_matches, elapsed_time = storage.get_or_store("", lambda x: self._run(pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, minsup_ratio, max_rule_length, orient_gene_pairs, optimize_metapath_thresholds, pproc=pproc, partitions_count=partitions_count))
            logger.info(f"Successfully optimised path thresholds for {len(thresholded_pattern_to_matches)} patterns")
            return thresholded_pattern_to_matches, elapsed_time
        else:
            logger.info(f"Skipping path thresholds optimisation (param optimize_metapath_thresholds={optimize_metapath_thresholds})")
            return pattern_to_positive_matches, 0

    def _run(self, pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, minsup_ratio, max_rule_length, orient_gene_pairs, optimize_metapath_thresholds, pproc=None, partitions_count=None):
        start = time.process_time()
        if len(pattern_to_positive_matches) == 0:
            return {}, 0
        minsup = get_absolute_minsup(metapath_dict_positives, sample_to_weight, minsup_ratio)

        init_pops = {}
        for metapath_count in range(1, max_rule_length+1):
            init_pops[metapath_count] = PathThresholdOptimiser.generate_init_population(metapath_count, PathThresholdOptimiser._DE_best_hyperparameters["pop_size"])

        vectorize = version.parse(scipy.__version__) >= version.parse("1.9")

        if pproc:
            results = pproc.map_collect(PathThresholdOptimiser.find_optimal_metapath_thresholds, pattern_to_positive_matches.items(), partitions_count,
                                        shared_variables_dict={"metapath_dict":metapath_dict_positives, "init_pops": init_pops, "minsup": minsup,
                                               "sample_to_weight": sample_to_weight, "orient_gene_pairs": orient_gene_pairs,
                                               "optimize_metapath_thresholds": optimize_metapath_thresholds, "vectorize": vectorize})
        else:
            results = [PathThresholdOptimiser.find_optimal_metapath_thresholds(pattern_and_pos_matches, metapath_dict_positives, init_pops, minsup, sample_to_weight, orient_gene_pairs, optimize_metapath_thresholds, vectorize) for pattern_and_pos_matches in tqdm(pattern_to_positive_matches.items())]

        thresholded_pattern_to_pos_matches = {pattern:pos_matches for pattern, pos_matches in results}

        elapsed_time = time.process_time() - start

        return thresholded_pattern_to_pos_matches, elapsed_time

    @staticmethod
    def find_optimal_metapath_thresholds(pattern_and_pos_matches, metapath_dict, init_pops, minsup, sample_to_weight, orient_gene_pairs, optimize_metapath_thresholds, vectorize=True):
        '''
        Find the optimal metapath thresholds for a given pattern.
        :param pattern_and_pos_matches: A tuple of (pattern, positive matches).
        :param metapath_dict: A dictionary mapping a metapath to the positive training examples that matches it.
        :param init_pops: A dictionary mapping a metapath count to its initial population.
        :param minsup: The minimum support.
        :param sample_to_weight: A dictionary mapping a sample to its weight.
        :param orient_gene_pairs: Whether to orient the gene pairs or not.
        :param optimize_metapath_thresholds: The optimisation method to use.
        :param vectorize: Whether to use vectorization or not.
        :return: A tuple of (pattern, positive matches) with the optimal metapath thresholds.
        '''
        pattern, rule_pos_matches = pattern_and_pos_matches
        metapath_count = len(pattern.metapaths)
        gp_to_subgraph_scores, avg_path_count = PathThresholdOptimiser.get_all_gene_pairs_subgraph_scores(pattern, rule_pos_matches, metapath_dict, sample_to_weight, orient_gene_pairs)
        logger.debug(f"Got all subgraph score for #{len(rule_pos_matches)} matches, with avg_path_count = {avg_path_count}")

        path_score_data, weights, ordered_gene_pairs = PathThresholdOptimiser.prepare_data_for_optimization(gp_to_subgraph_scores, sample_to_weight)

        if optimize_metapath_thresholds.upper() == "DE":
            metapath_optimal_thresholds = PathThresholdOptimiser.differential_evolution_optimization(path_score_data, weights, metapath_count, avg_path_count, minsup, init_pops, vectorize)
        elif optimize_metapath_thresholds.upper() == "GREEDY":
            metapath_optimal_thresholds = PathThresholdOptimiser.greedy_optimization(path_score_data, weights, metapath_count, avg_path_count, minsup)
        else:
            raise Exception(f"Unsupported argument optimize_metapath_thresholds={optimize_metapath_thresholds}")

        logger.debug(f"Got optimal thresholds = {metapath_optimal_thresholds}")
        after_threshold_matches = PathThresholdOptimiser.get_threshold_based_matchings(metapath_optimal_thresholds, path_score_data, ordered_gene_pairs)
        logger.debug(f"# new matches = {len(after_threshold_matches)}")
        pattern.path_thresholds = metapath_optimal_thresholds
        logger.debug(f"{pattern} | Matches: {len(rule_pos_matches)} -> {len(after_threshold_matches)}")
        return pattern, after_threshold_matches

    @staticmethod
    def greedy_optimization(path_score_data, weights, metapath_count, initial_avg_path_count, minsup):
        '''
        Find the optimal metapath thresholds using a greedy approach.
        :param path_score_data: The path score data.
        :param weights: The weights of the samples.
        :param metapath_count: The number of metapaths.
        :param initial_avg_path_count: The initial average path count.
        :param minsup: The minimum support.
        :return: The optimal metapath thresholds.
        '''
        default_candidates = np.array([i for i in range(0,PathThresholdOptimiser._decimal_precision+1)])
        num_steps = len(default_candidates)
        solution = np.zeros(metapath_count, dtype=int)
        free_variables = set(range(metapath_count))

        for _ in range(metapath_count):
            best_cost = float('inf')
            best_variable = None
            best_value = None

            for var_idx in free_variables:
                candidates = np.zeros((metapath_count, num_steps), dtype=int)
                candidates[var_idx, :] = default_candidates
                candidates += np.expand_dims(solution, axis=-1)

                costs = PathThresholdOptimiser.compute_cost_vector(candidates, path_score_data, initial_avg_path_count, minsup, weights, unique_thr=False)
                min_cost_idx = np.argmin(costs)
                min_cost = costs[min_cost_idx]

                if min_cost < best_cost:
                    best_cost = min_cost
                    best_variable = var_idx
                    best_value = candidates[var_idx, min_cost_idx]

            solution[best_variable] = best_value
            free_variables.remove(best_variable)

        return solution / PathThresholdOptimiser._decimal_precision

    @staticmethod
    def differential_evolution_optimization(path_score_data, weights, metapath_count, initial_avg_path_count, minsup, init_pops, vectorize=True):
        '''
        Find the optimal metapath thresholds using differential evolution.
        :param path_score_data: The path score data.
        :param weights: The weights of the samples.
        :param metapath_count: The number of metapaths.
        :param initial_avg_path_count: The initial average path count.
        :param minsup: The minimum support.
        :param init_pops: A dictionary mapping a metapath count to its initial population.
        :param vectorize: Whether to use vectorization or not.
        :return: The optimal metapath thresholds.
        '''

        bounds = [(0,PathThresholdOptimiser._decimal_precision)] * metapath_count

        init_pop = np.array(init_pops[metapath_count] * PathThresholdOptimiser._decimal_precision, dtype=int)

        maxiter = PathThresholdOptimiser._DE_best_hyperparameters["maxiter"]
        recombination_rate = PathThresholdOptimiser._DE_best_hyperparameters["recombination_rate"]
        mutation_rate = PathThresholdOptimiser._DE_best_hyperparameters["mutation_rate"]
        strategy = PathThresholdOptimiser._DE_best_hyperparameters["strategy"]
        tol = PathThresholdOptimiser._DE_best_hyperparameters["tol"]

        if vectorize:  # Much faster implementation through vectorization
            integrality = [True] * metapath_count
            res = optimize.differential_evolution(PathThresholdOptimiser.compute_cost_vector, bounds,
                                                        args=(path_score_data, initial_avg_path_count, minsup, weights),
                                                        popsize=len(init_pop), init=init_pop, strategy=strategy, maxiter=maxiter, tol=tol,
                                                        mutation=mutation_rate, recombination=recombination_rate, polish=True,
                                                        updating="deferred", vectorized=True, integrality=integrality)
        else:
            res = optimize.differential_evolution(PathThresholdOptimiser.compute_cost_vector, bounds,
                                                        args=(path_score_data, initial_avg_path_count, minsup, weights, True),
                                                        popsize=len(init_pop), init=init_pop, strategy=strategy, maxiter=maxiter, tol=tol,
                                                        mutation=mutation_rate, recombination=recombination_rate, polish=True)

        optimized_thresholds = res.x

        rule_thresholds = optimized_thresholds / PathThresholdOptimiser._decimal_precision

        return rule_thresholds

    @staticmethod
    def convert_rule_support_and_avg_path_count_to_cost(score_components, minsup):
        '''
        Convert the rule support and average path count to a cost.
        :param score_components: The rule support and average path count.
        :param minsup: The minimum support.
        :return: The rule cost after thresholding.
        '''
        support_after_thr, support_reduction, path_count_reduction = score_components
        if support_after_thr < minsup:
            return 1.0
        return ((1-support_reduction) + path_count_reduction) / 2

    @staticmethod
    def get_cost_vector_from_path_counts(after_thresholding_matrix, weights, initial_avg_path_count, minsup):
        '''
        Compute the cost vector from the path counts.
        :param after_thresholding_matrix: The path counts after thresholding.
        :param weights: The weights of the samples.
        :param initial_avg_path_count: The initial average path count.
        :param minsup: The minimum support.
        :return: The cost vector after thresholding.
        '''
        initial_support = weights.sum()
        weighted_match_matrix = (after_thresholding_matrix>0).astype(int) * weights
        support_after_thr = weighted_match_matrix.sum(axis=0)
        avg_path_count_after_thr = (after_thresholding_matrix * weights).sum(axis=0) / support_after_thr

        support_reduction = support_after_thr / initial_support
        path_count_reduction = avg_path_count_after_thr / initial_avg_path_count

        stacked_cost_components = np.vstack([support_after_thr, support_reduction, path_count_reduction])
        cost_vector = np.apply_along_axis(PathThresholdOptimiser.convert_rule_support_and_avg_path_count_to_cost, 0, stacked_cost_components, minsup)
        return cost_vector

    @staticmethod
    def apply_thresholds(thresholds, scores_matrix):
        '''
        Apply the thresholds to the scores matrix.
        :param thresholds: The thresholds (numpy array).
        :param scores_matrix: The scores matrix (numpy array).
        :return: The path counts after applying the thresholds.
        '''
        path_counts = scores_matrix[np.arange(thresholds.size), thresholds]
        if np.any(path_counts == 0):
            return 0
        else:
            return np.sum(path_counts)

    @staticmethod
    def get_path_counts_after_thresholds(mult_thresholds_matrix, data):
        '''
        Get the path counts after applying the thresholds.
        :param mult_thresholds_matrix: The thresholds matrix (numpy array).
        :param data: The data (list of dictionaries).
        :return: The path counts after applying the thresholds.
        '''
        after_threshold_match_to_valid_path_scores = []
        for instance_scores in data:
            direction_path_counts = []
            for direction, path_score_matrix in instance_scores.items():
                path_count_vector = np.apply_along_axis(PathThresholdOptimiser.apply_thresholds, 0, mult_thresholds_matrix, path_score_matrix)
                direction_path_counts.append(path_count_vector)
            after_threshold_match_to_valid_path_scores.append(np.max(direction_path_counts, axis=0))
        return np.array(after_threshold_match_to_valid_path_scores)

    @staticmethod
    def compute_cost_vector(thresholds, data, initial_avg_path_count, minsup, weights, unique_thr=False):
        '''
        Compute the cost vector.
        :param thresholds: The thresholds (numpy array).
        :param data: The data (list of dictionaries).
        :param initial_avg_path_count: The initial average path count.
        :param minsup: The minimum support.
        :param weights: The weights of the samples.
        :param unique_thr: Whether the thresholds are unique.
        :return: The cost vector.
        '''
        if unique_thr:
            thresholds = thresholds.reshape(-1,1)
        thresholds = np.array(np.round(thresholds), dtype=int)
        after_threshold_matches_to_path_count = PathThresholdOptimiser.get_path_counts_after_thresholds(thresholds, data)
        cost = PathThresholdOptimiser.get_cost_vector_from_path_counts(after_threshold_matches_to_path_count, weights, initial_avg_path_count, minsup)
        return cost

    @staticmethod
    def order_metapath_scores(rule_antecedent, metapath_scores):
        '''
        Order the metapath scores according to the order of the metapaths in the rule antecedent.
        :param rule_antecedent: The rule antecedent.
        :param metapath_scores: The metapath scores.
        :return: The ordered metapath scores.
        '''
        ordered_metapath_scores = []
        path_count = 0
        for metapath in rule_antecedent.metapaths:
            path_scores = metapath_scores[metapath]
            ordered_metapath_scores.append(path_scores)
            path_count += len(path_scores)
        return ordered_metapath_scores, path_count

    @staticmethod
    def get_metapath_scores(metapath_to_paths):
        '''
        Get the metapath scores.
        :param metapath_to_paths: The metapath to paths dictionary.
        :return: The metapath scores.
        '''
        metapath_scores = {}
        for metapath, paths in metapath_to_paths.items():
            metapath_scores[metapath] = np.array([path_score for path, path_score in paths.items()])
        return metapath_scores

    @staticmethod
    def get_metapath_subgraph_scores(rule_antecedent, subgraph):
        '''
        Get the metapath scores of the subgraph.
        :param rule_antecedent: The rule antecedent.
        :param subgraph: The subgraph.
        :return: The metapath scores of the subgraph.
        '''
        direction_to_metapath_scores = {} # dict to handle both direction in case of bidirectional matching
        all_direction_path_counts = [] # list to handle both direction in case of bidirectional matching
        for direction, unif_to_metapaths in subgraph.items():
            direction_to_metapath_scores[direction] = [np.array([])] * len(rule_antecedent.metapaths)
            ununified_metapath_scores = {}
            path_count_total = 0
            if None in unif_to_metapaths:
                ununified_metapath_scores = PathThresholdOptimiser.get_metapath_scores(unif_to_metapaths[None])
                if len(unif_to_metapaths) == 1:
                    metapath_scores, path_count = PathThresholdOptimiser.order_metapath_scores(rule_antecedent, ununified_metapath_scores)
                    direction_to_metapath_scores[direction] = metapath_scores
                    path_count_total = path_count
            for unif, metapath_to_paths in unif_to_metapaths.items():
                if unif is None:
                    continue
                unified_metapath_scores = PathThresholdOptimiser.get_metapath_scores(metapath_to_paths)
                merged_metapath_scores = {**ununified_metapath_scores, **unified_metapath_scores}
                metapath_scores, path_count = PathThresholdOptimiser.order_metapath_scores(rule_antecedent, merged_metapath_scores)
                for i, path_scores in enumerate(metapath_scores):
                    existing_path_scores = direction_to_metapath_scores[direction][i]
                    direction_to_metapath_scores[direction][i] = np.concatenate((path_scores, existing_path_scores))
                path_count_total += path_count
            all_direction_path_counts.append(path_count_total)

        return direction_to_metapath_scores, max(all_direction_path_counts)

    @staticmethod
    def get_all_gene_pairs_subgraph_scores(pattern, pos_matches, metapath_dict, sample_to_weight, orient_gene_pairs):
        '''
        Get the subgraph scores of all gene pairs.
        :param pattern: The rule pattern.
        :param pos_matches: The positive matches.
        :param metapath_dict: The metapath dictionary.
        :param sample_to_weight: The sample weight dictionary.
        :param orient_gene_pairs: Whether to orient the gene pairs.
        :return: The subgraph scores of all gene pairs.
        '''
        positive_pairs_scores = {}
        weighted_path_counts = []
        for pos_match in pos_matches:
            pos_subgraph = MetapathRuleMatcher.evaluate_to_get_paths(pattern, metapath_dict[pos_match], orient_gene_pairs)
            subgraph_scores, path_count = PathThresholdOptimiser.get_metapath_subgraph_scores(pattern, pos_subgraph)
            positive_pairs_scores[pos_match] = subgraph_scores
            weighted_path_counts.append(sample_to_weight[pos_match] * path_count)

        avg_path_count = sum(weighted_path_counts) / sum(sample_to_weight[i] for i in pos_matches)

        return positive_pairs_scores, avg_path_count

    @staticmethod
    def prepare_data_for_optimization(gp_to_subgraph_scores, sample_to_weight):
        '''
        Prepare the data for optimization.
        :param gp_to_subgraph_scores: The gene pair to subgraph scores dictionary.
        :param sample_to_weight: The sample weight dictionary.
        :return: The path score data, weights, and ordered gene pairs.
        '''
        path_score_data = []
        weights = []
        ordered_gene_pairs = []
        threshold_candidates = np.array([i/PathThresholdOptimiser._decimal_precision for i in range(0,PathThresholdOptimiser._decimal_precision+1)])
        for gene_pair, direction_to_subgraph_scores in gp_to_subgraph_scores.items():
            formatted_data = {}
            for direction, subgraph_scores in direction_to_subgraph_scores.items():
                metapath_cumsumpaths = []
                for metapath_scores in subgraph_scores:
                    # Optimisation trick by pre-counting upward #paths per minimum thresholds, for all possible thresholds defined by decimal_precision
                    cum_sum_paths = np.cumsum(np.bincount(np.digitize(metapath_scores, threshold_candidates)-1, minlength=PathThresholdOptimiser._decimal_precision+1)[::-1])[::-1]
                    metapath_cumsumpaths.append(cum_sum_paths)
                formatted_data[direction] = np.array(metapath_cumsumpaths)
            path_score_data.append(formatted_data)
            weights.append(sample_to_weight[gene_pair])
            ordered_gene_pairs.append(gene_pair)
        weights = np.array(weights).reshape(-1, 1)
        return path_score_data, weights, ordered_gene_pairs

    @staticmethod
    def generate_random_vector_summing_to(vec_size, summing_limit):
        '''
        Generate a random vector summing to a given limit.
        :param vec_size: The vector size.
        :param summing_limit: The summing limit.
        :return: The random vector summing to the given limit.
        '''
        rands = []
        for i in range(vec_size):
            rands.append(random.uniform(0, summing_limit))
        a = []
        for r in rands[:-1]:
            a.append(r * summing_limit / (sum(rands) + 0.000000001))
        last = summing_limit
        for e in a:
            last -= e
        return a + [last]

    @staticmethod
    def generate_init_population(x, M):
        '''
        Generate the initial population for the differential evolution optimisation strategy.
        :param x: The vector size.
        :param M: The number of samples.
        :return: The initial population.
        '''
        limits = np.array([0.25, 0.5, 0.75, 1])
        sample_sizes = choices(limits, k=M)[:-1]
        samples = []
        samples.append([0] * x)
        for max_limit in sample_sizes:
            limit = random.uniform(0, max_limit)
            vec = PathThresholdOptimiser.generate_random_vector_summing_to(x, limit)
            samples.append(vec)
        return np.array(samples)

    @staticmethod
    def get_threshold_based_matchings(metapath_information_thresholds, path_score_data, ordered_gene_pairs):
        '''
        Get the gene pairs that match the thresholds.
        :param metapath_information_thresholds: The metapath information thresholds.
        :param path_score_data: The path score data.
        :param ordered_gene_pairs: The ordered gene pairs.
        :return: The gene pairs that match the thresholds.
        '''
        thresholds = metapath_information_thresholds.reshape(-1,1)
        thresholds = np.array(np.round(thresholds * PathThresholdOptimiser._decimal_precision), dtype=int)

        after_threshold_match_to_valid_path_scores = PathThresholdOptimiser.get_path_counts_after_thresholds(thresholds, path_score_data)

        match_indexes = np.where(after_threshold_match_to_valid_path_scores != 0)[0]
        return set([gene_pair for gene_pair_index, gene_pair in enumerate(ordered_gene_pairs) if gene_pair_index in match_indexes])





