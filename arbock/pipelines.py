from .rule_mining.frequent_metapath_mining import FrequentMetapathSetMiner
from .rule_mining.unification_mining import MetapathPatternUnificationMiner
from .rule_mining.rule_pruning import RulePruner
from .rule_mining.path_thresholds_optimisation import PathThresholdOptimiser
from .rule_mining.rule_querying import MetapathRuleMatcher
from .rule_mining.rule import Rule
from .model.decision_set_classifier import DecisionSetClassifier
import numpy as np
import time

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"


def train_decision_set_model(relevant_rules, training_positives, training_negatives, sample_to_weight, algo_params, cpu_cores=0):
    """
    Utils methods to train a decision set classifier with the given relevant rules and training data.

    Parameters:
    - relevant_rules: List of relevant rules for training
    - training_positives: List of positive training instances (gene pairs)
    - training_negatives: List of negative training instances (gene pairs)
    - sample_to_weight: Dictionary mapping instances to their weights (optional)
    - alpha: Trade-off parameter for combining true positive rate and false positive rate

    Returns:
    - rule_set_classifier: Trained rule set classifier
    """
    alpha = algo_params["alpha"]
    rule_matcher = MetapathRuleMatcher(algo_params)

    sample_list = training_positives + training_negatives
    X_train_list = []
    y_train_list = []
    sample_weight_list = []
    for gene_pair in sample_list:
        X_train_list.append([gene_pair, None])
        y_train_list.append(1 if gene_pair in training_positives else 0)
        if sample_to_weight:
            sample_weight_list.append(sample_to_weight[gene_pair])
    X_train = np.array(X_train_list, dtype=object)
    y_train = np.array(y_train_list)
    sample_weight = np.array(sample_weight_list) if sample_to_weight else None

    # Model training
    rule_set_classifier = DecisionSetClassifier(relevant_rules, rule_matcher, alpha, cpu_cores=cpu_cores)
    rule_set_classifier.fit(X_train, y_train, sample_weight=sample_weight)

    return rule_set_classifier


def mine_relevant_rules(training_positives, training_negatives, metapath_dict, sample_to_weight, algo_params, sample_name, update_cache=False, pproc=None):
    '''
    Utils method to mine relevant metapath-based rules for the given training data and apply pruning methods.

    Parameters:
    - training_positives: List of positive training instances (entity pairs)
    - training_negatives: List of negative training instances (entity pairs)
    - metapath_dict: Dictionary mapping entity pairs to their metapaths
    - sample_to_weight: Dictionary mapping instances to their weights (optional)
    - algo_params: Dictionary of all framework parameters
    - sample_name: Name of the analysis sample (for caching)
    - update_cache: Boolean flag to update the cache
    - pproc: Parallel processing context

    Returns:
    - relevant_rules: List of relevant rules
    '''
    rule_list, positive_matches_to_rule_ids, t1 = mine_candidate_rules(training_positives, metapath_dict, sample_to_weight, algo_params, sample_name, pproc=pproc, update_cache=update_cache)
    relevant_rules, t2 = apply_and_prune_rules(rule_list, positive_matches_to_rule_ids, training_negatives, metapath_dict, sample_to_weight, algo_params, sample_name, pproc=pproc, update_cache=update_cache)
    elapsed_time = t1 + t2
    return relevant_rules, elapsed_time


def mine_candidate_rules(training_positives, metapath_dict, sample_to_weight, algo_params, sample_name, update_cache=False, pproc=None):
    '''
    Utils method to mine candidate metapath-based rules for the given training data.
    Parameters:
    - training_positives: List of positive training instances (entity pairs)
    - metapath_dict: Dictionary mapping entity pairs to their metapaths
    - sample_to_weight: Dictionary mapping instances to their weights (optional)
    - algo_params: Dictionary of all framework parameters
    - sample_name: Name of the analysis sample (for caching)
    - update_cache: Boolean flag to update the cache
    - pproc: Parallel processing context
    '''

    metapath_dict_positive = {key: metapath_dict[key] for key in training_positives}

    # Pattern mining from positive instances
    pattern_to_pos_matches, t1 = FrequentMetapathSetMiner(algo_params).run(metapath_dict_positive, sample_to_weight, sample_name, pproc=pproc, update_cache=update_cache)
    pattern_to_pos_matches, t2 = MetapathPatternUnificationMiner(algo_params).run(pattern_to_pos_matches, metapath_dict_positive, sample_to_weight, sample_name, pproc=pproc, update_cache=update_cache)
    pattern_to_pos_matches, t3 = RulePruner(algo_params).prune_non_closed_itemsets(pattern_to_pos_matches)
    pattern_to_pos_matches, t4 = PathThresholdOptimiser(algo_params).run(pattern_to_pos_matches, metapath_dict_positive, sample_to_weight, sample_name, pproc=pproc, update_cache=update_cache)

    # Generating the set of candidate rules
    start = time.process_time()
    positive_matches_to_rule_ids = {}
    for positive_match in training_positives:
        positive_matches_to_rule_ids[positive_match] = set()
    rule_list = []
    rule_id = 1
    for pattern, pos_matches in sorted(pattern_to_pos_matches.items(), key=lambda x: x[0]):
        rule = Rule(rule_id, pattern, 1, pos_matches)
        rule_list.append(rule)
        for pos_match in pos_matches:
            positive_matches_to_rule_ids[pos_match].add(rule_id)
        rule_id += 1

    elapsed_time = t1+t2+t3+t4 + (time.process_time() - start)

    return rule_list, positive_matches_to_rule_ids, elapsed_time


def apply_and_prune_rules(rule_list, positive_matches_to_rule_ids, training_negatives, metapath_dict, sample_to_weight, algo_params, sample_name, update_cache=False, pproc=None):
    '''
    Utils method to apply and prune candidate metapath-based rules for the given training data.
    Parameters:
    - rule_list: List of candidate rules
    - positive_matches_to_rule_ids: Dictionary mapping positive instances to their rules
    - training_negatives: List of negative training instances (entity pairs)
    - metapath_dict: Dictionary mapping entity pairs to their metapaths
    - sample_to_weight: Dictionary mapping instances to their weights (optional)
    - algo_params: Dictionary of all framework parameters
    - sample_name: Name of the analysis sample (for caching)
    - update_cache: Boolean flag to update the cache
    - pproc: Parallel processing context
    Returns:
    - valid_rules: List of valid rules
    '''
    metapath_dict_negative = {key: metapath_dict[key] for key in training_negatives}
    negative_matches_to_rule_ids, t1 = MetapathRuleMatcher(algo_params).run(rule_list, metapath_dict_negative, sample_name, pproc=pproc, update_cache=update_cache)
    valid_rules, t2 = RulePruner(algo_params).prune_and_get_rules(rule_list, positive_matches_to_rule_ids, negative_matches_to_rule_ids, sample_to_weight)
    elapsed_time = t1+t2
    return valid_rules, elapsed_time



