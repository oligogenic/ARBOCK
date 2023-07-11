from .rule import TrainedRule
from collections import defaultdict
from scipy.stats import fisher_exact
import numpy as np
import time
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


class RulePruner:
    '''
    The class that implements the rule pruning algorithms.
    '''

    def __init__(self, algo_params):
        self.algo_params = algo_params

    def prune_and_get_rules(self, all_rules, positive_match_to_rule_ids, negative_match_to_rule_ids, sample_to_weight):
        '''
        Prune the rules according to the algorithm parameters and return the pruned rules.
        :param all_rules: the rules to prune
        :param positive_match_to_rule_ids: a dictionary mapping positive matches to the rules that cover them
        :param negative_match_to_rule_ids: a dictionary mapping negative matches to the rules that cover them
        :param sample_to_weight: a dictionary mapping samples to their weights
        :return: the pruned rules
        '''
        start = time.process_time()
        max_fpr = self.algo_params["max_fpr"]

        valid_rule_ids, rule_to_pos_matches, rule_to_neg_matches = RulePruner.local_get_low_fpr_rule_to_matches(all_rules, positive_match_to_rule_ids, negative_match_to_rule_ids, sample_to_weight, max_fpr)

        valid_rules = set()
        for rule_id in valid_rule_ids:
            rule = all_rules[rule_id]
            trained_rule = TrainedRule(rule, rule_to_neg_matches[rule_id])
            valid_rules.add(trained_rule)
        elapsed_time = time.process_time() - start
        logger.info(f"{len(valid_rules)} rules after FPR filtering (elapsed time={elapsed_time}s)")
        return valid_rules, elapsed_time

    @staticmethod
    def prune_non_closed_itemsets(pattern_to_positive_matches):
        '''
        Prune the non-closed itemsets from the patterns.
        :param pattern_to_positive_matches: a dictionary mapping patterns to their positive matches
        :return: the pruned patterns
        '''
        start = time.process_time()
        if len(pattern_to_positive_matches) == 0:
            return {}, 0
        invalid_patterns = []
        level_to_pattern = defaultdict(dict)
        for pattern, matching_transactions in pattern_to_positive_matches.items():
            rule_level = len(pattern.metapaths) + (1 if pattern.unification else 0)
            level_to_pattern[rule_level][pattern] = matching_transactions

        for level in range(1, max(level_to_pattern.keys())):
            pattern_to_prune = level_to_pattern[level]
            superset_patterns = level_to_pattern[level+1]

            metapath_to_patterns = defaultdict(set)
            for pattern, matching_transactions in superset_patterns.items():
                for mp in pattern.metapaths:
                    metapath_to_patterns[mp].add(pattern)

            for pattern, matching_transactions in pattern_to_prune.items():
                matching_superset = metapath_to_patterns.get(pattern.metapaths[0], set())
                for pattern_mp in pattern.metapaths[1:]:
                    matching_superset.intersection(metapath_to_patterns.get(pattern_mp, set()))
                for matching_superset_pattern in matching_superset:
                    superset_matching_transactions = superset_patterns[matching_superset_pattern]
                    if len(matching_transactions) <= len(superset_matching_transactions):
                        invalid_patterns.append(pattern)
                        break
        closed_patterns = {r:m for r,m in pattern_to_positive_matches.items() if r not in invalid_patterns}
        elapsed_time = time.process_time() - start
        logger.info(f"Excluding {len(invalid_patterns)} non-closed patterns. Now {len(closed_patterns)} patterns. (elapsed time={elapsed_time}s)")
        return closed_patterns, elapsed_time

    @staticmethod
    def local_get_low_fpr_rule_to_matches(all_rules, positive_match_to_rule_ids, negative_match_to_rule_ids, sample_to_weight, maxfpr):
        """
        Local implementation of the rule filtering by FPR
        :param all_rules: the rules to filter
        :param positive_match_to_rule_ids: a dictionary mapping positive matches to the rules that cover them
        :param negative_match_to_rule_ids: a dictionary mapping negative matches to the rules that cover them
        :param sample_to_weight: a dictionary mapping samples to their weights
        :param maxfpr: the maximum false positive rate
        :return: the rules that have a FPR lower than maxfpr, the positive matches of these rules, the negative matches of these rules
        """
        rule_to_pos_matches = defaultdict(set)
        rule_to_neg_matches = defaultdict(set)
        total_neg_weight = sum([sample_to_weight[i] for i in negative_match_to_rule_ids])
        max_neg_weighted = maxfpr * total_neg_weight
        for negative_match, rule_ids in negative_match_to_rule_ids.items():
            for rule_id in rule_ids:
                rule_to_neg_matches[rule_id].add(negative_match)
        valid_rule_ids = set()
        for rule_id in range(len(all_rules)):
            negative_matches = rule_to_neg_matches.get(rule_id, [])
            neg_count_weighted = sum([sample_to_weight[i] for i in negative_matches])
            if neg_count_weighted <= max_neg_weighted:
                valid_rule_ids.add(rule_id)
        for positive_match, rule_ids in positive_match_to_rule_ids.items():
            for rule_id in rule_ids:
                if rule_id in valid_rule_ids:
                    rule_to_pos_matches[rule_id].add(positive_match)
        rule_to_neg_matches = {r:rule_to_neg_matches.get(r, set()) for r in valid_rule_ids}
        return valid_rule_ids, rule_to_pos_matches, rule_to_neg_matches

    @staticmethod
    def fisher_exact_pval(tp, fp, fn, tn):
        '''
        Compute the p-value of the Fisher exact test for the contingency table
        :param tp: the number of true positives
        :param fp: the number of false positives
        :param fn: the number of false negatives
        :param tn: the number of true negatives
        '''
        contingency_table = np.array([[tp, fp], [fn, tn]])
        odd, pval = fisher_exact(contingency_table, alternative="greater")
        print(odd)
        return pval

    @staticmethod
    def local_get_significant_rules(all_rules, positive_match_to_rule_ids, negative_match_to_rule_ids, sample_to_weight, max_pval):
        '''
        Local implementation of the rule filtering by p-value
        :param all_rules: the rules to filter
        :param positive_match_to_rule_ids: a dictionary mapping positive matches to the rules that cover them
        :param negative_match_to_rule_ids: a dictionary mapping negative matches to the rules that cover them
        :param sample_to_weight: a dictionary mapping samples to their weights
        :param max_pval: the maximum p-value
        :return: the rules that have a p-value lower than max_pval, the positive matches of these rules, the negative matches of these rules
        '''
        P = sum([sample_to_weight[s] for s in positive_match_to_rule_ids])
        N = sum([sample_to_weight[s] for s in negative_match_to_rule_ids])

        r_to_n = defaultdict(set)
        for negative_match, rule_ids in negative_match_to_rule_ids.items():
            for rule_id in rule_ids:
                r_to_n[rule_id].add(negative_match)

        for rule in all_rules:
            TP = sum([sample_to_weight[s] for s in rule.positive_matches])
            FP = sum([sample_to_weight[s] for s in r_to_n[rule.id]])
            FN = P - TP
            TN = N - FP
            pval = RulePruner.fisher_exact_pval(TP, FP, FN, TN)
            print(rule, r_to_n[rule.id], pval)
            if pval < max_pval:
                yield TrainedRule(rule, r_to_n[rule.id])


