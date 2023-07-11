from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


class DecisionSetClassifier(BaseEstimator, ClassifierMixin):
    """
   This class represents a Decision Set Classifier, an interpretable binary classifier that considers a small set of rules to perform classification tasks.
   Applying this classifier consists in evaluating instance data against each selected rule in the model.
   If any rule matches, the predicted probability is the max probability from all rule matching.
   If none of the rule matches, then a default probability (based on uncovered instances during training) is returned.
   Additionally, the matching rules are returned as explanations for each instance.

   The training of the model involves searching for a small subset of rules maximizing the coverage of positive instances while minimizing the coverage of negative instances.
   This algorithm is based on the greedy weighted set cover approach.
   References:
   - Decision set model: https://doi.org/10.1145/2939672.2939874
   - RUDIK method: https://doi.org/10.14778/3229863.3236231
   - Greedy Weighted Set Cover explanation: http://cs.williams.edu/~shikha/teaching/spring20/cs256/lectures/Lecture31.pdf

   @author Alexandre Renaux [Universite Libre de Bruxelles / Vrije Universiteit Brussel]
   """

    def __init__(self, candidate_rules, rule_matcher, alpha=0.5, cpu_cores=0):
        """
        Initializes the classifier with candidate rules, a rule matcher and an alpha value.

        Parameters:
        - candidate_rules: Set of rules that will be used for classification
        - rule_matcher: a `arbock.rule_mining.rule_querying.RuleMatcher` instance
        - alpha: The trade-off parameter between true positive rate (TPR) and false positive rate (FPR)
        - cpu_cores: Number of CPU cores to use for training (0 means all available cores)
        """
        self.candidate_rules = candidate_rules
        self.rule_matcher = rule_matcher
        self.alpha = alpha
        self.cpu_cores = cpu_cores
        self._best_rules = None
        self._instance_to_weight = None
        self._label_to_instances = None
        self._label_to_weightsum = None
        self._imbalance_ratio = 1
        self._training_coverages = None
        self._rule_matching_probabilities = None
        self._rule_non_matching_probability = None
        self._cached_explanations = {}
        self.classes_ = np.array([0,1])

    def fit(self, X, y=None, sample_weight=None):
        """
        Fits the classifier on the given training data and labels.

        Parameters:
        - X: Training instances
        - y: Labels for the training instances
        - sample_to_weight: Weights for the training instances
        """
        sampled_indexes = X[:,0]
        sample_weight = [1] * len(X) if sample_weight is None else sample_weight
        self._instance_to_weight = {x:w for x, w in zip(sampled_indexes, sample_weight)}
        self._label_to_instances = self._get_label_to_training_samples(sampled_indexes, y)
        self._label_to_weightsum = self._get_label_to_weightsum()
        logger.info(f"Decision Set fitting ... [alpha={self.alpha}] -- {len(self.candidate_rules)} candidate rules, {len(self._label_to_instances[1])} positives, {len(self._label_to_instances[0])} negatives")
        self._best_rules = self._weighted_set_cover(alpha=self.alpha)
        self._imbalance_ratio = len(self._label_to_instances[1]) / len(self._label_to_instances[0])
        self._training_coverages = self._get_training_best_rule_coverages()
        self._rule_matching_probabilities = self._get_rule_probabilities()
        self._rule_non_matching_probability = self._get_non_matching_probability(sampled_indexes, y)
        logger.info(f"FITTING DONE -- {len(self._best_rules)} / {len(self.candidate_rules)} rules selected, covering {len(self._training_coverages[1])} positives (out of {len(self._label_to_instances[1])}) / {len(self._training_coverages[0])} negatives (out of {len(self._label_to_instances[0])}). (Non-matching positive prob = {self._rule_non_matching_probability})")

    def predict_proba(self, X):
        """
        Predicts class probabilities for the given test instances.

        Parameters:
        - X: Test instances: We expect X to be an array where each row is: sample_idx ; metapath dict

        Returns:
        - A numpy array of predicted probabilities
        """
        probas = []
        for instance in X:
            sample_idx, metapath_dict = instance
            proba, explanation = self._predict_proba_and_explain(metapath_dict)
            probas.append(proba)
            self._cached_explanations[sample_idx] = explanation
        return np.array(probas)

    def predict(self, X):
        """
        Predicts classes for the given test instances.

        Parameters:
        - X: Test instances: We expect X to be an array where each row is: sample_idx ; metapath dict

        Returns:
        - A numpy array of predicted labels
        """
        probas = self.predict_proba(X)
        return np.array([0,1]).take(np.apply_along_axis(lambda p: np.argmax(p), 1, probas), axis=0)

    def get_explanations(self, X):
        """
        Retrieves explanations for the predictions on the given test instances.

        Parameters:
        - X: Test instances: We expect X to be an array where each row is: sample_idx ; metapath dict

        Returns:
        - A list of predictive explanations (matching rules) for each instance
        """
        all_explanations = []
        for instance in X:
            sample_idx, metapath_dict = instance
            if sample_idx in self._cached_explanations:
                all_explanations.append(self._cached_explanations[sample_idx])
            else:
                proba, explanation = self._predict_proba_and_explain(metapath_dict)
                all_explanations.append(explanation)
        return all_explanations

    def predict_and_explain(self, sample_ids, metapath_dict):
        """
        Performs prediction and explanation together for given instances.

        Parameters:
        - sample_ids: Gene pair instances
        - metapath_dict: Dictionary of metapaths for each sample id

        Returns:
        - The list of ordered samples
        - Predicted probabilities in the same order
        - Predictive explanations (matched rules) in the same order
        """
        X_test_list = []
        for sample_id in sample_ids:
            X_test_list.append([sample_id, metapath_dict[sample_id]])
        X_test = np.array(X_test_list, dtype=object)

        ordered_samples = [x[0] for x in X_test_list]
        predict_probas = self.predict_proba(X_test)
        explanations = self.get_explanations(X_test)

        return ordered_samples, predict_probas, explanations

    def get_rules(self):
        """
        Retrieves the best rules selected during model fitting.

        Returns:
        - The rules selected during training
        """
        if not self._best_rules:
            logger.error("Calling get_rules() on unfitted model")
        return self._best_rules

    def get_positive_coverage(self):
        """
        Retrieves the coverage of positive instances by the selected rules.

        Returns:
        - The coverage of positive instances by the selected rules
        """
        if not self._training_coverages:
            logger.error("Calling get_positive_coverage() on unfitted model")
            return None
        return self._training_coverages[1]

    def get_negative_coverage(self):
        """
        Retrieves the coverage of negative instances by the selected rules.

        Returns:
        - The coverage of negative instances by the selected rules
        """
        if not self._training_coverages:
            logger.error("Calling get_positive_coverage() on unfitted model")
            return None
        return self._training_coverages[0]

    def _get_label_to_training_samples(self, X, y):
        """
        Groups training instances by their labels.

        Parameters:
        - X: Training instances
        - y: Labels for the training instances

        Returns:
        - A dictionary grouping set of instances per label
        """
        label_to_instances = defaultdict(set)
        for instance, label in zip(X,y):
            label_to_instances[label].add(instance)
        return label_to_instances

    def _get_label_to_weightsum(self):
        """
        Computes the total instance weight sum corresponding to each class label.

        Returns:
        - A dictionary associating the sum of instance weights for each label
        """
        label_to_weightsum = {}
        for label in self._label_to_instances:
            label_to_weightsum[label] = sum([self._instance_to_weight[i] for i in self._label_to_instances[label]])
        return label_to_weightsum

    def _get_training_best_rule_coverages(self):
        """
        Computes the coverage of each class by the best rules selected during fitting.

        Returns:
        - A dictionary of training samples covered for each label
        """
        positive_coverage = set()
        negative_coverage = set()
        for rule in self._best_rules:
            positive_coverage.update(rule.positive_matches)
            negative_coverage.update(rule.negative_matches)
        positive_coverage = positive_coverage.intersection(self._instance_to_weight.keys())
        negative_coverage = negative_coverage.intersection(self._instance_to_weight.keys())
        return {1:positive_coverage, 0: negative_coverage}

    def _get_rule_probabilities(self):
        """
        Computes the probabilities for each rule.

        Returns:
        - A dictionary with the associated probability for each rule
        """
        rule_to_probabilities = {}
        for rule in self._best_rules:
            rule_to_probabilities[rule] = self._get_weighted_rule_confidence(rule)
        return rule_to_probabilities

    def _get_non_matching_probability(self, X, y):
        """
        Computes the default positive probability based on training instances not matching any rule.

        Parameters:
        - X: Training instances
        - y: Labels for the training instances

        Returns:
        - The default probability based on training instances not matching any rules
        """
        label_to_non_match_weights = defaultdict(list)
        for i, label in zip(X,y):
            if i not in self._training_coverages[0] and i not in self._training_coverages[1]:
                label_to_non_match_weights[label].append(self._instance_to_weight[i])

        weighted_pos = sum(label_to_non_match_weights[1])
        weighted_neg = self._imbalance_ratio * sum(label_to_non_match_weights[0])

        weighted_precision = weighted_pos / (weighted_pos + weighted_neg) if (weighted_pos + weighted_neg) != 0 else 0
        return weighted_precision

    def _weighted_set_cover(self, alpha):
        if self.cpu_cores != 1 and len(self.candidate_rules) > 1000:
            n_cores = cpu_count() + self.cpu_cores if self.cpu_cores <= 0 else self.cpu_cores
            partitions_count = (cpu_count() * 4)
            partitions_count = 1 if partitions_count < 1 or partitions_count > len(self.candidate_rules) else partitions_count
            logger.info(f"Using {n_cores} cores for parallelization")
            with Pool(n_cores) as pool:
                chunksize = len(self.candidate_rules) // partitions_count
                return self._select_best_rules_to_add(alpha, pool, chunksize)
        else:
            return self._select_best_rules_to_add(alpha, None, None)

    def _select_best_rules_to_add(self, alpha, pool, chunksize):
        """
        Selects the best rules to add to the decision set following the Greedy Weighted Set Cover algorithm

        Parameters:
        - alpha: The trade-off parameter between true positive rate (TPR) and false positive rate (FPR)
        - pool: The multiprocessing pool to use for parallelization
        - chunksize: The chunksize to use for parallelization

        Returns:
        - The set of optimal rule set selected by the Greedy Weighted Set Cover algorithm
        """

        optimal_rule_set = set()

        while True:
            selected_rule = self._select_next_best_rule(optimal_rule_set, alpha, pool, chunksize)
            if selected_rule is None:
                break
            else:
                optimal_rule_set.add(selected_rule)
                logger.debug(f"Adding rule {selected_rule} to set")

        return optimal_rule_set

    def _select_next_best_rule(self, selected_rule_set, alpha, pool, chunksize):
        """
        Selects the next best rule to add to the decision set.

        Parameters:
        - selected_rule_set: Current set of selected rules
        - alpha: The trade-off parameter between true positive rate (TPR) and false positive rate (FPR)
        - pool: The pool of processes to use for parallelization
        - chunksize: The size of the chunks to use for parallelization

        Returns:
        - The best rule to add to the selected rule set or None if the exit condition is met (i.e the rule set is optimal)
        """
        candidate_rules = set(self.candidate_rules) - selected_rule_set
        if len(candidate_rules) == 0:
            return None

        weight_without_rule = DecisionSetClassifier._get_rule_set_weight(selected_rule_set, self._instance_to_weight, self._label_to_weightsum, alpha)
        logger.debug(f"rule_set (size={len(selected_rule_set)}) weight without rule: {weight_without_rule}")

        rule_marginal_weight_func = partial(DecisionSetClassifier._get_rule_marginal_weight,
                                            rule_set=selected_rule_set,
                                            weight_without_rule=weight_without_rule,
                                            instance_to_weight=self._instance_to_weight,
                                            label_to_weightsum=self._label_to_weightsum,
                                            alpha=alpha)

        if pool is not None:
            r_to_w = {rule: weight for rule, weight in pool.imap_unordered(rule_marginal_weight_func, candidate_rules, chunksize)}
        else:
            r_to_w = {r:rule_marginal_weight_func(r)[1] for r in candidate_rules}

        min_marginal_weight = min(r_to_w.values())

        logger.info(f"Comparing rule_set (size={len(selected_rule_set)}) to candidates (size={len(candidate_rules)}): min marginal weight = {min_marginal_weight}")
        if min_marginal_weight >= 0:
            return None
        selected_rules = [r for r, w in r_to_w.items() if w == min_marginal_weight]
        if len(selected_rules) == 1:
            return selected_rules[0]
        else:
            # Breaking ties
            return max(selected_rules, key=lambda r: (self._get_weighted_rule_confidence(r), r.id))

    @staticmethod
    def _get_rule_marginal_weight(rule, rule_set, weight_without_rule, instance_to_weight, label_to_weightsum, alpha):
        """
        Computes the marginal weight of a rule (i.e the difference between the set incl. the rule vs the set excl. the rule)

        Parameters:
        - rule: Rule for which the marginal weight is computed
        - rule_set: Current set of rules
        - weight_without_rule: Weight of the current set of rules
        - instance_to_weight: Weight for each instance
        - label_to_training_samples: Training instances grouped by their labels

        Returns:
        - The rule itself
        - The associated marginal weight for the rule (float)
        """
        weight_with_rule = DecisionSetClassifier._get_rule_set_weight(rule_set.union([rule]), instance_to_weight, label_to_weightsum, alpha)

        logger.debug(f"rule_set (size={len(rule_set)}) weight with rule ({rule}): {weight_with_rule}")

        rule_marginal_weight = weight_with_rule - weight_without_rule
        return rule, rule_marginal_weight

    @staticmethod
    def _get_rule_set_weight(rule_set, instance_to_weight, label_to_weightsum, alpha):
        """
        Computes the weight of a rule set based on its positive and negative coverage.

        Parameters:
        - rule_set: Set of rules in the rule set
        - instance_to_weight: Weight for each instance
        - label_to_weightsum: Total weight for each label
        - alpha: Trade-off parameter for combining true positive rate and false positive rate

        Returns:
        - set_weight: Weight of the rule set
        """

        if len(rule_set) == 0:
            return 1.0

        overall_positive_coverage = set()
        overall_negative_coverage = set()

        for rule in rule_set:
            overall_positive_coverage.update(rule.positive_matches)
            overall_negative_coverage.update(rule.negative_matches)

        weighted_matching_pos = sum([instance_to_weight.get(i,0) for i in overall_positive_coverage])
        weighted_matching_neg = sum([instance_to_weight.get(i,0) for i in overall_negative_coverage])

        label_to_weightsum_1 = label_to_weightsum[1]
        label_to_weightsum_0 = label_to_weightsum[0]

        tpr = weighted_matching_pos / label_to_weightsum_1 if label_to_weightsum_1 > 0 else 0
        fpr = weighted_matching_neg / label_to_weightsum_0 if label_to_weightsum_0 > 0 else 0

        set_weight = (alpha * (1 - tpr)) + ((1-alpha) * fpr)

        return set_weight

    def _get_weighted_rule_positive_coverage(self, rule):
        """
        Computes the weighted positive coverage of a rule.

        Parameters:
        - rule: Rule for which the weighted positive coverage is computed

        Returns:
        - weighted_positive_coverage: Weighted positive coverage of the rule
        """
        return sum([self._instance_to_weight[i] for i in rule.positive_matches if i in self._instance_to_weight])

    def _get_weighted_rule_negative_coverage(self, rule):
        """
        Computes the weighted negative coverage of a rule.

        Parameters:
        - rule: Rule for which the weighted negative coverage is computed

        Returns:
        - weighted_negative_coverage: Weighted negative coverage of the rule
        """
        return sum([self._instance_to_weight[i] for i in rule.negative_matches if i in self._instance_to_weight])

    def _get_weighted_rule_confidence(self, rule):
        """
        Computes the weighted confidence (precision) of a rule.

        Parameters:
        - rule: Rule for which the weighted confidence is computed

        Returns:
        - weighted_precision: Weighted confidence (precision) of the rule
        """
        weighted_pos = self._get_weighted_rule_positive_coverage(rule)
        weighted_neg = self._imbalance_ratio * self._get_weighted_rule_negative_coverage(rule)

        weighted_precision = weighted_pos / (weighted_pos + weighted_neg) if (weighted_pos + weighted_neg) != 0 else 0

        return weighted_precision

    def _predict_proba_and_explain(self, sample_data):
        """
        Predicts the class probabilities for a given sample based on the rule set.

        Parameters:
        - sample_data: Dictionary of data associated to a sample, to be matched against rules.

        Returns:
        - class_probas: Predicted class probabilities for the sample
        - matching_rules: Rules that matched the sample and their corresponding probabilities (for explanation purposes)
        """
        matching_rules = {}
        for rule in self._best_rules:
            if self.rule_matcher.evaluate_rule(rule, sample_data):
                matching_rules[rule] = self._rule_matching_probabilities[rule]

        if matching_rules:
            top_matching_rule = sorted(matching_rules.items(), key=lambda x: x[1], reverse=True)[0]
            selected_rule, proba = top_matching_rule
            class_probas = np.array([1-proba, proba])
            return class_probas, matching_rules
        else:
            probas = [1 - self._rule_non_matching_probability, self._rule_non_matching_probability]
            return np.array(probas), None

    def persist(self, output):
        """
        Persists the trained model to a pickle file.

        Parameters:
        - output: File path to save the model
        """
        with open(output, 'wb') as f:
            logger.debug(f"Dumping model to {output} via pickle")
            pickle.dump(self, f, protocol=4)

    @staticmethod
    def instanciate(model_pkl):
        """
        Instantiates a pretrained model from a pickle file.

        Parameters:
        - model_pkl: File path of the pretrained model

        Returns:
        - Pretrained model
        """
        with open(model_pkl, 'rb') as f:
            logger.debug(f"Using pretrained model from {model_pkl}.")
            return pickle.load(f)
