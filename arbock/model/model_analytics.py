from ..model.decision_set_classifier import DecisionSetClassifier
from ..config.paths import default_paths
from ..rule_mining.rule_querying import MetapathRuleMatcher
from collections import OrderedDict
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from pathlib import Path
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)

class Analytics:
    '''
    Analytics enabling to record the results of a model training / testing.
    '''

    def __init__(self, output_folder, file_prefix, analysis_name):
        if output_folder is None:
            output_folder = default_paths.model_analytics_folder
        base_file_name = f"{output_folder}/{file_prefix}_{analysis_name}"
        i = 0
        while os.path.exists(f"{base_file_name}_{i}.tsv") and Path(f"{base_file_name}_{i}.tsv").stat().st_size != 0:
            i += 1
        self.output_file = f"{base_file_name}_{i}.tsv"
        open(self.output_file, 'a').close()
        logger.info(f"Will write analytics in: {self.output_file}")
        self.new_file = True
        self.analysis_name = analysis_name

    def write_analytics(self, data):
        pd.DataFrame(data).to_csv(self.output_file, sep="\t", index=False, header=self.new_file, mode='a')
        self.new_file = False

    def add_analytics(self, dictionary, key, value):
        dictionary.setdefault(key, []).append(value)


class TrainingAnalytics(Analytics):
    '''
    Analytics enabling to record the results of a model training.
    '''

    def __init__(self, output_folder, analysis_name):
        super().__init__(output_folder, "model_train_analysis", analysis_name)

    def _get_estimator(self, wrapped_estimator):
        if isinstance(wrapped_estimator, Pipeline):
            return [m for c,m in wrapped_estimator.steps if c == "classifier"][0]
        return wrapped_estimator

    def extract_model_analytics(self, model, algo_params, index=None):
        '''
        Extracts the analytics of a model.
        :param model: the model to extract the analytics from.
        :param algo_params: the parameters of the algorithm.
        :param index: the index of the model in the ensemble.
        '''
        data = OrderedDict()
        analysis_name = self.analysis_name + (f"_{index}" if index is not None else "")
        if isinstance(model, DecisionSetClassifier):
            models = [model]
        else:
            models = [self._get_estimator(e) for e in model.estimators_]
        for submodel in models:
            self.add_analytics(data, 'sample_name', analysis_name)
            for param, param_value in algo_params.items():
                if isinstance(param_value, set) or isinstance(param_value, list):
                    param_value = ";".join(sorted(param_value))
                self.add_analytics(data, param, param_value)
            self.add_analytics(data, 'training_positive_coverage', len(submodel.get_positive_coverage()))
            self.add_analytics(data, 'training_negative_coverage', len(submodel.get_negative_coverage()))
            self.add_analytics(data, 'training_number_of_rules', len(submodel.get_rules()))
        super().write_analytics(data)


class PerformanceAnalytics(Analytics):

    def __init__(self, output_folder, analysis_name):
        super().__init__(output_folder, "model_performances", analysis_name)

    def extract_performances(self, performances, algo_params, index=None):
        '''
        Extracts the performances of a model.
        :param performances: the performances of the model.
        :param algo_params: the parameters of the algorithm.
        :param index: the index of the model in the ensemble.
        '''
        performance_analytics = OrderedDict()
        analysis_name = self.analysis_name + (f"_{index}" if index is not None else "")
        self.add_analytics(performance_analytics, 'sample_name', analysis_name)
        for param, param_value in algo_params.items():
            if isinstance(param_value, set) or isinstance(param_value, list):
                param_value = ";".join(sorted(param_value))
            self.add_analytics(performance_analytics, param, param_value)
        for measure, value in performances.items():
            self.add_analytics(performance_analytics, measure, value)
        super().write_analytics(performance_analytics)


class PredictionAnalytics(Analytics):
    '''
    Analytics enabling to record the results of a model prediction.
    '''

    def __init__(self, output_folder, analysis_name):
        super().__init__(output_folder, "model_predictions", analysis_name)

    def extract_predictions(self, sample_indices, sample_weights, predictions, y_test, algo_params, index=None):
        '''
        Extracts the predictions of a model.
        :param sample_indices: the indices of the samples.
        :param sample_weights: the weights of the samples.
        :param predictions: the predictions of the model.
        :param y_test: the true labels of the samples.
        :param algo_params: the parameters of the algorithm.
        :param index: the index of the model in the ensemble.
        '''
        prediction_analytics = OrderedDict()
        analysis_name = self.analysis_name + (f"_{index}" if index is not None else "")
        self.add_analytics(prediction_analytics, 'sample_name', analysis_name)
        for param, param_value in algo_params.items():
            if isinstance(param_value, set) or isinstance(param_value, list):
                param_value = ";".join(sorted(param_value))
            self.add_analytics(prediction_analytics, param, param_value)
        self.add_analytics(prediction_analytics, "sample_indices", list(sample_indices))
        self.add_analytics(prediction_analytics, "sample_weights", list(sample_weights))
        self.add_analytics(prediction_analytics, "predictions", list(predictions))
        self.add_analytics(prediction_analytics, "y", list(y_test))
        super().write_analytics(prediction_analytics)


class ExplanationAnalytics(Analytics):
    '''
    Analytics enabling to record the results of a model explanation.
    '''

    def __init__(self, output_folder, analysis_name):
        super().__init__(output_folder, "model_explanations", analysis_name)

    def extract_explanations(self, sample_indices, sample_weights, predict_explanations, algo_params, index=None):
        '''
        Extracts the explanations of a model.
        :param sample_indices: the indices of the samples.
        :param sample_weights: the weights of the samples.
        :param predict_explanations: the explanations of the model.
        :param algo_params: the parameters of the algorithm.
        :param index: the index of the model in the ensemble.
        '''
        explanation_analytics = OrderedDict()
        analysis_name = self.analysis_name + (f"_{index}" if index is not None else "")
        self.add_analytics(explanation_analytics, 'sample_name', analysis_name)
        for param, param_value in algo_params.items():
            if isinstance(param_value, set) or isinstance(param_value, list):
                param_value = ";".join(sorted(param_value))
            self.add_analytics(explanation_analytics, param, param_value)
        self.add_analytics(explanation_analytics, "sample_indices", list(sample_indices))
        self.add_analytics(explanation_analytics, "sample_weights", list(sample_weights))
        all_explanation_rules = []
        all_explanation_rule_count = []
        for predict_explanation in predict_explanations:
            rules = []
            rule_to_score = predict_explanation
            if rule_to_score:
                for rule, score in rule_to_score.items():
                    rules.append(rule.antecedent)
            all_explanation_rules.append(rules)
            all_explanation_rule_count.append(len(rules))
        self.add_analytics(explanation_analytics, "rule_counts", all_explanation_rule_count)
        self.add_analytics(explanation_analytics, "rules", all_explanation_rules)
        super().write_analytics(explanation_analytics)


class SubgraphExplanationAnalytics(Analytics):
    '''
    Analytics enabling to record the results of a model subgraph explanation.
    '''

    def __init__(self, output_folder, analysis_name):
        super().__init__(output_folder, "model_subgraph_explanations", analysis_name)

    def extract_subgraph_explanations(self, sample_indices, sample_weights, predict_explanations, metapath_dict, kg, algo_params, index=None):
        '''
        Extracts the explanations of a model.
        :param sample_indices: the indices of the samples.
        :param sample_weights: the weights of the samples.
        :param predict_explanations: the explanations of the model.
        :param metapath_dict: the metapath dictionary.
        :param kg: the knowledge graph.
        :param algo_params: the parameters of the algorithm.
        :param index: the index of the model in the ensemble.
        '''
        explanation_analytics = OrderedDict()
        analysis_name = self.analysis_name + (f"_{index}" if index is not None else "")
        self.add_analytics(explanation_analytics, 'sample_name', analysis_name)
        for param, param_value in algo_params.items():
            if isinstance(param_value, set) or isinstance(param_value, list):
                param_value = ";".join(sorted(param_value))
            self.add_analytics(explanation_analytics, param, param_value)
        self.add_analytics(explanation_analytics, "sample_indices", list(sample_indices))
        self.add_analytics(explanation_analytics, "sample_weights", list(sample_weights))
        all_explanation_path_counts = []
        all_explanation_entity_counts = []
        for gene_ensg_pair, predict_explanation in zip(sample_indices, predict_explanations):
            rules_path_counts = []
            rules_entity_counts = []
            rule_to_score = predict_explanation
            if rule_to_score:
                for rule, score in rule_to_score.items():
                    nodes, edges, direction_to_edge_paths = MetapathRuleMatcher(algo_params).retrieve_subgraph_and_paths(rule, gene_ensg_pair,
                                                                                        metapath_dict[gene_ensg_pair], kg)
                    rules_path_counts.append(max([len(p) for d,p in direction_to_edge_paths.items()]))
                    rules_entity_counts.append(len(nodes))
            all_explanation_path_counts.append(rules_path_counts)
            all_explanation_entity_counts.append(rules_entity_counts)
        self.add_analytics(explanation_analytics, "rule_path_counts", all_explanation_path_counts)
        self.add_analytics(explanation_analytics, "rule_entity_counts", all_explanation_entity_counts)
        super().write_analytics(explanation_analytics)


class TrainingTimeAnalysis(Analytics):
    '''
    Analytics enabling to record the results of a model training time analysis.
    '''

    def __init__(self, output_folder, analysis_name):
        super().__init__(output_folder, "model_training_times", analysis_name)

    def extract_running_times(self, mining_time, training_time, algo_params, index=None):
        '''
        Extracts the running times of a model.
        :param mining_time: the mining time.
        :param training_time: the training time.
        :param algo_params: the parameters of the algorithm.
        :param index: the index of the model in the ensemble.
        '''
        explanation_analytics = OrderedDict()
        analysis_name = self.analysis_name + (f"_{index}" if index is not None else "")
        self.add_analytics(explanation_analytics, 'sample_name', analysis_name)
        for param, param_value in algo_params.items():
            if isinstance(param_value, set) or isinstance(param_value, list):
                param_value = ";".join(sorted(param_value))
            self.add_analytics(explanation_analytics, param, param_value)
        self.add_analytics(explanation_analytics, "mining_time", mining_time)
        self.add_analytics(explanation_analytics, "training_time", training_time)
        super().write_analytics(explanation_analytics)