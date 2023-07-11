import math
import numpy as np

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"


def get_absolute_minsup(samples, sample_to_weight, minsup_ratio):
    '''
    Get the absolute minimum support from a ratio.
    :param samples: the samples
    :param sample_to_weight: the weight of each sample
    :param minsup_ratio: the ratio
    :return: the absolute minimum support
    '''
    if sample_to_weight:
        minsup = sum([sample_to_weight[i] for i in samples]) * minsup_ratio
    else:
        minsup = math.floor(len(samples) * minsup_ratio)
    return minsup


def valid_support(matching_samples, min_support, sample_to_weight=None):
    '''
    Check if the support of a rule is valid.
    :param matching_samples: the samples matching the rule
    :param min_support: the minimum support
    :param sample_to_weight: the weight of each sample
    :return: True if the support is valid, False otherwise
    '''
    if sample_to_weight is None:
        return len(matching_samples) >= min_support
    else:
        return sum([sample_to_weight[i] for i in matching_samples]) >= min_support


def directs_metapath(metapath, direction):
    '''
    Get the metapath in the specified direction.
    :param metapath: the metapath
    :param direction: the direction in which the metapeath should be directed
    :return: the metapath in the specified direction
    '''
    edge_types, node_types, edge_direction = metapath
    return (edge_types[::direction], node_types[::direction], tuple(np.array(edge_direction) * direction))