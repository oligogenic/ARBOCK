from collections import defaultdict

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"


def default_to_regular(d):
    '''
    Recursively convert defaultdicts to regular dicts.
    '''
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def max_scaling_dict_vals(val_dict, max_val=None):
    '''
    Maximum scale the values of a dictionary to the range [0, 1].
    '''
    scaled_dict = {}
    max_val = max(val_dict.values()) if max_val is None else max_val
    for key, val in val_dict.items():
        scaled_dict[key] = val / max_val
    return scaled_dict


def minmax_scaling_dict_vals(val_dict):
    '''
    Min-max scale the values of a dictionary to the range [0, 1].
    '''
    scaled_dict = {}
    max_val = max(val_dict.values())
    min_val = min(val_dict.values())
    for key, val in val_dict.items():
        scaled_dict[key] = (val - min_val) / (max_val - min_val)
    return scaled_dict