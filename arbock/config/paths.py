import json
import os
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)

_path_file = f"{os.path.dirname(__file__)}/../../config/paths.json"


class DefaultPaths:
    '''
    Default paths for ARBOCK.
    '''

    def __init__(self):
        self.models_folder = f"{os.path.dirname(__file__)}/../../models/"
        self.model_analytics_folder = f"{self.models_folder}/analytics/"
        self.cache_folder = f"{os.path.dirname(__file__)}/../../caches/"
        self.kg_graphml = f"{os.path.dirname(__file__)}/../../datasets/bock.graphml"
        self.neutral_pairs_path = f"{os.path.dirname(__file__)}/../../datasets/neutrals_1KGP_100x.tsv"
        self.update_from_json()

    def update_from_json(self):
        '''
        Update default paths from a JSON file.
        '''
        if os.path.isfile(_path_file):
            logger.info(f"Setting default paths with {_path_file}.")
            with open(_path_file, "r") as f:
                json_dict = json.load(f)
                for key, v in json_dict.items():
                    setattr(self, key, v)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.__dict__)


default_paths = DefaultPaths()