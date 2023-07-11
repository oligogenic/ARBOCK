import json
import os
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)

_local_spark_config = f"{os.path.dirname(__file__)}/../../config/spark_config_local.json"
_yarn_spark_config = f"{os.path.dirname(__file__)}/../../config/spark_config_yarn.json"


class SparkConfig:
    '''
    Spark configuration for ARBOCK.
    '''

    driver_location = None
    spark_conf_dict = None

    def __init__(self, master="local"):
        yarn_config_file = _local_spark_config if master == 'local' else _yarn_spark_config
        if os.path.isfile(yarn_config_file):
            logger.info(f"Setting default paths with {yarn_config_file}.")
            with open(yarn_config_file, "r") as f:
                json_dict = json.load(f)
                for key, v in json_dict.items():
                    setattr(self, key, v)
        for var, value in vars(self).items():
            if value is None:
                print(f"{var} should be set in {yarn_config_file}")
                exit(1)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.__dict__)
