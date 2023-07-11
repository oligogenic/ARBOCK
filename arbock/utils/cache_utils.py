from ..config import paths
import pickledb
import pickle
import logging
import os
import gc

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"


logger = logging.getLogger(__name__)


class Cache:

    STORAGE_ROOT = paths.default_paths.cache_folder + "/"

    def __init__(self, name, update=False, single_file=False, storage_root=STORAGE_ROOT):
        self.storage_path = os.path.abspath(storage_root + name)
        if not single_file:
            self.storage = pickledb.load(self.storage_path, True)
        self.update = update
        self.single_file = single_file

    def dump(self, data):
        self.store(None, data)

    def store(self, key, data):
        if self.single_file:
            gc.disable()
            with open(self.storage_path, 'wb') as f:
                logger.info(f"Caching data at: {self.storage_path}")
                pickle.dump(data, f, protocol=4)
            gc.enable()
        else:
            logger.info(f"Caching data at: {self.storage_path} [key= {key}]")
            self.storage.set(key, data)

    def get(self, key=None):
        if self.single_file:
            gc.disable()
            with open(self.storage_path, 'rb') as f:
                logger.info(f"Getting data from cache at: {self.storage_path}")
                r = pickle.load(f)
            gc.enable()
            return r
        else:
            logger.info(f"Getting data from cache at: {self.storage_path} [key= {key}]")
            return self.storage.get(key)

    def get_or_store(self, key, fetch_data_func):
        if self.single_file:
            if os.path.exists(self.storage_path) and os.path.getsize(self.storage_path) > 0 and not self.update:
                return self.get()
            else:
                data = fetch_data_func(key)
                self.dump(data)
                return data
        else:
            local_data = self.get(key)
            if local_data != False and not self.update:
                return local_data
            else:
                data = fetch_data_func(key)
                self.store(key, data)
                return data

    @staticmethod
    def generate_cache_file_name(process_name, sample_name, algo_params, *param_names):
        output_name = f"{sample_name}_{process_name}"
        for param_name in param_names:
            val = algo_params[param_name]
            if isinstance(algo_params[param_name], set) or isinstance(algo_params[param_name], list):
                val = "-".join(sorted(val))
            output_name += f"_{param_name}_{val}"
        return output_name