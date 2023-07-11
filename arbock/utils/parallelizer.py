from ..config import spark_config
import zipfile
import os
import copy
import abc
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


class Parallelizer(metaclass=abc.ABCMeta):
    '''
    Abstract class for parallelization of tasks.
    '''

    @abc.abstractmethod
    def map_collect(self, method, list_elements, partitions_count=None, shared_variables_dict=None):
        pass

    @abc.abstractmethod
    def stop(self):
        pass


class SparkParallelizer(Parallelizer):
    '''
    A parallelizer that uses the pyspark library (for locally or yarn distributed tasks).
    Can be used as a context manager with the 'with' statement.
    @author Alexandre Renaux [Universite Libre de Brussel / Vrije Universiteit Brussel]
    '''

    def __init__(self, master="yarn", serializer=None, project_folder=None):
        self.sc = SparkParallelizer.initialize(master, serializer, project_folder)

    @staticmethod
    def initialize(master='yarn', serializer=None, project_folder=None):

        import pyspark
        import findspark
        from pyspark.serializers import PickleSerializer

        spark_conf = spark_config.SparkConfig(master)

        os.environ['PYSPARK_PYTHON'] = spark_conf.driver_location
        os.environ['PYSPARK_DRIVER_PYTHON'] = spark_conf.driver_location

        findspark.init()
        logger.info(f"Initializing Spark with version: {pyspark.__version__}")

        spark_setup = spark_conf.spark_conf_dict.items()
        conf = pyspark.SparkConf().setAll(spark_setup)
        if "local" in master:
            master = "local[*]"
        conf.setMaster(master)

        try:
            if serializer is None:
                serializer = PickleSerializer()  # Default serializer (pickle) is good enough most of the case.
            sc = pyspark.SparkContext(conf=conf, serializer=serializer)
        except Exception:
            sc = pyspark.SparkContext.getOrCreate()
        sc.setLogLevel("WARN")

        if master == 'yarn':
            proj_folder = project_folder if project_folder is not None else os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..'))
            zipped_codebase = SparkParallelizer.zip_codebase(proj_folder, ['caches', 'datasets', 'models', '.git'])
            sc.addPyFile(zipped_codebase)

        return sc

    def map_collect(self, method, list_elements, partitions_count=None, shared_variables_dict=None):
        '''
        Map a method to a list of elements over multiple Spark executors and collect the results
        '''
        return self.parallelize(method, list_elements, partitions_count, shared_variables_dict).collect()

    def stop(self):
        '''
        Stop the parallelizer.
        '''
        self.__del__()

    def parallelize(self, method, list_elements, partitions_count=None, shared_variables_dict=None):
        broadcasted_variables = {}
        if shared_variables_dict:
            for variable_name, variable_value in shared_variables_dict.items():
                broadcasted_variables[variable_name] = self.sc.broadcast(variable_value)

        if partitions_count:
            rdd = self.sc.parallelize(list_elements, partitions_count)
        else:
            rdd = self.sc.parallelize(list_elements)

        rdd = rdd.mapPartitions(lambda x: SparkParallelizer.consume_chunk(method, x, broadcasted_variables))

        return rdd

    @staticmethod
    def consume_chunk(method, chunk, broadcasted_args):
        for arg_key, arg_value in broadcasted_args.items():
            broadcasted_args[arg_key] = arg_value.value

        for el in chunk:
            yield method(el, **broadcasted_args)

    @staticmethod
    def exclude_folders(subdirs, excluded_folders):
        for excluded_folder in excluded_folders:
            if excluded_folder in subdirs:
                subdirs.remove(excluded_folder)

    @staticmethod
    def zip_codebase(input_folder, excluded_folders):
        output_zip = f"{input_folder}.zip"
        zf = zipfile.ZipFile(output_zip, "w")
        for dirname, subdirs, files in os.walk(input_folder):
            SparkParallelizer.exclude_folders(subdirs, excluded_folders)
            zf.write(dirname, os.path.relpath(dirname, input_folder))
            for filename in files:
                f = os.path.join(dirname, filename)
                zf.write(f, os.path.relpath(f, input_folder))
        zf.close()
        return output_zip

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "pproc":
                setattr(result, k, self.sc)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __enter__(self):
        if not self.sc:
            raise ValueError("Spark context not running")
        return self

    def __del__(self):
        if hasattr(self, 'sc') and self.sc:
            self.sc.stop()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'sc') and self.sc:
            self.sc.stop()


class MultiprocessingParallelizer(Parallelizer):
    '''
    A parallelizer that uses the multiprocessing library.
    Can be used as a context manager with the 'with' statement.
    @author Alexandre Renaux [Universite Libre de Brussel / Vrije Universiteit Brussel]
    '''

    def __init__(self, cpu_cores=0):
        '''
        Initialize the multiprocessing parallelizer.
        :param cpu_cores: The number of CPU cores to use (0 for all ; -1 for all minus one ; 1 for exactly one ; ...).
        '''
        super().__init__()
        self.cpu_cores = cpu_cores
        self.pool = None

    def initialize(self):
        if (not self.pool or self.pool._state != 'RUN') and self.cpu_cores != 1:
            n_cores = cpu_count() + self.cpu_cores if self.cpu_cores <= 0 else self.cpu_cores
            logger.info(f"Initializing multiprocessing parallelizer with {n_cores} cores")
            self.pool = Pool(n_cores)
        elif self.cpu_cores == 1:
            logger.info(f"All tasks will be executed sequentially [cpu_cores={self.cpu_cores}]")

    def map_collect(self, method, list_elements, partitions_count=None, shared_variables_dict=None):
        '''
        Map a method to a list of elements over multiple processes and collect the results
        :param method: The method to map.
        :param list_elements: The list of elements to map the method to.
        :param number_of_partitions: The number of partitions to use.
        :param shared_variables_dict: A dictionary of shared variables to pass to the method.
        :return: The collected results of the method for all elements in the list.
        '''
        if len(list_elements) == 0:
            return []

        partitions_count = (cpu_count() * 4) if partitions_count is None else partitions_count
        partitions_count = 1 if partitions_count < 1 or partitions_count > len(list_elements) else partitions_count
        chunksize = len(list_elements) // partitions_count
        method_func = partial(method, **shared_variables_dict)

        if self.cpu_cores != 1:
            results = []
            with tqdm(total=len(list_elements)) as pbar:
                for r in self.pool.imap_unordered(method_func, list_elements, chunksize):
                    results.append(r)
                    pbar.update()
        else:
            logger.debug("Running task sequentially")
            results = [method_func(el) for el in tqdm(list_elements)]

        return results

    def stop(self):
        '''
        Stop the parallelizer.
        '''
        self.__del__()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def __del__(self):
        if self.pool:
            self.pool.close()
            self.pool.join()




