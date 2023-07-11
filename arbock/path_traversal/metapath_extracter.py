from ..utils.cache_utils import Cache
from ..kg.bock import all_paths
from ..utils.dict_utils import default_to_regular
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import gmean
import numpy as np
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


class MetapathExtracter:
    '''
    This class is responsible for extracting metapaths between pairs of entities from a knowledge graph.
    @author Alexandre Renaux [Universite Libre de Bruxelles / Vrije Universiteit Brussel]
    '''

    def __init__(self, algo_params):
        '''
        Initialize the metapath extracter.
        :param algo_params: A dictionary of parameters relevant for the metapath extracter.
        '''
        self.algo_params = algo_params

    def run(self, entity_pair_ids, kg, sample_name, update_cache=False, pproc=None, partitions_count=None):
        '''
        Extract all metapaths for a given list of entity pair ids (with caching).
        :param entity_pair_ids: A list of entity pair ids.
        :param kg: A knowledge graph.
        :param sample_name: The name of the sample for caching.
        :param update_cache: Whether to update the cache.
        :param pproc: A parallel processing context (None for no parallelization).
        :param partitions_count: The number of partitions to chunk data into for parallel computing (None for auto).
        '''
        path_cutoff = self.algo_params["path_cutoff"]
        excl_node_types = set(self.algo_params.get("excl_node_types", set()))
        logger.info(f"Running metapath extracter on {len(entity_pair_ids)} entity pairs. [path_cutoff={path_cutoff}, excl_node_types={excl_node_types}]")
        cache_name = Cache.generate_cache_file_name("metapath_extracter", sample_name, self.algo_params, 'path_cutoff', 'excl_node_types')
        storage = Cache(cache_name, update_cache, single_file=True)
        metapath_dict = storage.get_or_store("", lambda x: self._run(kg, entity_pair_ids, path_cutoff, excl_node_types,
                                                                     pproc, partitions_count))
        return metapath_dict

    def _run(self, kg, entity_pair_ids, path_cutoff, excl_node_types, pproc, partitions_count=None):
        '''
        Extract all metapaths for a given list of entity pair ids (internal).
        :param kg: A knowledge graph.
        :param entity_pair_ids: A list of entity pair ids.
        :param path_cutoff: The maximum length of a path.
        :param excl_node_types: A set of node types to exclude.
        :param pproc: A parallel processing context (None for no parallelization).
        :param partitions_count: The number of partitions to chunk data into for parallel computing (None for auto).
        '''
        gene_node_pairs = list(kg.get_node_pairs(entity_pair_ids))
        if pproc:
            results = pproc.map_collect(MetapathExtracter.fetch_all_metapaths, gene_node_pairs, partitions_count,
                                        {"kg": kg, "path_cutoff": path_cutoff, "excl_node_types": excl_node_types})
        else:
            results = [MetapathExtracter.fetch_all_metapaths(pair, kg, path_cutoff, excl_node_types) for pair in tqdm(gene_node_pairs)]

        metapath_dict = MetapathExtracter.metapaths_to_dict(results)
        logger.info(f"Number of gene pairs with path information retrieved = {len(metapath_dict)}")
        return metapath_dict

    @staticmethod
    def fetch_all_metapaths(source_target, kg, path_cutoff, excl_node_types):
        '''
        Fetch all metapaths for a given source-target pair (internal KG node ids)
        :param source_target: A source-target pair (internal KG node ids)
        :param kg: A knowledge graph.
        :param path_cutoff: The maximum length of a path.
        :param excl_node_types: A set of node types to exclude.
        '''
        graph = kg.g
        source, target = source_target
        source_id = graph.vp.id[source]
        target_id = graph.vp.id[target]
        metapath_to_paths = defaultdict(dict)
        for p in all_paths(kg.g, source, target, cutoff=path_cutoff, edges=True):
            if MetapathExtracter.valid_path(p, kg, excl_node_types):
                MetapathExtracter.process_path(metapath_to_paths, p, kg)
        pair_id = (source_id, target_id)
        return pair_id, default_to_regular(metapath_to_paths)

    @staticmethod
    def get_path_score(path, kg):
        '''
        Get the path score (geometric mean of edge scores).
        :param path: A path (list of edges).
        :param kg: A knowledge graph.
        '''
        edge_scores = []
        for edge in path:
            edge_scores.append(kg.get_edge_score(edge))
        return gmean(np.array(edge_scores) + 0.0000001)

    @staticmethod
    def valid_path(p, kg, excl_node_types):
        '''
        Check if a path is valid.
        :param p: A path (list of edges).
        :param kg: A knowledge graph.
        :param excl_node_types: A set of node types to exclude.
        '''
        prev_in_property = None
        for edge in p:
            # Filtering by node types
            target_node_type = kg.get_node_label(edge.target())
            if target_node_type in excl_node_types:
                return False
            # Filtering by mixed "in" properties in edges
            in_property = kg.get_edge_property(edge, "in")
            if in_property:
                in_property = set(eval(in_property))
                if prev_in_property is None:
                    prev_in_property = in_property
                else:
                    prev_in_property = prev_in_property.intersection(in_property)
                    if len(prev_in_property) == 0:
                        return False
        return True

    @staticmethod
    def process_path(metapath_to_paths, path, kg):
        '''
        Process a path by adding it to the metapath_to_paths dictionary.
        Each path is added to the dictionary under the metapath that it belongs to.
        A metapath is defined as a sequence of edge types and node types, recorded here as a tuple (edge_labels, node_labels, edge_directionalities).
        Path information is stored, for a given metapath, as a tuple of intermediate node ids (internal KG node ids).
        We associate to each path a path score calculated by get_path_score().
        :param metapath_to_paths: A dictionary of metapaths to paths.
        :param path: A path (list of edges).
        :param kg: A knowledge graph.
        '''
        nodes = []
        node_labels = []
        edge_labels = []
        edge_directionalities = []
        path_score = MetapathExtracter.get_path_score(path, kg)
        first_edge = True
        if path_score > 0:
            for edge in path:
                target_node = edge.target()
                edge_label = kg.g.ep.label[edge]
                target_id = kg.get_edge_property(edge, "target")
                # Capture directionality of edges
                edge_directionality = 0
                if target_id:
                    if target_id == kg.get_node_property(target_node, "id"):
                        edge_directionality = 1
                    else:
                        edge_directionality = -1
                if first_edge:
                    source_node = edge.source()
                    source_label = kg.g.vp.labels[source_node].split(":")[1:][0]
                    nodes.append(int(source_node))
                    node_labels.append(source_label)
                    first_edge = False
                edge_labels.append(edge_label)
                target_label = kg.g.vp.labels[target_node].split(":")[1:][0]
                nodes.append(int(target_node))
                node_labels.append(target_label)
                edge_directionalities.append(edge_directionality)
            intermediate_nodes = tuple(nodes[1:len(nodes) - 1])
            node_labels = tuple(node_labels)
            edge_labels = tuple(edge_labels)
            edge_directionalities = tuple(edge_directionalities)
            metapath = (edge_labels, node_labels, edge_directionalities)
            metapath_to_paths[metapath][intermediate_nodes] = path_score

    @staticmethod
    def metapaths_to_dict(gene_pair_to_metapaths):
        '''
        Convert a metapath tree iterator to a dictionary.
        :param gene_pair_to_metapaths: A metapath tree iterator.
        '''
        gene_pair_to_metapath_dict = defaultdict(lambda: defaultdict(dict))
        for gene_pair, metapath_dict in gene_pair_to_metapaths:
            for metapath, path_to_scores in metapath_dict.items():
                if len(metapath) > 0 and len(metapath[0]) > 0:
                    for path, score in path_to_scores.items():
                        gene_pair_to_metapath_dict[gene_pair][metapath][path] = score
            if gene_pair not in gene_pair_to_metapath_dict:
                gene_pair_to_metapath_dict[gene_pair] = defaultdict(dict)
        return default_to_regular(gene_pair_to_metapath_dict)





