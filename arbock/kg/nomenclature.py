from ..utils.cache_utils import Cache
from tqdm import tqdm

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"


class BOCKNomenclature:
    '''
    Nomenclature for BOCK, enabling to abbreviate node and edge types as well as metapaths.
    '''

    _METAPATH_ABBREV_SEPARATOR = ""

    def __init__(self, kg, update_cache=False):
        self.kg = kg
        cache = Cache("bock_nomenclature", update_cache, single_file=True)
        node_type_to_abbrev, edge_type_to_abbrev = cache.get_or_store("", lambda x: self.init_nomenclature())
        self.node_type_to_abbrev = node_type_to_abbrev
        self.edge_type_to_abbrev = edge_type_to_abbrev

    def init_nomenclature(self):
        '''
        Initialize the nomenclature by parsing the KG.
        '''
        node_type_to_abbrev = {}
        edge_type_to_abbrev = {}
        for edge in tqdm(self.kg.g.edges()):
            self.parse_node(edge.source(), node_type_to_abbrev)
            self.parse_node(edge.target(), node_type_to_abbrev)
            self.parse_edge(edge, edge_type_to_abbrev)
        return node_type_to_abbrev, edge_type_to_abbrev

    def parse_node(self, node, node_type_to_abbrev):
        '''
        Parse a node and add it to the nomenclature.
        :param node: The node to parse.
        :param node_type_to_abbrev: The dictionary of node types to abbreviations.
        '''
        node_label = self.kg.get_node_label(node)
        if node_label not in node_type_to_abbrev:
            abbrev = self.kg.get_node_property(node, "abbrevType")
            node_type_to_abbrev[node_label] = abbrev

    def parse_edge(self, edge, edge_type_to_abbrev):
        '''
        Parse an edge and add it to the nomenclature.
        :param edge: The edge to parse.
        :param edge_type_to_abbrev: The dictionary of edge types to abbreviations.
        '''
        edge_label = self.kg.get_edge_label(edge)
        if edge_label not in edge_type_to_abbrev:
            abbrev = self.kg.get_edge_property(edge, "abbrevType")
            edge_type_to_abbrev[edge_label] = abbrev

    def abbreviate_metapath(self, metapath):
        '''
        Abbreviate a metapath.
        :param metapath: The metapath to abbreviate.
        :return: The abbreviated metapath.
        '''
        edge_types, node_types, edge_directions = metapath
        edge_types = iter(edge_types)
        edge_directions = iter(edge_directions)
        abbrev_metapath = []
        previous_node_type = next_edge_type = next_edge_direction = None
        for node_type in node_types:
            node_type, *remainder = node_type.split("#")
            node_var = next(iter(remainder), None)
            if previous_node_type is not None:
                next_edge_type = next(edge_types, None)
                next_edge_direction = next(edge_directions, None)
            if next_edge_type is not None:
                if next_edge_direction == -1 and node_type == previous_node_type:
                    abbrev_metapath.append("<")
                abbrev_metapath.append(self.edge_type_to_abbrev[next_edge_type])
                if next_edge_direction == 1 and node_type == previous_node_type:
                    abbrev_metapath.append(">")
            else:
                pass
            node_var_formatted = f"_{node_var}" if node_var else ""
            abbrev_metapath.append(self.node_type_to_abbrev[node_type] + node_var_formatted)
            previous_node_type = node_type
        return self._METAPATH_ABBREV_SEPARATOR.join(abbrev_metapath)

    def abbreviate_metaedge(self, metaedge, direction):
        """
        Abbreviate a metaedge
        :param metaedge: Should be formatted as tuple: (NodeType1, EdgeType, NodeType2)
        :param direction: 1 for forward and -1 for reverse direction
        :return: abbreviation of metaedge
        """
        abbrev_metaedge = []
        abbrev_metaedge.append(self.node_type_to_abbrev[metaedge[0]])
        if metaedge[0] == metaedge[2] and direction == -1:
            abbrev_metaedge.append("<")
        abbrev_metaedge.append(self.edge_type_to_abbrev[metaedge[1]])
        if metaedge[0] == metaedge[2] and direction == 1:
            abbrev_metaedge.append(">")
        abbrev_metaedge.append(self.node_type_to_abbrev[metaedge[2]])
        return "".join(abbrev_metaedge)

    def prettify_metaedge(self, metaedge, direction):
        """
        Abbreviate a metaedge
        :param metaedge: Should be formatted as tuple: (NodeType1, EdgeType, NodeType2)
        :param direction: 1 for forward and -1 for reverse direction
        :return: abbreviation of metaedge
        """
        abbrev_metaedge = []
        abbrev_metaedge.append(metaedge[0])
        rel = ""
        if metaedge[0] == metaedge[2] and direction == -1:
            rel += "<"
        rel += metaedge[1]
        if metaedge[0] == metaedge[2] and direction == 1:
            rel += ">"
        abbrev_metaedge.append(rel)
        abbrev_metaedge.append(metaedge[2])
        return "--".join(abbrev_metaedge)
