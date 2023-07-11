from ..utils.csv_utils import CSVFileWriter
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


def write_predictions(kg, ordered_samples, predict_probas, unresolved_gene_pairs, prediction_output_folder, analysis_name):
    '''
    Write the predictions for a given gene pair.
    :param kg: the knowledge graph
    :param ordered_samples: the ordered samples (e.g gene pairs)
    :param predict_probas: the predicted probabilities (numpy array)
    :param unresolved_gene_pairs: the unresolved gene pairs (gene pairs that couldn't be parsed / identified)
    :param prediction_output_folder: the output folder
    :param analysis_name: the analysis name
    '''
    logger.info(f"Writing predictions in {prediction_output_folder}")
    unresolved_gene_pairs = [] if unresolved_gene_pairs is None else unresolved_gene_pairs
    predict_positive_probas = predict_probas[:, 1]
    res = [(proba, s, kg.convert_to_gene_name(s)) for s, proba in zip(ordered_samples, predict_positive_probas)]
    base_name = f"{analysis_name}_" if analysis_name else ""
    with CSVFileWriter(f"{prediction_output_folder}/{base_name}predictions.csv", delimiter="\t") as f:
        f.write_row(["gene_name_A", "gene_name_B", "ens_id_A", "ens_id_B", "predicted_proba"])
        for proba, ensg_ids, gene_names in sorted(res, key=lambda x: x[0], reverse=True):
            f.write_row([gene_names[0], gene_names[1], ensg_ids[0], ensg_ids[1], proba])
        for gp in unresolved_gene_pairs:
            f.write_row([gp[0], gp[1], gp[0], gp[1], "NaN"])
