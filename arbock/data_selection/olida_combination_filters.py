import enum
import datetime
import logging

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

logger = logging.getLogger(__name__)


class OLIDAEvidenceFilter(enum.Enum):

    WEAK = (1, lambda evidences: int(evidences["FAMmanual"]) >= 1 or (int(evidences["STATmeta"]) >= 1 and int(evidences["STATmanual"]) >= 1))
    MODERATE = (2, lambda evidences: int(evidences["FAMmanual"]) >= 2 or int(evidences["STATmeta"]) >= 2)
    STRONG = (3, lambda evidences: int(evidences["FAMmanual"]) >= 3 or int(evidences["STATmeta"]) >= 2)

    @staticmethod
    def get_confidence_level(evidences):
        confidence_levels = []
        for strategy in OLIDAEvidenceFilter:
            confidence_level, strategy_filter = strategy.value
            if strategy_filter(evidences):
                confidence_levels.append(confidence_level)
        if not confidence_levels:
            return 0
        return max(confidence_levels)


def select_most_recent_non_redundant_combinations(combination_to_publication_timestamp, most_recent_count):
    if most_recent_count == 0:
        return []

    sorted_comb_timestamps_desc = sorted(combination_to_publication_timestamp.items(), key=lambda x: x[1], reverse=True)

    # Redundancy filtering (to avoid genes from appearing multiple times in combinations)
    unique_gene_id = set()
    selected = []
    for comb, timestamp in sorted_comb_timestamps_desc:
        if len(selected) == most_recent_count:
            break
        if comb[0] in unique_gene_id or comb[1] in unique_gene_id:
            continue
        unique_gene_id.update(comb)
        selected.append((comb, timestamp))

    # Selection of the most recent combinations based on publication timestamp
    from_selected_timestamp = selected[-1][1]
    from_selected_datetime = datetime.datetime.fromtimestamp(from_selected_timestamp)
    most_recent_combs = [comb for comb, timestamp in selected]
    logger.info(f"Holding out {len(most_recent_combs)} combinations from the positive training set. Temporal filtering starting from publication date: {from_selected_datetime.strftime('%d/%m/%Y')}.")
    for comb, timestamp in selected:
        logger.debug(f"Held out combination = {comb}, timestamp={timestamp}, {datetime.datetime.fromtimestamp(timestamp).strftime('%d/%m/%Y')}")
    return most_recent_combs