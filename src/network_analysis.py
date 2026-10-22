"""
network_analysis.py
===================

Costruzione e descrizione delle reti di co-condivisione
tra gruppi Facebook.

Metodologia
-----------
Nodi:
    Gruppi Facebook identificati da un ID univoco.

Archi:
    Due gruppi sono collegati se hanno pubblicato
    almeno un contenuto in comune.

Contenuti condivisi:
    1. Identificativo del post ricondiviso.
    2. URL del link allegato, quando manca l'ID.

Peso dell'arco:
    Numero di contenuti distinti condivisi
    dai due gruppi.

Il grafo è semplice, non orientato e pesato.

La rete finale comprende soltanto gruppi
con almeno un collegamento.

Il modulo non utilizza hashtag, parole chiave
o similarità semantica.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""

from collections import defaultdict
from itertools import combinations
from urllib.parse import urlsplit, urlunsplit

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

DEFAULT_GROUP_COLUMN = "group_id"

DEFAULT_SHARED_POST_COLUMN = "shared_post_id"

DEFAULT_LINK_COLUMN = "link_attachment"

DEFAULT_CONTENT_COLUMN = "shared_content"


# ============================================================
# 2. NORMALIZZAZIONE DEGLI IDENTIFICATIVI
# ============================================================

def normalize_identifier(value):
    """
    Normalizza un identificativo.

    Restituisce None per valori mancanti
    o non validi.

    Gli identificativi vengono trattati
    come stringhe, non come numeri.
    """

    if value is None:
        return None

    if isinstance(value, (dict, list, tuple, set)):
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.lower() in {
        "nan",
        "none",
        "null",
        "nat",
    }:
        return None

    return value


# ============================================================
# 3. ESTRAZIONE DELL'IDENTIFICATIVO DEL GRUPPO
# ============================================================

def extract_group_id(
    value,
    id_key="id",
):
    """
    Estrae l'identificativo di un gruppo.

    Supporta:
    - ID già presente come stringa;
    - dizionario contenente la chiave 'id'.

    Non identifica automaticamente il gruppo
    a partire da un nome visualizzato.
    """

    if isinstance(value, dict):
        value = value.get(id_key)

    return normalize_identifier(value)


def prepare_group_ids(
    df,
    source_column=DEFAULT_GROUP_COLUMN,
    output_column="group_id",
):
    """
    Prepara la colonna degli identificativi
    dei gruppi Facebook.

    Parameters
    ----------
    source_column : str
        Colonna contenente l'ID del gruppo
        oppure un dizionario con il suo ID.

    output_column : str
        Colonna da creare.

    Returns
    -------
    pandas.DataFrame
        Copia del dataset con group_id.
    """

    if source_column not in df.columns:
        raise KeyError(
            f"Colonna mancante: {source_column}"
        )

    result = df.copy()

    result[output_column] = (
        result[source_column]
        .map(extract_group_id)
    )

    return result


# ============================================================
# 4. ESTRAZIONE DEI LINK
# ============================================================

def extract_link(value):
    """
    Estrae un URL da una stringa o da un dizionario.

    Chiavi supportate:
    - url
    - link
    - href

    La funzione non inventa URL
    quando il campo è assente.
    """

    if isinstance(value, dict):

        value = (
            value.get("url")
            or value.get("link")
            or value.get("href")
        )

    return normalize_identifier(value)


def normalize_url(url):
    """
    Applica una normalizzazione conservativa agli URL.

    Operazioni:
    - rimozione degli spazi esterni;
    - uniformazione di schema e hostname;
    - rimozione del frammento finale.

    Non elimina parametri di query.
    Non risolve redirect.
    Non modifica il percorso.

    La normalizzazione può essere disattivata
    nella costruzione degli identificativi.
    """

    url = extract_link(url)

    if url is None:
        return None

    try:
        parsed = urlsplit(url)

        if not parsed.scheme or not parsed.netloc:
            return url

        normalized = urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                parsed.query,
                "",
            )
        )

        return normalized

    except ValueError:
        return url


# ============================================================
# 5. IDENTIFICAZIONE DEL CONTENUTO CONDIVISO
# ============================================================

def extract_shared_content(
    shared_post_id=None,
    link_attachment=None,
    normalize_links=False,
):
    """
    Identifica il contenuto condiviso.

    Priorità:
    1. shared_post_id
    2. link_attachment

    Returns
    -------
    str or None
        Identificativo del contenuto.

    Notes
    -----
    I prefissi POST_ e URL_ evitano collisioni
    tra identificativi di tipo diverso.

    normalize_links=False mantiene gli URL
    nella forma registrata nel dataset.
    """

    shared_id = normalize_identifier(
        shared_post_id
    )

    if shared_id is not None:

        return "POST_" + shared_id

    link = extract_link(
        link_attachment
    )

    if link is None:
        return None

    if normalize_links:
        link = normalize_url(link)

    return "URL_" + link


def prepare_shared_contents(
    df,
    shared_post_column=DEFAULT_SHARED_POST_COLUMN,
    link_column=DEFAULT_LINK_COLUMN,
    output_column=DEFAULT_CONTENT_COLUMN,
    normalize_links=False,
):
    """
    Aggiunge l'identificativo del contenuto condiviso.

    Il dataset deve contenere almeno una
    delle due colonne sorgente.
    """

    if (
        shared_post_column not in df.columns
        and link_column not in df.columns
    ):
        raise ValueError(
            "Nessuna colonna disponibile per "
            "identificare i contenuti condivisi."
        )

    result = df.copy()

    if shared_post_column in result.columns:
        shared_values = result[
            shared_post_column
        ].tolist()
    else:
        shared_values = [
            None
        ] * len(result)

    if link_column in result.columns:
        link_values = result[
            link_column
        ].tolist()
    else:
        link_values = [
            None
        ] * len(result)

    result[output_column] = [
        extract_shared_content(
            shared_post_id=shared_id,
            link_attachment=link,
            normalize_links=normalize_links,
        )
        for shared_id, link in zip(
            shared_values,
            link_values,
        )
    ]

    return result


# ============================================================
# 6. PREPARAZIONE DELLE COPPIE GRUPPO-CONTENUTO
# ============================================================

def prepare_group_content_pairs(
    df,
    group_column=DEFAULT_GROUP_COLUMN,
    content_column=DEFAULT_CONTENT_COLUMN,
):
    """
    Prepara le coppie distinte gruppo-contenuto.

    Un gruppo che pubblica più volte lo stesso
    contenuto viene contato una sola volta
    per quel contenuto.

    Returns
    -------
    pandas.DataFrame
        Colonne:
        - group_id
        - shared_content
    """

    required = [
        group_column,
        content_column,
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonne mancanti: {missing}"
        )

    pairs = df[
        required
    ].copy()

    pairs[group_column] = (
        pairs[group_column]
        .map(extract_group_id)
    )

    pairs[content_column] = (
        pairs[content_column]
        .map(normalize_identifier)
    )

    pairs = (
        pairs
        .dropna(
            subset=required
        )
        .drop_duplicates(
            subset=required
        )
        .reset_index(drop=True)
    )

    return pairs


# ============================================================
# 7. IDENTIFICAZIONE DEI CONTENUTI CONDIVISI
# ============================================================

def get_shared_content_groups(
    pairs,
    group_column=DEFAULT_GROUP_COLUMN,
    content_column=DEFAULT_CONTENT_COLUMN,
):
    """
    Identifica i contenuti pubblicati
    in almeno due gruppi distinti.

    Returns
    -------
    dict
        Chiave: contenuto.
        Valore: lista ordinata di gruppi.
    """

    required = {
        group_column,
        content_column,
    }

    if not required.issubset(pairs.columns):
        raise ValueError(
            "Il dataset non contiene le colonne "
            "gruppo e contenuto."
        )

    grouped = (
        pairs.groupby(
            content_column
        )[group_column]
        .agg(
            lambda values: sorted(
                set(values)
            )
        )
    )

    grouped = grouped[
        grouped.map(len) >= 2
    ]

    return grouped.to_dict()


# ============================================================
# 8. COSTRUZIONE DEGLI ARCHI
# ============================================================

def build_edge_weights(
    content_groups,
):
    """
    Costruisce gli archi pesati.

    Per ogni contenuto condiviso vengono
    generate tutte le coppie di gruppi.

    Il peso aumenta di uno per ciascun
    contenuto distinto condiviso.

    Returns
    -------
    dict
        (group_a, group_b) -> weight
    """

    edge_weights = defaultdict(int)

    for groups in content_groups.values():

        unique_groups = sorted(
            set(groups)
        )

        if len(unique_groups) < 2:
            continue

        for group_a, group_b in combinations(
            unique_groups,
            2,
        ):

            edge_weights[
                (group_a, group_b)
            ] += 1

    return dict(edge_weights)


# ============================================================
# 9. COSTRUZIONE DEL GRAFO
# ============================================================

def build_network(
    edge_weights,
    include_isolates=False,
    all_groups=None,
):
    """
    Costruisce un grafo NetworkX non orientato.

    Parameters
    ----------
    edge_weights : dict
        Dizionario delle coppie e dei pesi.

    include_isolates : bool
        Se True, aggiunge anche i gruppi
        privi di collegamenti.

    all_groups : iterable, optional
        Elenco completo dei gruppi.

    Returns
    -------
    networkx.Graph
        Grafo non orientato e pesato.
    """

    graph = nx.Graph()

    for (group_a, group_b), weight in (
        edge_weights.items()
    ):

        if group_a == group_b:
            continue

        if weight <= 0:
            raise ValueError(
                "I pesi degli archi devono "
                "essere positivi."
            )

        graph.add_edge(
            group_a,
            group_b,
            weight=int(weight),
        )

    if include_isolates:

        if all_groups is None:
            raise ValueError(
                "Per includere i nodi isolati "
                "è necessario fornire all_groups."
            )

        for group in all_groups:

            group_id = extract_group_id(
                group
            )

            if group_id is not None:
                graph.add_node(group_id)

    return graph


# ============================================================
# 10. PIPELINE COMPLETA DELLA RETE
# ============================================================

def construct_co_sharing_network(
    df,
    group_source_column="group_id",
    shared_post_column="shared_post_id",
    link_column="link_attachment",
    normalize_links=False,
    include_isolates=False,
):
    """
    Costruisce la rete di co-condivisione
    a partire dal dataset dei post.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset contenente i post.

    group_source_column : str
        Colonna da cui ricavare l'ID del gruppo.

    shared_post_column : str
        Colonna dell'ID del post ricondiviso.

    link_column : str
        Colonna del link allegato.

    normalize_links : bool
        Se True, normalizza gli URL.

    include_isolates : bool
        Se True, conserva anche i gruppi isolati.

    Returns
    -------
    dict
        - graph
        - prepared_data
        - group_content_pairs
        - shared_content_groups
        - edge_weights
    """

    prepared = prepare_group_ids(
        df,
        source_column=group_source_column,
        output_column="group_id",
    )

    prepared = prepare_shared_contents(
        prepared,
        shared_post_column=shared_post_column,
        link_column=link_column,
        output_column="shared_content",
        normalize_links=normalize_links,
    )

    pairs = prepare_group_content_pairs(
        prepared,
        group_column="group_id",
        content_column="shared_content",
    )

    content_groups = get_shared_content_groups(
        pairs,
        group_column="group_id",
        content_column="shared_content",
    )

    edge_weights = build_edge_weights(
        content_groups
    )

    all_groups = (
        prepared["group_id"]
        .dropna()
        .unique()
    )

    graph = build_network(
        edge_weights,
        include_isolates=include_isolates,
        all_groups=all_groups,
    )

    return {
        "graph": graph,
        "prepared_data": prepared,
        "group_content_pairs": pairs,
        "shared_content_groups": content_groups,
        "edge_weights": edge_weights,
    }


# ============================================================
# 11. STATISTICHE DESCRITTIVE DELLA RETE
# ============================================================

def network_statistics(graph):
    """
    Calcola le principali statistiche
    descrittive del grafo.

    Metriche:
    - numero di nodi;
    - numero di archi;
    - densità;
    - componenti connesse;
    - dimensione della componente maggiore;
    - grado medio;
    - peso medio degli archi.
    """

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    density = nx.density(graph)

    components = list(
        nx.connected_components(graph)
    )

    n_components = len(components)

    largest_component = max(
        (
            len(component)
            for component in components
        ),
        default=0,
    )

    if n_nodes > 0:

        mean_degree = (
            sum(
                dict(graph.degree()).values()
            ) / n_nodes
        )

    else:
        mean_degree = np.nan

    weights = [
        data.get("weight", 1)
        for _, _, data in graph.edges(
            data=True
        )
    ]

    mean_weight = (
        float(np.mean(weights))
        if weights
        else np.nan
    )

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "n_components": n_components,
        "largest_component": largest_component,
        "mean_degree": mean_degree,
        "mean_edge_weight": mean_weight,
    }


# ============================================================
# 12. TABELLA DEGLI ARCHI
# ============================================================

def network_edge_table(graph):
    """
    Restituisce una tabella degli archi.

    Ogni riga rappresenta una coppia
    di gruppi collegati.
    """

    records = []

    for group_a, group_b, data in (
        graph.edges(data=True)
    ):

        records.append({
            "group_a": group_a,
            "group_b": group_b,
            "weight": data.get(
                "weight",
                1,
            ),
        })

    return pd.DataFrame(
        records,
        columns=[
            "group_a",
            "group_b",
            "weight",
        ],
    )


# ============================================================
# 13. TABELLA DEI NODI
# ============================================================

def network_node_table(graph):
    """
    Restituisce una tabella dei gruppi
    presenti nella rete.

    Metriche:
    - grado;
    - grado pesato;
    - componente connessa.
    """

    component_map = {}

    for component_id, component in enumerate(
        nx.connected_components(graph),
        start=1,
    ):

        for node in component:
            component_map[node] = component_id

    records = []

    for node in graph.nodes():

        records.append({
            "group_id": node,
            "degree": graph.degree(node),
            "weighted_degree": graph.degree(
                node,
                weight="weight",
            ),
            "component": component_map.get(
                node
            ),
        })

    return pd.DataFrame(
        records,
        columns=[
            "group_id",
            "degree",
            "weighted_degree",
            "component",
        ],
    )


# ============================================================
# 14. RIEPILOGO COMPLETO
# ============================================================

def network_analysis_summary(
    df,
    group_source_column="group_id",
    shared_post_column="shared_post_id",
    link_column="link_attachment",
    normalize_links=False,
):
    """
    Esegue la costruzione della rete
    e produce le statistiche descrittive.

    Il grafo finale contiene soltanto
    gruppi con almeno un collegamento.

    Returns
    -------
    dict
        Risultati della costruzione
        e descrizione della rete.
    """

    results = construct_co_sharing_network(
        df,
        group_source_column=group_source_column,
        shared_post_column=shared_post_column,
        link_column=link_column,
        normalize_links=normalize_links,
        include_isolates=False,
    )

    graph = results["graph"]

    results["statistics"] = (
        network_statistics(graph)
    )

    results["nodes"] = (
        network_node_table(graph)
    )

    results["edges"] = (
        network_edge_table(graph)
    )

    return results
