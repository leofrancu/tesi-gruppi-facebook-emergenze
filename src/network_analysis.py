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

Per impostazione predefinita, la rete comprende
soltanto gruppi con almeno un collegamento.

Il modulo non utilizza hashtag, parole chiave
o similarità semantica.

La sincronia temporale non viene utilizzata
come criterio di costruzione degli archi.

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
# 2. VALIDAZIONE DEL DATASET
# ============================================================

def validate_dataframe(df):
    """
    Verifica che l'input sia un DataFrame pandas
    e che non contenga nomi di colonne duplicati.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "L'input deve essere un pandas.DataFrame."
        )

    if not df.columns.is_unique:
        raise ValueError(
            "Il DataFrame contiene nomi di colonne duplicati."
        )

    return df


def validate_columns(df, required_columns):
    """
    Verifica la presenza delle colonne richieste.
    """

    validate_dataframe(df)

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonne mancanti nel dataset: {missing}"
        )

    return True


# ============================================================
# 3. NORMALIZZAZIONE DEGLI IDENTIFICATIVI
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
# 4. ESTRAZIONE DELL'IDENTIFICATIVO DEL GRUPPO
# ============================================================

def extract_group_id(value, id_key="id"):
    """
    Estrae l'identificativo di un gruppo.

    Supporta:
    - identificativo già presente come stringa;
    - dizionario contenente la chiave 'id'.

    Non ricava automaticamente l'ID da nomi
    visualizzati o da altri campi testuali.
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
    Prepara gli identificativi dei gruppi.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset dei post.

    source_column : str
        Colonna contenente l'ID del gruppo
        oppure un dizionario con il suo ID.

    output_column : str
        Nome della colonna da creare.

    Returns
    -------
    pandas.DataFrame
        Copia del dataset con identificativi
        normalizzati.
    """

    validate_columns(
        df,
        [source_column],
    )

    result = df.copy()

    result[output_column] = (
        result[source_column]
        .map(extract_group_id)
    )

    return result


# ============================================================
# 5. ESTRAZIONE E NORMALIZZAZIONE DEGLI URL
# ============================================================

def extract_link(value):
    """
    Estrae un URL da una stringa o da un dizionario.

    Chiavi supportate:
    - url
    - link
    - href
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
    Normalizzazione conservativa degli URL.

    Operazioni:
    - rimozione degli spazi esterni;
    - uniformazione di schema e hostname;
    - rimozione del frammento finale.

    Non elimina parametri di query.
    Non risolve redirect.
    Non modifica il percorso.
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
# 6. IDENTIFICAZIONE DEL CONTENUTO CONDIVISO
# ============================================================

def extract_shared_content(
    shared_post_id=None,
    link_attachment=None,
    normalize_links=False,
):
    """
    Identifica un contenuto condiviso.

    Priorità:
    1. shared_post_id
    2. link_attachment

    I prefissi POST_ e URL_ impediscono
    collisioni tra identificativi diversi.
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

    È necessaria almeno una delle due colonne:
    shared_post_id oppure link_attachment.
    """

    validate_dataframe(df)

    if (
        shared_post_column not in df.columns
        and link_column not in df.columns
    ):
        raise ValueError(
            "Impossibile identificare i contenuti condivisi: "
            f"mancano sia '{shared_post_column}' "
            f"sia '{link_column}'."
        )

    result = df.copy()

    if shared_post_column in result.columns:
        shared_values = result[
            shared_post_column
        ].tolist()
    else:
        shared_values = [None] * len(result)

    if link_column in result.columns:
        link_values = result[
            link_column
        ].tolist()
    else:
        link_values = [None] * len(result)

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
# 7. PREPARAZIONE DELLE COPPIE GRUPPO-CONTENUTO
# ============================================================

def prepare_group_content_pairs(
    df,
    group_column=DEFAULT_GROUP_COLUMN,
    content_column=DEFAULT_CONTENT_COLUMN,
):
    """
    Prepara le coppie distinte gruppo-contenuto.

    Un gruppo che pubblica più volte lo stesso
    contenuto viene contato una sola volta.
    """

    validate_columns(
        df,
        [group_column, content_column],
    )

    pairs = df[
        [group_column, content_column]
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
            subset=[
                group_column,
                content_column,
            ]
        )
        .drop_duplicates(
            subset=[
                group_column,
                content_column,
            ]
        )
        .reset_index(drop=True)
    )

    return pairs


# ============================================================
# 8. IDENTIFICAZIONE DEI CONTENUTI CONDIVISI
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

    validate_columns(
        pairs,
        [group_column, content_column],
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
# 9. COSTRUZIONE DEGLI ARCHI
# ============================================================

def build_edge_weights(content_groups):
    """
    Costruisce gli archi pesati.

    Per ogni contenuto condiviso vengono
    generate tutte le coppie di gruppi.

    Il peso aumenta di uno per ogni
    contenuto distinto condiviso.
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
# 10. COSTRUZIONE DEL GRAFO
# ============================================================

def build_network(
    edge_weights,
    include_isolates=False,
    all_groups=None,
):
    """
    Costruisce un grafo NetworkX non orientato
    e pesato.

    Per impostazione predefinita, esclude
    i gruppi privi di collegamenti.
    """

    graph = nx.Graph()

    for (group_a, group_b), weight in (
        edge_weights.items()
    ):

        if group_a == group_b:
            continue

        if weight <= 0:
            raise ValueError(
                "I pesi degli archi devono essere positivi."
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
# 11. PIPELINE COMPLETA DELLA RETE
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

    Returns
    -------
    dict
        - graph
        - prepared_data
        - group_content_pairs
        - shared_content_groups
        - edge_weights
    """

    validate_dataframe(df)

    # Verifica esplicita della colonna dei gruppi.
    # Nei dati Meta l'identificativo può provenire
    # dal campo 'surface', che deve contenere un ID
    # direttamente oppure in un dizionario.

    if group_source_column not in df.columns:
        raise ValueError(
            f"Colonna '{group_source_column}' non trovata. "
            "Identificare i gruppi Facebook a partire "
            "dal campo 'surface' durante il preprocessing "
            "oppure specificare group_source_column='surface' "
            "se tale campo contiene identificativi validi."
        )

    prepared = prepare_group_ids(
        df,
        source_column=group_source_column,
        output_column="group_id",
    )

    # Verifica che siano presenti identificativi validi.

    if prepared["group_id"].notna().sum() == 0:
        raise ValueError(
            "Nessun identificativo di gruppo valido. "
            "Verificare il formato della colonna "
            f"'{group_source_column}'."
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
# 12. STATISTICHE DESCRITTIVE DELLA RETE
# ============================================================

def network_statistics(graph):
    """
    Calcola le statistiche descrittive del grafo.

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
# 13. TABELLA DEGLI ARCHI
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
# 14. TABELLA DEI NODI
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
# 15. RIEPILOGO COMPLETO
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
