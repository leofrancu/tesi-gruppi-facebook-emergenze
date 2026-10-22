"""
network_statistics.py
=====================

Analisi statistica delle reti di co-condivisione
tra gruppi Facebook.

Metodologia
-----------
1. Profili di reaction per gruppo.
2. Similarità coseno tra profili.
3. Coppie collegate e non collegate.
4. Test di permutazione a due code.
5. Dimensione dell'effetto: d di Cohen.
6. Analisi di sensibilità per soglie di reaction.
7. Controllo della componente circolare.

Il grafo è costruito dal modulo network_analysis.py.

Le reaction considerate sono:
like, love, wow, haha, sad, angry.

I risultati descrivono associazioni e non
dimostrano relazioni causali o influenza
tra gruppi Facebook.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

REACTION_COLUMNS = (
    "like",
    "love",
    "wow",
    "haha",
    "sad",
    "angry",
)

DEFAULT_PERMUTATIONS = 10_000

DEFAULT_RANDOM_STATE = 42

REACTION_THRESHOLDS = (0, 10, 50)


# ============================================================
# 2. VALIDAZIONE
# ============================================================

def validate_reaction_columns(
    df,
    reaction_columns=REACTION_COLUMNS,
):
    """
    Verifica la presenza e la validità
    delle colonne delle reaction.
    """

    missing = [
        column
        for column in reaction_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonne reaction mancanti: {missing}"
        )

    for column in reaction_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Valori non validi in '{column}'."
            )

        if not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Valori non finiti in '{column}'."
            )

        if (values < 0).any():
            raise ValueError(
                f"Conteggi negativi in '{column}'."
            )


def validate_graph(graph):
    """
    Verifica che il grafo sia semplice
    e non orientato.
    """

    if not isinstance(graph, nx.Graph):
        raise TypeError(
            "È richiesto un grafo NetworkX."
        )

    if graph.is_directed():
        raise ValueError(
            "Il grafo deve essere non orientato."
        )

    if graph.is_multigraph():
        raise ValueError(
            "Il grafo non deve essere un multigrafo."
        )


# ============================================================
# 3. PROFILI DI REACTION PER GRUPPO
# ============================================================

def build_group_reaction_profiles(
    df,
    group_column="group_id",
    reaction_columns=REACTION_COLUMNS,
    min_reactions=0,
):
    """
    Costruisce i profili di reaction dei gruppi.

    Il profilo è il vettore delle quote
    delle sei reaction sul totale del gruppo.

    Formula
    -------
    profile(g, r) =
        reaction(g, r) / total_reactions(g)

    I gruppi senza reaction sono esclusi.

    Parameters
    ----------
    min_reactions : int
        Soglia minima di reaction complessive.

        Anche con min_reactions=0 vengono
        esclusi i gruppi senza reaction.

    Returns
    -------
    pandas.DataFrame
        Un profilo normalizzato per gruppo.
    """

    if group_column not in df.columns:
        raise KeyError(
            f"Colonna mancante: {group_column}"
        )

    validate_reaction_columns(
        df,
        reaction_columns,
    )

    if min_reactions < 0:
        raise ValueError(
            "La soglia non può essere negativa."
        )

    grouped = (
        df.dropna(subset=[group_column])
        .groupby(group_column)[
            list(reaction_columns)
        ]
        .sum()
    )

    totals = grouped.sum(axis=1)

    valid = (
        (totals > 0)
        & (totals >= min_reactions)
    )

    grouped = grouped.loc[valid]
    totals = totals.loc[valid]

    profiles = grouped.div(
        totals,
        axis=0,
    )

    profiles.index.name = "group_id"

    return profiles


# ============================================================
# 4. SIMILARITÀ COSENO
# ============================================================

def cosine_similarity(
    vector_a,
    vector_b,
):
    """
    Calcola la similarità coseno
    tra due vettori numerici.

    Formula
    -------
    cos(a, b) =
        dot(a, b) / (norm(a) * norm(b))

    Returns
    -------
    float
        Similarità compresa tra 0 e 1
        per vettori non negativi.

    Restituisce NaN per vettori nulli.
    """

    a = np.asarray(
        vector_a,
        dtype=float,
    )

    b = np.asarray(
        vector_b,
        dtype=float,
    )

    if a.shape != b.shape:
        raise ValueError(
            "I vettori devono avere "
            "la stessa dimensione."
        )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return np.nan

    return float(
        np.dot(a, b)
        / (norm_a * norm_b)
    )


# ============================================================
# 5. COPPIE COLLEGATE
# ============================================================

def get_connected_pairs(
    graph,
    valid_groups,
):
    """
    Estrae le coppie collegate da un arco.

    Sono mantenute soltanto le coppie
    per cui entrambi i gruppi possiedono
    un profilo di reaction valido.
    """

    validate_graph(graph)

    valid_groups = set(valid_groups)

    pairs = []

    for group_a, group_b in graph.edges():

        if (
            group_a in valid_groups
            and group_b in valid_groups
        ):
            pairs.append(
                (group_a, group_b)
            )

    return pairs


# ============================================================
# 6. CAMPIONAMENTO DELLE COPPIE NON COLLEGATE
# ============================================================

def sample_unconnected_pairs(
    graph,
    valid_groups,
    n_pairs,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Campiona coppie di gruppi non collegati.

    Il numero richiesto corrisponde
    al numero di coppie collegate.

    Il campionamento avviene senza
    reinserimento delle coppie.

    Se le coppie disponibili sono insufficienti,
    viene generato un errore.

    Notes
    -----
    La procedura è pensata per reti di dimensioni
    compatibili con l'enumerazione delle coppie.
    """

    validate_graph(graph)

    rng = np.random.default_rng(
        random_state
    )

    groups = sorted(
        set(valid_groups)
        .intersection(graph.nodes()),
        key=str,
    )

    candidates = [
        (group_a, group_b)
        for group_a, group_b in combinations(
            groups,
            2,
        )
        if not graph.has_edge(
            group_a,
            group_b,
        )
    ]

    if len(candidates) < n_pairs:
        raise ValueError(
            "Numero insufficiente di coppie "
            "non collegate per il confronto."
        )

    if n_pairs == 0:
        return []

    selected = rng.choice(
        len(candidates),
        size=n_pairs,
        replace=False,
    )

    return [
        candidates[index]
        for index in selected
    ]


# ============================================================
# 7. SIMILARITÀ DELLE COPPIE
# ============================================================

def pair_similarities(
    pairs,
    profiles,
):
    """
    Calcola la similarità coseno
    per un insieme di coppie di gruppi.

    Returns
    -------
    numpy.ndarray
        Similarità delle coppie.
    """

    similarities = []

    for group_a, group_b in pairs:

        vector_a = profiles.loc[
            group_a
        ].to_numpy(dtype=float)

        vector_b = profiles.loc[
            group_b
        ].to_numpy(dtype=float)

        similarity = cosine_similarity(
            vector_a,
            vector_b,
        )

        similarities.append(
            similarity
        )

    return np.asarray(
        similarities,
        dtype=float,
    )


# ============================================================
# 8. TEST DI PERMUTAZIONE A DUE CODE
# ============================================================

def permutation_test(
    connected_values,
    unconnected_values,
    n_permutations=DEFAULT_PERMUTATIONS,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Test di permutazione a due code.

    Statistica:
        media_collegate - media_non_collegate

    Le similarità dei due insiemi vengono
    riaggregate casualmente per stimare
    la distribuzione nulla della differenza.

    Il valore p utilizza la correzione:
        (extreme + 1) / (n_permutations + 1)

    Returns
    -------
    dict
        Differenza osservata e valore p.

    Notes
    -----
    Il test tratta le similarità come osservazioni
    scambiabili, sebbene più coppie possano
    condividere gli stessi gruppi.

    Il valore p è pertanto indicativo.
    """

    connected = np.asarray(
        connected_values,
        dtype=float,
    )

    unconnected = np.asarray(
        unconnected_values,
        dtype=float,
    )

    connected = connected[
        np.isfinite(connected)
    ]

    unconnected = unconnected[
        np.isfinite(unconnected)
    ]

    if (
        len(connected) == 0
        or len(unconnected) == 0
    ):
        return {
            "observed_difference": np.nan,
            "p_value": np.nan,
        }

    if n_permutations < 1:
        raise ValueError(
            "n_permutations deve essere positivo."
        )

    observed = (
        connected.mean()
        - unconnected.mean()
    )

    combined = np.concatenate(
        [
            connected,
            unconnected,
        ]
    )

    n_connected = len(connected)

    rng = np.random.default_rng(
        random_state
    )

    extreme_count = 0

    for _ in range(n_permutations):

        shuffled = rng.permutation(
            combined
        )

        difference = (
            shuffled[:n_connected].mean()
            - shuffled[n_connected:].mean()
        )

        if abs(difference) >= abs(observed):
            extreme_count += 1

    p_value = (
        (extreme_count + 1)
        / (n_permutations + 1)
    )

    return {
        "observed_difference": float(
            observed
        ),
        "p_value": float(
            p_value
        ),
    }


# ============================================================
# 9. DIMENSIONE DELL'EFFETTO
# ============================================================

def cohens_d(
    connected_values,
    unconnected_values,
):
    """
    Calcola la d di Cohen.

    Formula
    -------
    d = (mean_1 - mean_2) / pooled_std

    La deviazione standard pooled utilizza
    la varianza campionaria dei due gruppi.

    Returns
    -------
    float
        Dimensione standardizzata dell'effetto.
    """

    a = np.asarray(
        connected_values,
        dtype=float,
    )

    b = np.asarray(
        unconnected_values,
        dtype=float,
    )

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    n_a = len(a)
    n_b = len(b)

    if n_a < 2 or n_b < 2:
        return np.nan

    pooled_variance = (
        (n_a - 1) * np.var(a, ddof=1)
        + (n_b - 1) * np.var(b, ddof=1)
    ) / (n_a + n_b - 2)

    if pooled_variance <= 0:
        return np.nan

    return float(
        (np.mean(a) - np.mean(b))
        / np.sqrt(pooled_variance)
    )


# ============================================================
# 10. CONFRONTO TRA GRUPPI COLLEGATI E NON
# ============================================================

def compare_network_similarities(
    graph,
    profiles,
    n_permutations=DEFAULT_PERMUTATIONS,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Confronta la similarità dei profili
    tra gruppi collegati e non collegati.

    Procedura:
    1. Selezione dei gruppi con profilo valido.
    2. Estrazione delle coppie collegate.
    3. Campionamento di un numero uguale
       di coppie non collegate.
    4. Calcolo delle similarità coseno.
    5. Test di permutazione a due code.
    6. Calcolo della d di Cohen.

    Returns
    -------
    dict
        Statistiche del confronto.
    """

    validate_graph(graph)

    valid_groups = (
        set(profiles.index)
        .intersection(graph.nodes())
    )

    connected_pairs = get_connected_pairs(
        graph,
        valid_groups,
    )

    n_connected = len(
        connected_pairs
    )

    if n_connected == 0:
        return {
            "n_groups": len(valid_groups),
            "n_connected_pairs": 0,
            "n_unconnected_pairs": 0,
            "mean_connected": np.nan,
            "mean_unconnected": np.nan,
            "median_connected": np.nan,
            "median_unconnected": np.nan,
            "mean_difference": np.nan,
            "p_value": np.nan,
            "cohens_d": np.nan,
        }

    unconnected_pairs = sample_unconnected_pairs(
        graph,
        valid_groups,
        n_pairs=n_connected,
        random_state=random_state,
    )

    connected_similarities = pair_similarities(
        connected_pairs,
        profiles,
    )

    unconnected_similarities = pair_similarities(
        unconnected_pairs,
        profiles,
    )

    test = permutation_test(
        connected_similarities,
        unconnected_similarities,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    effect_size = cohens_d(
        connected_similarities,
        unconnected_similarities,
    )

    return {
        "n_groups": len(valid_groups),

        "n_connected_pairs": len(
            connected_pairs
        ),

        "n_unconnected_pairs": len(
            unconnected_pairs
        ),

        "mean_connected": float(
            np.mean(connected_similarities)
        ),

        "mean_unconnected": float(
            np.mean(unconnected_similarities)
        ),

        "median_connected": float(
            np.median(connected_similarities)
        ),

        "median_unconnected": float(
            np.median(unconnected_similarities)
        ),

        "mean_difference": test[
            "observed_difference"
        ],

        "p_value": test[
            "p_value"
        ],

        "cohens_d": effect_size,
    }


# ============================================================
# 11. ANALISI DI SENSIBILITÀ
# ============================================================

def network_sensitivity_analysis(
    df,
    graph,
    group_column="group_id",
    reaction_columns=REACTION_COLUMNS,
    thresholds=REACTION_THRESHOLDS,
    n_permutations=DEFAULT_PERMUTATIONS,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Ripete il confronto per diverse soglie
    minime di reaction per gruppo.

    Soglie predefinite:
    - almeno 1 reaction effettiva;
    - almeno 10 reaction;
    - almeno 50 reaction.

    La soglia 0 conserva comunque soltanto
    gruppi con profilo non nullo.

    Returns
    -------
    pandas.DataFrame
        Una riga per ciascuna soglia.
    """

    results = []

    for threshold in thresholds:

        profiles = build_group_reaction_profiles(
            df,
            group_column=group_column,
            reaction_columns=reaction_columns,
            min_reactions=threshold,
        )

        comparison = compare_network_similarities(
            graph,
            profiles,
            n_permutations=n_permutations,
            random_state=random_state,
        )

        comparison["min_reactions"] = threshold

        results.append(
            comparison
        )

    result = pd.DataFrame(
        results
    )

    columns = [
        "min_reactions",
        "n_groups",
        "n_connected_pairs",
        "n_unconnected_pairs",
        "mean_connected",
        "mean_unconnected",
        "median_connected",
        "median_unconnected",
        "mean_difference",
        "p_value",
        "cohens_d",
    ]

    return result[columns]


# ============================================================
# 12. CONTROLLO DELLA COMPONENTE CIRCOLARE
# ============================================================

def remove_shared_content_posts(
    df,
    shared_content_groups,
    content_column="shared_content",
):
    """
    Esclude i post associati ai contenuti
    che hanno generato collegamenti nella rete.

    Parameters
    ----------
    shared_content_groups : dict
        Dizionario dei contenuti presenti
        in almeno due gruppi.

    Returns
    -------
    pandas.DataFrame
        Dataset senza i post relativi
        ai contenuti condivisi.

    Notes
    -----
    Il grafo originale non viene modificato.

    Vengono ricalcolati soltanto i profili
    di reaction sui post rimanenti.
    """

    if content_column not in df.columns:
        raise KeyError(
            f"Colonna mancante: {content_column}"
        )

    shared_contents = set(
        shared_content_groups.keys()
    )

    mask = ~df[
        content_column
    ].isin(shared_contents)

    return df.loc[
        mask
    ].copy()


def circularity_control(
    df,
    graph,
    shared_content_groups,
    group_column="group_id",
    content_column="shared_content",
    reaction_columns=REACTION_COLUMNS,
    min_reactions=0,
    n_permutations=DEFAULT_PERMUTATIONS,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Controlla la componente circolare
    della similarità tra gruppi collegati.

    Procedura:
    1. Mantiene il grafo originale.
    2. Rimuove i post che hanno generato gli archi.
    3. Ricostruisce i profili sui post rimanenti.
    4. Ripete il confronto tra gruppi
       collegati e non collegati.

    Returns
    -------
    dict
        Risultati del confronto dopo esclusione.
    """

    filtered = remove_shared_content_posts(
        df,
        shared_content_groups,
        content_column=content_column,
    )

    profiles = build_group_reaction_profiles(
        filtered,
        group_column=group_column,
        reaction_columns=reaction_columns,
        min_reactions=min_reactions,
    )

    result = compare_network_similarities(
        graph,
        profiles,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    result["remaining_posts"] = len(
        filtered
    )

    result["excluded_posts"] = (
        len(df) - len(filtered)
    )

    result["min_reactions"] = min_reactions

    return result


# ============================================================
# 13. RIEPILOGO COMPLETO
# ============================================================

def network_statistics_summary(
    df,
    graph,
    shared_content_groups=None,
    group_column="group_id",
    content_column="shared_content",
    reaction_columns=REACTION_COLUMNS,
    thresholds=REACTION_THRESHOLDS,
    n_permutations=DEFAULT_PERMUTATIONS,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Esegue le analisi statistiche della rete.

    Restituisce:
    - profili di reaction dei gruppi;
    - confronto principale;
    - analisi di sensibilità;
    - eventuale controllo della circolarità.

    Il controllo della circolarità richiede
    il dizionario dei contenuti condivisi.
    """

    profiles = build_group_reaction_profiles(
        df,
        group_column=group_column,
        reaction_columns=reaction_columns,
        min_reactions=0,
    )

    comparison = compare_network_similarities(
        graph,
        profiles,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    sensitivity = network_sensitivity_analysis(
        df,
        graph,
        group_column=group_column,
        reaction_columns=reaction_columns,
        thresholds=thresholds,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    results = {
        "group_profiles": profiles,
        "comparison": comparison,
        "sensitivity": sensitivity,
    }

    if shared_content_groups is not None:

        results["circularity_control"] = (
            circularity_control(
                df,
                graph,
                shared_content_groups,
                group_column=group_column,
                content_column=content_column,
                reaction_columns=reaction_columns,
                n_permutations=n_permutations,
                random_state=random_state,
            )
        )

    return results
