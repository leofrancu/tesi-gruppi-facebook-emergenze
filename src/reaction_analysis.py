"""
reaction_analysis.py
====================

Analisi delle reaction nei post dei gruppi Facebook.

Il modulo implementa:

1. Aggregazione dei conteggi delle reaction.
2. Distribuzione percentuale pesata delle reaction.
3. Analisi delle reaction per fase temporale.
4. Numero medio di reaction per post.
5. Evoluzione delle reaction tra le fasi.
6. Distribuzione delle reaction per intento comunicativo.
7. Distribuzione delle reaction per intento e fase.

Il codice è indipendente dall'evento analizzato.

Nota metodologica
-----------------
Le percentuali delle reaction vengono calcolate
sui conteggi aggregati, non come media delle
percentuali dei singoli post.

Le reaction rappresentano i conteggi disponibili
al momento dell'estrazione dei dati e non
necessariamente le reaction ricevute durante
la fase temporale di pubblicazione del post.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""

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

PHASE_ORDER = (
    "Pre",
    "Evento",
    "Post1",
    "Post2",
)

INTENT_ORDER = (
    "INFORM",
    "INTERPRET",
    "INSPIRE",
    "INVESTIGATE",
)


# ============================================================
# 2. VERIFICA DEL DATASET
# ============================================================

def validate_reaction_data(
    df,
    reaction_columns=REACTION_COLUMNS,
):
    """
    Verifica la presenza e la validità
    delle colonne delle reaction.

    I valori devono essere numerici,
    finiti e non negativi.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "L'input deve essere un pandas DataFrame."
        )

    missing = [
        col
        for col in reaction_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonne reaction mancanti: {missing}"
        )

    for col in reaction_columns:

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"La colonna '{col}' contiene "
                "valori mancanti o non numerici. "
                "Eseguire prima il preprocessing."
            )

        if not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"La colonna '{col}' contiene "
                "valori non finiti."
            )

        if (values < 0).any():
            raise ValueError(
                f"La colonna '{col}' contiene "
                "conteggi negativi."
            )

    return True


# ============================================================
# 3. AGGREGAZIONE DELLE REACTION
# ============================================================

def aggregate_reactions(
    df,
    reaction_columns=REACTION_COLUMNS,
):
    """
    Somma i conteggi delle reaction.

    Returns
    -------
    pandas.Series
        Totale per ciascuna reaction.
    """

    validate_reaction_data(
        df,
        reaction_columns,
    )

    totals = (
        df[list(reaction_columns)]
        .sum()
        .reindex(reaction_columns)
    )

    return totals


# ============================================================
# 4. DISTRIBUZIONE PERCENTUALE PESATA
# ============================================================

def reaction_distribution(
    df,
    reaction_columns=REACTION_COLUMNS,
):
    """
    Calcola la distribuzione percentuale
    pesata delle reaction.

    Formula
    -------
    share(r) = somma reaction(r) /
               somma di tutte le reaction

    Returns
    -------
    pandas.DataFrame
        Conteggi e percentuali per reaction.

    Notes
    -----
    Se il totale delle reaction è zero,
    le percentuali vengono restituite come NaN.
    """

    totals = aggregate_reactions(
        df,
        reaction_columns,
    )

    grand_total = totals.sum()

    if grand_total > 0:

        percentages = (
            totals / grand_total * 100
        )

    else:

        percentages = pd.Series(
            np.nan,
            index=totals.index,
            dtype=float,
        )

    result = pd.DataFrame({
        "n_reactions": totals,
        "percentage": percentages,
    })

    result.index.name = "reaction"

    return result


# ============================================================
# 5. REACTION TOTALI PER FASE
# ============================================================

def reactions_by_phase(
    df,
    phase_column="fase",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Aggrega le reaction nelle quattro
    fasi temporali dell'emergenza.

    Returns
    -------
    pandas.DataFrame
        Conteggi delle sei reaction per fase.
    """

    validate_reaction_data(
        df,
        reaction_columns,
    )

    if phase_column not in df.columns:
        raise KeyError(
            f"Colonna mancante: {phase_column}"
        )

    grouped = (
        df.groupby(
            phase_column,
            observed=True,
        )[list(reaction_columns)]
        .sum()
        .reindex(
            PHASE_ORDER,
            fill_value=0,
        )
    )

    grouped.index.name = "fase"

    return grouped


# ============================================================
# 6. DISTRIBUZIONE DELLE REACTION PER FASE
# ============================================================

def reaction_shares_by_phase(
    df,
    phase_column="fase",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Calcola la distribuzione percentuale
    delle reaction per ciascuna fase.

    Le percentuali vengono calcolate
    a partire dai conteggi aggregati.

    Ogni fase con almeno una reaction
    presenta percentuali che sommano a 100.
    """

    totals = reactions_by_phase(
        df,
        phase_column,
        reaction_columns,
    )

    row_totals = totals.sum(axis=1)

    shares = (
        totals
        .div(
            row_totals.replace(0, np.nan),
            axis=0,
        )
        * 100
    )

    return shares


# ============================================================
# 7. REACTION MEDIE PER POST
# ============================================================

def mean_reactions_per_post(
    df,
    phase_column="fase",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Calcola il numero medio di reaction
    per post in ciascuna fase.

    Formula
    -------
    media = reaction totali / numero di post

    Questa misura descrive il volume medio
    di interazioni associato ai post.
    """

    totals = reactions_by_phase(
        df,
        phase_column,
        reaction_columns,
    )

    post_counts = (
        df[phase_column]
        .value_counts()
        .reindex(
            PHASE_ORDER,
            fill_value=0,
        )
    )

    means = totals.div(
        post_counts.replace(0, np.nan),
        axis=0,
    )

    means["total_mean"] = (
        totals.sum(axis=1)
        / post_counts.replace(0, np.nan)
    )

    means["n_posts"] = post_counts

    return means


# ============================================================
# 8. EVOLUZIONE DELLE REACTION TRA LE FASI
# ============================================================

def reaction_share_changes(
    df,
    phase_column="fase",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Calcola le variazioni in punti percentuali
    delle quote di reaction tra fasi consecutive.

    Formula
    -------
    delta = quota_fase_attuale -
            quota_fase_precedente

    Returns
    -------
    pandas.DataFrame
        Variazioni in punti percentuali.

    Notes
    -----
    La prima fase non ha una fase precedente,
    quindi presenta valori NaN.
    """

    shares = reaction_shares_by_phase(
        df,
        phase_column,
        reaction_columns,
    )

    return shares.diff()


# ============================================================
# 9. REACTION PER INTENTO COMUNICATIVO
# ============================================================

def reactions_by_intent(
    df,
    intent_column="intent",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Aggrega le reaction per funzione comunicativa.

    Sono considerate le quattro categorie:
    INFORM, INTERPRET, INSPIRE, INVESTIGATE.

    I post privi di classificazione non vengono
    inclusi in questa specifica aggregazione.
    """

    validate_reaction_data(
        df,
        reaction_columns,
    )

    if intent_column not in df.columns:
        raise KeyError(
            f"Colonna mancante: {intent_column}"
        )

    grouped = (
        df.groupby(
            intent_column,
            observed=True,
        )[list(reaction_columns)]
        .sum()
        .reindex(
            INTENT_ORDER,
            fill_value=0,
        )
    )

    grouped.index.name = "intent"

    return grouped


# ============================================================
# 10. DISTRIBUZIONE DELLE REACTION PER INTENTO
# ============================================================

def reaction_shares_by_intent(
    df,
    intent_column="intent",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Calcola la distribuzione percentuale
    delle reaction per intento comunicativo.
    """

    totals = reactions_by_intent(
        df,
        intent_column,
        reaction_columns,
    )

    row_totals = totals.sum(axis=1)

    shares = (
        totals
        .div(
            row_totals.replace(0, np.nan),
            axis=0,
        )
        * 100
    )

    return shares


# ============================================================
# 11. REACTION PER INTENTO E FASE
# ============================================================

def reactions_by_intent_and_phase(
    df,
    phase_column="fase",
    intent_column="intent",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Aggrega le reaction per combinazione
    di fase temporale e intento comunicativo.

    Returns
    -------
    pandas.DataFrame
        Tabella con indice multiplo:
        (fase, intent).
    """

    validate_reaction_data(
        df,
        reaction_columns,
    )

    for column in (
        phase_column,
        intent_column,
    ):

        if column not in df.columns:
            raise KeyError(
                f"Colonna mancante: {column}"
            )

    grouped = (
        df.groupby(
            [phase_column, intent_column],
            observed=True,
        )[list(reaction_columns)]
        .sum()
    )

    full_index = pd.MultiIndex.from_product(
        [
            PHASE_ORDER,
            INTENT_ORDER,
        ],
        names=[
            phase_column,
            intent_column,
        ],
    )

    grouped = grouped.reindex(
        full_index,
        fill_value=0,
    )

    return grouped


# ============================================================
# 12. QUOTE REACTION PER INTENTO E FASE
# ============================================================

def reaction_shares_by_intent_and_phase(
    df,
    phase_column="fase",
    intent_column="intent",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Calcola la distribuzione percentuale
    delle reaction all'interno di ogni
    combinazione fase-intento.

    Ogni combinazione con reaction positive
    somma a 100%.
    """

    totals = reactions_by_intent_and_phase(
        df,
        phase_column,
        intent_column,
        reaction_columns,
    )

    row_totals = totals.sum(axis=1)

    shares = (
        totals
        .div(
            row_totals.replace(0, np.nan),
            axis=0,
        )
        * 100
    )

    return shares


# ============================================================
# 13. RIEPILOGO COMPLETO
# ============================================================

def reaction_analysis_summary(
    df,
    phase_column="fase",
    intent_column="intent",
):
    """
    Esegue l'analisi descrittiva delle reaction.

    Returns
    -------
    dict
        Dizionario contenente le tabelle
        prodotte dalle funzioni precedenti.

    Le analisi per intento vengono eseguite
    soltanto se la colonna è disponibile.
    """

    results = {
        "overall_distribution": (
            reaction_distribution(df)
        ),

        "phase_totals": (
            reactions_by_phase(
                df,
                phase_column,
            )
        ),

        "phase_shares": (
            reaction_shares_by_phase(
                df,
                phase_column,
            )
        ),

        "mean_reactions_per_post": (
            mean_reactions_per_post(
                df,
                phase_column,
            )
        ),

        "phase_share_changes": (
            reaction_share_changes(
                df,
                phase_column,
            )
        ),
    }

    if intent_column in df.columns:

        results["intent_totals"] = (
            reactions_by_intent(
                df,
                intent_column,
            )
        )

        results["intent_shares"] = (
            reaction_shares_by_intent(
                df,
                intent_column,
            )
        )

        results["intent_phase_totals"] = (
            reactions_by_intent_and_phase(
                df,
                phase_column,
                intent_column,
            )
        )

        results["intent_phase_shares"] = (
            reaction_shares_by_intent_and_phase(
                df,
                phase_column,
                intent_column,
            )
        )

    return results
