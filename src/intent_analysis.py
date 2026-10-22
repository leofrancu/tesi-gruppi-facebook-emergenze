"""
intent_analysis.py
==================

Analisi descrittiva degli intenti comunicativi
dei post pubblicati nei gruppi Facebook.

Funzioni comunicative:
- INFORM
- INTERPRET
- INSPIRE
- INVESTIGATE

Fasi temporali:
- Pre
- Evento
- Post1
- Post2

Il modulo permette di:

1. Analizzare la distribuzione complessiva degli intenti.
2. Studiare l'evoluzione degli intenti nelle quattro fasi.
3. Calcolare le variazioni tra fasi consecutive.
4. Analizzare i punteggi del classificatore NLI.
5. Confrontare intenti e reaction.
6. Studiare le reaction per combinazione intento-fase.
7. Produrre tabelle riepilogative per l'analisi comparativa.

Il codice è indipendente dall'evento analizzato.

Nota metodologica
-----------------
Le percentuali degli intenti sono calcolate sui post
effettivamente classificati.

Le percentuali delle reaction sono calcolate sui
conteggi aggregati delle reaction, non sulla media
delle percentuali dei singoli post.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

INTENT_ORDER = (
    "INFORM",
    "INTERPRET",
    "INSPIRE",
    "INVESTIGATE",
)

PHASE_ORDER = (
    "Pre",
    "Evento",
    "Post1",
    "Post2",
)

REACTION_COLUMNS = (
    "like",
    "love",
    "wow",
    "haha",
    "sad",
    "angry",
)


# ============================================================
# 2. VALIDAZIONE DEL DATASET
# ============================================================

def validate_columns(df, columns):
    """
    Verifica la presenza delle colonne richieste.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "L'input deve essere un pandas DataFrame."
        )

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonne mancanti: {missing}"
        )

    return True


def validate_intents(
    df,
    intent_column="intent",
):
    """
    Verifica che le etichette non mancanti
    appartengano alle quattro funzioni previste.
    """

    validate_columns(
        df,
        [intent_column],
    )

    valid = set(INTENT_ORDER)

    observed = set(
        df[intent_column].dropna().unique()
    )

    invalid = observed - valid

    if invalid:
        raise ValueError(
            f"Intenti non riconosciuti: {sorted(invalid)}"
        )

    return True


def validate_reactions(
    df,
    reaction_columns=REACTION_COLUMNS,
):
    """
    Verifica che i conteggi delle reaction
    siano numerici, finiti e non negativi.

    Il preprocessing deve essere già stato eseguito.
    """

    validate_columns(
        df,
        reaction_columns,
    )

    for column in reaction_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Valori mancanti o non numerici "
                f"nella colonna '{column}'."
            )

        if not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Valori non finiti nella colonna '{column}'."
            )

        if (values < 0).any():
            raise ValueError(
                f"Conteggi negativi nella colonna '{column}'."
            )

    return True


# ============================================================
# 3. DISTRIBUZIONE COMPLESSIVA DEGLI INTENTI
# ============================================================

def intent_distribution(
    df,
    intent_column="intent",
):
    """
    Calcola frequenze e percentuali degli intenti.

    Il denominatore comprende esclusivamente
    i post con un intento valido.

    Returns
    -------
    pandas.DataFrame
        n_posts, percentage
    """

    validate_intents(
        df,
        intent_column,
    )

    counts = (
        df[intent_column]
        .value_counts()
        .reindex(INTENT_ORDER, fill_value=0)
        .astype(int)
    )

    total = counts.sum()

    if total > 0:
        percentages = counts / total * 100
    else:
        percentages = pd.Series(
            np.nan,
            index=counts.index,
            dtype=float,
        )

    result = pd.DataFrame({
        "n_posts": counts,
        "percentage": percentages,
    })

    result.index.name = "intent"

    return result


# ============================================================
# 4. DISTRIBUZIONE DEGLI INTENTI PER FASE
# ============================================================

def intents_by_phase(
    df,
    phase_column="fase",
    intent_column="intent",
):
    """
    Conta i post classificati per fase e intento.

    Returns
    -------
    pandas.DataFrame
        Righe: fasi temporali.
        Colonne: intenti comunicativi.
    """

    validate_columns(
        df,
        [phase_column],
    )

    validate_intents(
        df,
        intent_column,
    )

    counts = pd.crosstab(
        df[phase_column],
        df[intent_column],
    )

    counts = counts.reindex(
        index=PHASE_ORDER,
        columns=INTENT_ORDER,
        fill_value=0,
    )

    counts.index.name = "fase"
    counts.columns.name = "intent"

    return counts.astype(int)


# ============================================================
# 5. QUOTE DEGLI INTENTI PER FASE
# ============================================================

def intent_shares_by_phase(
    df,
    phase_column="fase",
    intent_column="intent",
):
    """
    Calcola la distribuzione percentuale
    degli intenti all'interno di ogni fase.

    Formula
    -------
    share(intent, fase) =
        post dell'intento nella fase /
        post classificati nella fase * 100

    Ogni fase con post classificati somma a 100%.

    Le fasi senza post classificati
    presentano valori NaN.
    """

    counts = intents_by_phase(
        df,
        phase_column,
        intent_column,
    )

    totals = counts.sum(axis=1)

    shares = (
        counts.div(
            totals.replace(0, np.nan),
            axis=0,
        )
        * 100
    )

    return shares


# ============================================================
# 6. VARIAZIONE DEGLI INTENTI TRA LE FASI
# ============================================================

def intent_share_changes(
    df,
    phase_column="fase",
    intent_column="intent",
):
    """
    Calcola le variazioni delle quote degli intenti
    tra fasi consecutive.

    Le differenze sono espresse
    in punti percentuali.

    Returns
    -------
    pandas.DataFrame
        Variazioni rispetto alla fase precedente.
    """

    shares = intent_shares_by_phase(
        df,
        phase_column,
        intent_column,
    )

    return shares.diff()


# ============================================================
# 7. COPERTURA DELLA CLASSIFICAZIONE
# ============================================================

def classification_coverage(
    df,
    phase_column="fase",
    intent_column="intent",
):
    """
    Calcola quanti post hanno ricevuto
    una classificazione valida.

    La copertura viene calcolata per fase.

    Returns
    -------
    pandas.DataFrame
        n_posts, n_classified, coverage_pct
    """

    validate_columns(
        df,
        [phase_column],
    )

    validate_intents(
        df,
        intent_column,
    )

    valid_phase = df[
        df[phase_column].isin(PHASE_ORDER)
    ].copy()

    total_posts = (
        valid_phase.groupby(
            phase_column,
            observed=True,
        )
        .size()
        .reindex(PHASE_ORDER, fill_value=0)
    )

    classified_posts = (
        valid_phase[
            valid_phase[intent_column].notna()
        ]
        .groupby(
            phase_column,
            observed=True,
        )
        .size()
        .reindex(PHASE_ORDER, fill_value=0)
    )

    coverage = (
        classified_posts
        .div(
            total_posts.replace(0, np.nan)
        )
        * 100
    )

    result = pd.DataFrame({
        "n_posts": total_posts,
        "n_classified": classified_posts,
        "coverage_pct": coverage,
    })

    result.index.name = "fase"

    return result


# ============================================================
# 8. ANALISI DEI PUNTEGGI NLI
# ============================================================

def intent_confidence_summary(
    df,
    intent_column="intent",
    score_column="score_top1",
    margin_column="margin",
):
    """
    Riassume i punteggi del classificatore
    per intento comunicativo.

    Metriche:
    - numero di post classificati;
    - media e mediana del punteggio vincente;
    - media e mediana del margine top1-top2.

    Notes
    -----
    Le metriche sono descrittive.

    Non rappresentano una misura diretta
    dell'accuratezza del classificatore.

    I post con lo stesso testo possono avere
    lo stesso punteggio NLI.
    """

    validate_columns(
        df,
        [
            intent_column,
            score_column,
            margin_column,
        ],
    )

    validate_intents(
        df,
        intent_column,
    )

    classified = df.loc[
        df[intent_column].notna()
    ].copy()

    classified[score_column] = pd.to_numeric(
        classified[score_column],
        errors="coerce",
    )

    classified[margin_column] = pd.to_numeric(
        classified[margin_column],
        errors="coerce",
    )

    grouped = classified.groupby(
        intent_column,
        observed=True,
    )

    result = grouped.agg(
        n_posts=(intent_column, "size"),
        mean_score=(score_column, "mean"),
        median_score=(score_column, "median"),
        mean_margin=(margin_column, "mean"),
        median_margin=(margin_column, "median"),
    )

    result = result.reindex(INTENT_ORDER)

    result["n_posts"] = (
        result["n_posts"]
        .fillna(0)
        .astype(int)
    )

    result.index.name = "intent"

    return result


# ============================================================
# 9. REACTION TOTALI PER INTENTO
# ============================================================

def reactions_by_intent(
    df,
    intent_column="intent",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Aggrega i conteggi delle reaction
    per funzione comunicativa.

    I post senza intento vengono esclusi
    esclusivamente da questa analisi.
    """

    validate_intents(
        df,
        intent_column,
    )

    validate_reactions(
        df,
        reaction_columns,
    )

    classified = df.loc[
        df[intent_column].notna()
    ]

    result = (
        classified.groupby(
            intent_column,
            observed=True,
        )[list(reaction_columns)]
        .sum()
        .reindex(
            INTENT_ORDER,
            fill_value=0,
        )
    )

    result.index.name = "intent"

    return result


# ============================================================
# 10. PROFILI PERCENTUALI DELLE REACTION
# ============================================================

def reaction_profiles_by_intent(
    df,
    intent_column="intent",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Calcola la distribuzione percentuale
    delle reaction per intento.

    Formula
    -------
    share(reaction, intent) =
        somma della reaction nell'intento /
        somma di tutte le reaction nell'intento * 100

    Le percentuali sono pesate
    sui conteggi aggregati.

    Non viene calcolata la media
    delle percentuali dei singoli post.
    """

    totals = reactions_by_intent(
        df,
        intent_column,
        reaction_columns,
    )

    row_totals = totals.sum(axis=1)

    shares = (
        totals.div(
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
    Aggrega le reaction per fase e intento.

    Returns
    -------
    pandas.DataFrame
        Indice multiplo (fase, intento).
        Colonne: sei reaction.
    """

    validate_columns(
        df,
        [phase_column],
    )

    validate_intents(
        df,
        intent_column,
    )

    validate_reactions(
        df,
        reaction_columns,
    )

    classified = df.loc[
        df[intent_column].notna()
        & df[phase_column].isin(PHASE_ORDER)
    ]

    grouped = (
        classified.groupby(
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
            "fase",
            "intent",
        ],
    )

    result = grouped.reindex(
        full_index,
        fill_value=0,
    )

    return result


# ============================================================
# 12. PROFILI REACTION PER INTENTO E FASE
# ============================================================

def reaction_profiles_by_intent_and_phase(
    df,
    phase_column="fase",
    intent_column="intent",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Calcola le quote percentuali delle reaction
    per ogni combinazione fase-intento.

    Ogni combinazione con almeno una reaction
    presenta percentuali che sommano a 100%.

    Le combinazioni senza reaction
    presentano valori NaN.
    """

    totals = reactions_by_intent_and_phase(
        df,
        phase_column,
        intent_column,
        reaction_columns,
    )

    row_totals = totals.sum(axis=1)

    shares = (
        totals.div(
            row_totals.replace(0, np.nan),
            axis=0,
        )
        * 100
    )

    return shares


# ============================================================
# 13. REACTION MEDIE PER POST E INTENTO
# ============================================================

def mean_reactions_by_intent(
    df,
    intent_column="intent",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Calcola il numero medio di reaction
    per post classificato, per intento.

    Returns
    -------
    pandas.DataFrame
        Media delle sei reaction e totale medio.
    """

    totals = reactions_by_intent(
        df,
        intent_column,
        reaction_columns,
    )

    counts = (
        df[intent_column]
        .value_counts()
        .reindex(INTENT_ORDER, fill_value=0)
    )

    means = totals.div(
        counts.replace(0, np.nan),
        axis=0,
    )

    means["total_mean"] = (
        totals.sum(axis=1)
        / counts.replace(0, np.nan)
    )

    means["n_posts"] = counts

    return means


# ============================================================
# 14. TABELLA RIASSUNTIVA INTENTO × FASE
# ============================================================

def intent_phase_summary(
    df,
    phase_column="fase",
    intent_column="intent",
):
    """
    Produce una tabella in formato lungo
    con frequenze e percentuali degli intenti
    per fase temporale.

    Returns
    -------
    pandas.DataFrame
        fase, intent, n_posts, percentage
    """

    counts = intents_by_phase(
        df,
        phase_column,
        intent_column,
    )

    shares = intent_shares_by_phase(
        df,
        phase_column,
        intent_column,
    )

    counts_long = (
        counts.stack()
        .rename("n_posts")
        .reset_index()
    )

    shares_long = (
        shares.stack(dropna=False)
        .rename("percentage")
        .reset_index()
    )

    result = counts_long.merge(
        shares_long,
        on=["fase", "intent"],
        how="left",
        validate="one_to_one",
    )

    return result


# ============================================================
# 15. RIEPILOGO COMPLETO
# ============================================================

def intent_analysis_summary(
    df,
    phase_column="fase",
    intent_column="intent",
    score_column="score_top1",
    margin_column="margin",
):
    """
    Esegue l'analisi descrittiva degli intenti.

    Restituisce un dizionario contenente
    tutte le tabelle disponibili.

    Le analisi dei punteggi vengono eseguite
    soltanto se le relative colonne sono presenti.

    Le analisi delle reaction vengono eseguite
    soltanto se sono disponibili tutte le
    colonne necessarie.
    """

    results = {
        "intent_distribution": (
            intent_distribution(
                df,
                intent_column,
            )
        ),

        "intent_phase_counts": (
            intents_by_phase(
                df,
                phase_column,
                intent_column,
            )
        ),

        "intent_phase_shares": (
            intent_shares_by_phase(
                df,
                phase_column,
                intent_column,
            )
        ),

        "intent_phase_changes": (
            intent_share_changes(
                df,
                phase_column,
                intent_column,
            )
        ),

        "classification_coverage": (
            classification_coverage(
                df,
                phase_column,
                intent_column,
            )
        ),

        "intent_phase_summary": (
            intent_phase_summary(
                df,
                phase_column,
                intent_column,
            )
        ),
    }

    if all(
        column in df.columns
        for column in [
            score_column,
            margin_column,
        ]
    ):

        results["confidence_summary"] = (
            intent_confidence_summary(
                df,
                intent_column,
                score_column,
                margin_column,
            )
        )

    if all(
        column in df.columns
        for column in REACTION_COLUMNS
    ):

        results["reaction_totals_by_intent"] = (
            reactions_by_intent(
                df,
                intent_column,
            )
        )

        results["reaction_profiles_by_intent"] = (
            reaction_profiles_by_intent(
                df,
                intent_column,
            )
        )

        results["reaction_profiles_by_phase_intent"] = (
            reaction_profiles_by_intent_and_phase(
                df,
                phase_column,
                intent_column,
            )
        )

        results["mean_reactions_by_intent"] = (
            mean_reactions_by_intent(
                df,
                intent_column,
            )
        )

    return results
