"""
statistical_analysis.py
=======================

Procedure statistiche per l'analisi delle reaction
e il confronto quantitativo tra eventi di emergenza.

Funzionalità:
- costruzione della tabella fase × reaction;
- distribuzione percentuale delle reaction;
- test chi-quadro di Pearson;
- V di Cramér;
- residui standardizzati di Pearson;
- correlazione di Spearman tra eventi.

Le analisi sono descrittive e associative.
Non consentono inferenze causali.

Nota metodologica:
le reaction sono conteggi cumulativi rilevati al momento
dell'estrazione dei dati. Le fasi temporali si riferiscono
alla pubblicazione dei post.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, spearmanr


# ============================================================
# CONFIGURAZIONE
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


# ============================================================
# VALIDAZIONE
# ============================================================

def validate_reaction_data(
    df,
    phase_column="fase",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Verifica la presenza delle colonne e la validità
    dei conteggi delle reaction.

    I conteggi devono essere numerici, finiti,
    non negativi e interi.

    I valori mancanti devono essere gestiti
    durante il preprocessing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "L'input deve essere un pandas DataFrame."
        )

    required = [
        phase_column,
        *reaction_columns,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonne mancanti: {missing}"
        )

    invalid_phases = (
        set(df[phase_column].dropna().unique())
        - set(PHASE_ORDER)
    )

    if invalid_phases:
        raise ValueError(
            "Fasi temporali non riconosciute: "
            f"{sorted(invalid_phases)}"
        )

    for column in reaction_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Valori mancanti o non numerici in '{column}'."
            )

        numeric = values.to_numpy(dtype=float)

        if not np.isfinite(numeric).all():
            raise ValueError(
                f"Valori non finiti in '{column}'."
            )

        if (numeric < 0).any():
            raise ValueError(
                f"Conteggi negativi in '{column}'."
            )

        if not np.equal(
            numeric,
            np.floor(numeric),
        ).all():
            raise ValueError(
                f"Conteggi non interi in '{column}'."
            )

    return True


# ============================================================
# TABELLA FASE × REACTION
# ============================================================

def reaction_contingency_table(
    df,
    phase_column="fase",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Costruisce la tabella di contingenza fase × reaction.

    Ogni cella rappresenta il totale delle reaction
    di un determinato tipo associate ai post
    pubblicati nella fase corrispondente.

    Le quattro fasi sono mantenute nell'ordine
    metodologico della ricerca.
    """

    validate_reaction_data(
        df,
        phase_column=phase_column,
        reaction_columns=reaction_columns,
    )

    table = (
        df.loc[df[phase_column].notna()]
        .groupby(
            phase_column,
            observed=True,
        )[list(reaction_columns)]
        .sum()
        .reindex(
            PHASE_ORDER,
            fill_value=0,
        )
    )

    table.index.name = "fase"
    table.columns.name = "reaction"

    return table


# ============================================================
# PERCENTUALI
# ============================================================

def reaction_percentages_by_phase(
    contingency_table,
):
    """
    Calcola la distribuzione percentuale delle reaction
    all'interno di ciascuna fase.

    Formula:
        percentuale = reaction / totale_fase * 100

    Le fasi prive di reaction restituiscono NaN.
    """

    totals = contingency_table.sum(axis=1)

    percentages = (
        contingency_table.div(
            totals.replace(0, np.nan),
            axis=0,
        )
        * 100
    )

    return percentages


# ============================================================
# PREPARAZIONE DEL CHI-QUADRO
# ============================================================

def prepare_chi_square_table(
    contingency_table,
):
    """
    Rimuove righe e colonne con totale zero
    prima del test chi-quadro.

    La tabella descrittiva originale non viene modificata.

    Il test richiede almeno due righe e due colonne
    con frequenze positive.
    """

    if not isinstance(
        contingency_table,
        pd.DataFrame,
    ):
        raise TypeError(
            "È richiesto un pandas DataFrame."
        )

    table = contingency_table.copy()

    values = table.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "La tabella contiene valori non finiti."
        )

    if (values < 0).any():
        raise ValueError(
            "La tabella contiene conteggi negativi."
        )

    if not np.equal(
        values,
        np.floor(values),
    ).all():
        raise ValueError(
            "La tabella contiene conteggi non interi."
        )

    table = table.loc[
        table.sum(axis=1) > 0,
        table.sum(axis=0) > 0,
    ]

    if (
        table.shape[0] < 2
        or table.shape[1] < 2
    ):
        raise ValueError(
            "Il test chi-quadro richiede almeno "
            "due fasi e due categorie di reaction "
            "con frequenze positive."
        )

    return table


# ============================================================
# V DI CRAMÉR
# ============================================================

def cramers_v(
    chi2,
    n,
    n_rows,
    n_columns,
):
    """
    Calcola la V di Cramér.

    V = sqrt(chi2 / (n * min(r - 1, c - 1)))

    La misura descrive l'intensità
    dell'associazione tra le variabili.
    """

    denominator = (
        n * min(
            n_rows - 1,
            n_columns - 1,
        )
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.sqrt(chi2 / denominator)
    )


# ============================================================
# RESIDUI STANDARDIZZATI
# ============================================================

def standardized_residuals(
    observed,
    expected,
):
    """
    Calcola i residui standardizzati di Pearson.

    R = (O - E) / sqrt(E)

    O: frequenza osservata
    E: frequenza attesa

    Valori positivi indicano conteggi superiori
    alle frequenze attese sotto indipendenza.

    Valori negativi indicano conteggi inferiori.

    Non sono residui aggiustati per i margini.
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    expected = np.asarray(
        expected,
        dtype=float,
    )

    if observed.shape != expected.shape:
        raise ValueError(
            "Le matrici devono avere la stessa dimensione."
        )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        residuals = (
            (observed - expected)
            / np.sqrt(expected)
        )

    residuals = np.where(
        np.isfinite(residuals),
        residuals,
        np.nan,
    )

    return residuals


# ============================================================
# ANALISI CHI-QUADRO
# ============================================================

def chi_square_reaction_analysis(
    df,
    phase_column="fase",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Analizza l'associazione tra fase temporale
    e tipo di reaction.

    Restituisce:
    - tabella di contingenza;
    - distribuzioni percentuali;
    - chi-quadro;
    - gradi di libertà;
    - valore p;
    - V di Cramér;
    - frequenze attese;
    - residui standardizzati.

    ATTENZIONE:
    le reaction associate allo stesso post o gruppo
    non sono necessariamente indipendenti.

    Il valore p del chi-quadro deve quindi essere
    interpretato con cautela.
    """

    contingency = reaction_contingency_table(
        df,
        phase_column=phase_column,
        reaction_columns=reaction_columns,
    )

    percentages = reaction_percentages_by_phase(
        contingency
    )

    try:
        test_table = prepare_chi_square_table(
            contingency
        )

    except ValueError:
        return {
            "contingency": contingency,
            "percentages": percentages,
            "test_table": None,
            "chi2": np.nan,
            "p_value": np.nan,
            "dof": np.nan,
            "cramers_v": np.nan,
            "n_reactions": int(
                contingency.to_numpy().sum()
            ),
            "expected": None,
            "residuals": None,
            "test_available": False,
        }

    chi2, p_value, dof, expected = (
        chi2_contingency(
            test_table.to_numpy(),
            correction=False,
        )
    )

    n_reactions = int(
        test_table.to_numpy().sum()
    )

    effect_size = cramers_v(
        chi2=chi2,
        n=n_reactions,
        n_rows=test_table.shape[0],
        n_columns=test_table.shape[1],
    )

    expected_df = pd.DataFrame(
        expected,
        index=test_table.index,
        columns=test_table.columns,
    )

    residuals_df = pd.DataFrame(
        standardized_residuals(
            test_table.to_numpy(),
            expected,
        ),
        index=test_table.index,
        columns=test_table.columns,
    )

    return {
        "contingency": contingency,
        "percentages": percentages,
        "test_table": test_table,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "cramers_v": effect_size,
        "n_reactions": n_reactions,
        "expected": expected_df,
        "residuals": residuals_df,
        "test_available": True,
    }


# ============================================================
# PREPARAZIONE DEL CONFRONTO TRA EVENTI
# ============================================================

def prepare_event_comparison(
    df,
    lead_time_column="lead_time_hours",
    indicator_column="indicator",
    event_column="event",
):
    """
    Prepara un dataset per il confronto tra eventi.

    Ogni riga rappresenta un evento.

    Il lead time deve essere espresso in ore.

    Le osservazioni prive di lead time
    o indicatore valido vengono escluse
    soltanto dal confronto richiesto.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "L'input deve essere un pandas DataFrame."
        )

    required = [
        event_column,
        lead_time_column,
        indicator_column,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonne mancanti: {missing}"
        )

    data = df[required].copy()

    if data[event_column].isna().any():
        raise ValueError(
            "Sono presenti eventi senza identificativo."
        )

    if data[event_column].duplicated().any():
        raise ValueError(
            "Ogni evento deve comparire una sola volta."
        )

    for column in (
        lead_time_column,
        indicator_column,
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        data[column] = data[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    data = (
        data.dropna(
            subset=[
                lead_time_column,
                indicator_column,
            ]
        )
        .reset_index(drop=True)
    )

    if (data[lead_time_column] < 0).any():
        raise ValueError(
            "Il lead time non può essere negativo."
        )

    return data


# ============================================================
# CORRELAZIONE DI SPEARMAN
# ============================================================

def spearman_event_correlation(
    df,
    lead_time_column="lead_time_hours",
    indicator_column="indicator",
    event_column="event",
):
    """
    Calcola la correlazione di Spearman
    tra lead time e indicatore quantitativo.

    Il coefficiente rho misura una relazione monotona.

    Non implica causalità.

    Con un numero limitato di eventi,
    il valore p deve essere interpretato
    con particolare cautela.
    """

    data = prepare_event_comparison(
        df,
        lead_time_column=lead_time_column,
        indicator_column=indicator_column,
        event_column=event_column,
    )

    n_events = len(data)

    if n_events < 3:
        return {
            "n_events": n_events,
            "rho": np.nan,
            "p_value": np.nan,
            "data": data,
        }

    x = data[
        lead_time_column
    ].to_numpy(dtype=float)

    y = data[
        indicator_column
    ].to_numpy(dtype=float)

    if (
        np.unique(x).size < 2
        or np.unique(y).size < 2
    ):
        return {
            "n_events": n_events,
            "rho": np.nan,
            "p_value": np.nan,
            "data": data,
        }

    result = spearmanr(x, y)

    return {
        "n_events": n_events,
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "data": data,
    }


# ============================================================
# ANALISI LEAVE-ONE-OUT
# ============================================================

def spearman_leave_one_out(
    df,
    lead_time_column="lead_time_hours",
    indicator_column="indicator",
    event_column="event",
):
    """
    Ripete la correlazione di Spearman
    escludendo un evento alla volta.

    Si tratta di un controllo aggiuntivo
    della sensibilità ai singoli eventi.

    Non aumenta la numerosità del campione.
    """

    data = prepare_event_comparison(
        df,
        lead_time_column=lead_time_column,
        indicator_column=indicator_column,
        event_column=event_column,
    )

    results = []

    for event in data[event_column]:

        subset = data.loc[
            data[event_column] != event
        ].copy()

        correlation = spearman_event_correlation(
            subset,
            lead_time_column=lead_time_column,
            indicator_column=indicator_column,
            event_column=event_column,
        )

        results.append({
            "excluded_event": event,
            "n_events": correlation["n_events"],
            "rho": correlation["rho"],
            "p_value": correlation["p_value"],
        })

    return pd.DataFrame(
        results,
        columns=[
            "excluded_event",
            "n_events",
            "rho",
            "p_value",
        ],
    )


# ============================================================
# CONFRONTO DI PIÙ INDICATORI
# ============================================================

def compare_multiple_indicators(
    df,
    indicator_columns,
    lead_time_column="lead_time_hours",
    event_column="event",
):
    """
    Calcola la correlazione di Spearman
    tra lead time e diversi indicatori quantitativi.

    Ogni indicatore viene analizzato separatamente.

    Non applica automaticamente correzioni
    per confronti multipli.
    """

    results = []

    for indicator in indicator_columns:

        correlation = spearman_event_correlation(
            df,
            lead_time_column=lead_time_column,
            indicator_column=indicator,
            event_column=event_column,
        )

        results.append({
            "indicator": indicator,
            "n_events": correlation["n_events"],
            "rho": correlation["rho"],
            "p_value": correlation["p_value"],
        })

    return pd.DataFrame(
        results,
        columns=[
            "indicator",
            "n_events",
            "rho",
            "p_value",
        ],
    )


# ============================================================
# RIEPILOGO DEL SINGOLO EVENTO
# ============================================================

def event_statistical_summary(
    df,
    phase_column="fase",
    reaction_columns=REACTION_COLUMNS,
):
    """
    Esegue l'analisi statistica delle reaction
    per un singolo evento.
    """

    return chi_square_reaction_analysis(
        df,
        phase_column=phase_column,
        reaction_columns=reaction_columns,
    )


# ============================================================
# RIEPILOGO COMPARATIVO
# ============================================================

def comparative_statistical_summary(
    df,
    indicator_columns,
    lead_time_column="lead_time_hours",
    event_column="event",
    include_leave_one_out=False,
):
    """
    Esegue le correlazioni tra eventi.

    Il controllo leave-one-out è facoltativo
    e disattivato per impostazione predefinita.
    """

    correlations = compare_multiple_indicators(
        df,
        indicator_columns=indicator_columns,
        lead_time_column=lead_time_column,
        event_column=event_column,
    )

    results = {
        "correlations": correlations,
    }

    if include_leave_one_out:

        leave_one_out = {}

        for indicator in indicator_columns:

            leave_one_out[indicator] = (
                spearman_leave_one_out(
                    df,
                    lead_time_column=lead_time_column,
                    indicator_column=indicator,
                    event_column=event_column,
                )
            )

        results["leave_one_out"] = leave_one_out

    return results
