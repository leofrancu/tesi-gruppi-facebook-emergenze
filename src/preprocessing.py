"""
preprocessing.py
================

Funzioni generiche per la preparazione dei post
estratti dalla Meta Content Library.

Il modulo implementa:

1. Verifica delle colonne richieste.
2. Normalizzazione dei testi.
3. Conversione dei timestamp in UTC.
4. Preparazione dei conteggi delle reaction.
5. Identificazione dei testi distinti per la classificazione NLI.
6. Propagazione delle etichette ai post originali.

Principi metodologici
---------------------
- I post originali non vengono deduplicati in base al testo.
- La deduplicazione dei testi riguarda esclusivamente
  l'ottimizzazione della classificazione NLI.
- I post privi di testo non vengono eliminati dal dataset
  destinato alle analisi delle reaction e delle reti.
- I conteggi delle reaction non vengono normalizzati
  individualmente: le quote sono calcolate successivamente
  aggregando le reaction dei post.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""

import re

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================

REACTION_COLUMNS = (
    "like",
    "love",
    "wow",
    "haha",
    "sad",
    "angry",
)

INTENT_COLUMNS = (
    "intent",
    "intent_score",
    "intent_margin",
)


# ============================================================
# 2. VERIFICA DELLE COLONNE
# ============================================================

def validate_columns(df, required_columns):
    """
    Verifica che il DataFrame contenga le colonne richieste.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset da verificare.

    required_columns : iterable
        Nomi delle colonne obbligatorie.

    Returns
    -------
    bool
        True se tutte le colonne sono presenti.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "L'input deve essere un pandas DataFrame."
        )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonne mancanti: {missing}"
        )

    return True


# ============================================================
# 3. NORMALIZZAZIONE DEI TESTI
# ============================================================

def normalize_text(text):
    """
    Normalizza un testo mantenendone il contenuto.

    Operazioni:
    - gestione dei valori mancanti;
    - rimozione degli spazi iniziali e finali;
    - uniformazione degli spazi consecutivi.

    Non vengono effettuate:
    - traduzioni;
    - rimozioni di parole;
    - modifiche alla punteggiatura;
    - conversioni in minuscolo;
    - stemming o lemmatizzazione.

    Returns
    -------
    str or None
        Testo normalizzato oppure None.
    """

    if text is None:
        return None

    if not isinstance(text, str):
        return None

    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not normalized:
        return None

    return normalized


def normalize_text_column(
    df,
    text_column="text",
    output_column="text_normalized",
):
    """
    Aggiunge una colonna contenente i testi normalizzati.

    Il testo originale viene conservato.
    """

    validate_columns(
        df,
        [text_column],
    )

    result = df.copy()

    result[output_column] = (
        result[text_column]
        .apply(normalize_text)
    )

    return result


# ============================================================
# 4. NORMALIZZAZIONE DEI TIMESTAMP
# ============================================================

def normalize_datetime_column(
    df,
    timestamp_column="created_time",
    output_column="dt",
    assume_utc=False,
):
    """
    Converte una colonna temporale in datetime UTC.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset originale.

    timestamp_column : str
        Colonna contenente i timestamp.

    output_column : str
        Colonna temporale da creare.

    assume_utc : bool
        Se True, considera UTC i timestamp privi
        di indicazione del fuso orario.

        Utilizzare questa opzione soltanto quando
        l'origine UTC dei dati è documentata.

    Notes
    -----
    I timestamp non validi vengono convertiti in NaT.

    Se assume_utc=False, i timestamp privi di
    fuso orario generano un errore.
    """

    validate_columns(
        df,
        [timestamp_column],
    )

    result = df.copy()

    values = result[timestamp_column]

    if not assume_utc:

        for value in values.dropna():

            timestamp = pd.Timestamp(value)

            if (
                timestamp.tzinfo is None
                or timestamp.utcoffset() is None
            ):
                raise ValueError(
                    "Sono presenti timestamp senza fuso "
                    "orario. Verificare l'origine dei dati "
                    "oppure impostare assume_utc=True."
                )

    result[output_column] = pd.to_datetime(
        values,
        utc=True,
        errors="coerce",
    )

    return result


# ============================================================
# 5. PREPARAZIONE DELLE REACTION
# ============================================================

def prepare_reactions(
    df,
    reaction_columns=REACTION_COLUMNS,
):
    """
    Prepara le colonne numeriche delle reaction.

    Operazioni:
    - verifica della presenza delle colonne;
    - conversione dei valori in numerici;
    - sostituzione dei valori mancanti con zero;
    - verifica dell'assenza di conteggi negativi.

    Non modifica i valori validi.

    Notes
    -----
    La reaction 'care' non è inclusa nella
    configurazione predefinita, coerentemente
    con l'analisi definitiva della tesi.

    I valori non numerici non mancanti generano
    un errore anziché essere sostituiti con zero.
    """

    validate_columns(
        df,
        reaction_columns,
    )

    result = df.copy()

    for column in reaction_columns:

        original = result[column]

        numeric = pd.to_numeric(
            original,
            errors="coerce",
        )

        invalid = (
            original.notna()
            & numeric.isna()
        )

        if invalid.any():
            raise ValueError(
                f"La colonna '{column}' contiene "
                "valori non numerici."
            )

        numeric = numeric.fillna(0)

        if (numeric < 0).any():
            raise ValueError(
                f"La colonna '{column}' contiene "
                "conteggi negativi."
            )

        if not np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"La colonna '{column}' contiene "
                "valori non finiti."
            )

        result[column] = numeric

    return result


# ============================================================
# 6. TOTALE REACTION PER POST
# ============================================================

def add_total_reactions(
    df,
    reaction_columns=REACTION_COLUMNS,
    output_column="total_reactions",
):
    """
    Calcola il numero totale di reaction per post.

    Il totale comprende esclusivamente
    le reaction specificate.
    """

    validate_columns(
        df,
        reaction_columns,
    )

    result = df.copy()

    result[output_column] = (
        result[list(reaction_columns)]
        .sum(axis=1)
    )

    return result


# ============================================================
# 7. IDENTIFICAZIONE DEI TESTI CLASSIFICABILI
# ============================================================

def get_classifiable_posts(
    df,
    text_column="text_normalized",
):
    """
    Seleziona i post che contengono testo classificabile.

    Non modifica il dataset originale.

    I post privi di testo vengono esclusi soltanto
    dall'insieme destinato alla classificazione NLI.
    """

    validate_columns(
        df,
        [text_column],
    )

    result = df.loc[
        df[text_column].notna()
        & df[text_column].map(
            lambda value: (
                isinstance(value, str)
                and bool(value.strip())
            )
        )
    ].copy()

    return result


# ============================================================
# 8. DEDUPLICAZIONE DEI TESTI PER NLI
# ============================================================

def get_unique_texts(
    df,
    text_column="text_normalized",
):
    """
    Restituisce i testi distinti da classificare.

    La deduplicazione viene eseguita esclusivamente
    sul testo normalizzato.

    Due post differenti con lo stesso testo
    producono una sola inferenza NLI.

    Returns
    -------
    pandas.DataFrame
        DataFrame contenente una riga per testo distinto.
    """

    classifiable = get_classifiable_posts(
        df,
        text_column=text_column,
    )

    unique_texts = (
        classifiable[[text_column]]
        .drop_duplicates(subset=[text_column])
        .reset_index(drop=True)
    )

    return unique_texts


# ============================================================
# 9. PROPAGAZIONE DELLE ETICHETTE NLI
# ============================================================

def merge_intent_predictions(
    df,
    predictions,
    text_column="text_normalized",
):
    """
    Associa le predizioni NLI ai post originali.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset originale con testi normalizzati.

    predictions : pandas.DataFrame
        Risultati della classificazione dei testi distinti.

        Colonne richieste:
        - text_normalized
        - intent
        - intent_score
        - intent_margin

    Returns
    -------
    pandas.DataFrame
        Dataset con le etichette propagate.

    Notes
    -----
    Tutti i post originali vengono conservati.

    I post senza testo classificabile mantengono
    valori mancanti nelle colonne NLI.
    """

    validate_columns(
        df,
        [text_column],
    )

    validate_columns(
        predictions,
        [text_column, *INTENT_COLUMNS],
    )

    duplicated = predictions[
        text_column
    ].duplicated()

    if duplicated.any():
        raise ValueError(
            "Le predizioni contengono testi duplicati."
        )

    existing = [
        column
        for column in INTENT_COLUMNS
        if column in df.columns
    ]

    if existing:
        raise ValueError(
            "Il dataset contiene già colonne di "
            f"classificazione: {existing}"
        )

    if predictions[text_column].isna().any():
        raise ValueError(
            "Le predizioni contengono testi mancanti."
        )

    result = df.merge(
        predictions[
            [text_column, *INTENT_COLUMNS]
        ],
        on=text_column,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    return result


# ============================================================
# 10. RIEPILOGO DELLA PREPARAZIONE
# ============================================================

def preprocessing_summary(
    df,
    text_column="text_normalized",
):
    """
    Produce un riepilogo della disponibilità
    dei testi per la classificazione.
    """

    validate_columns(
        df,
        [text_column],
    )

    total_posts = len(df)

    classifiable = get_classifiable_posts(
        df,
        text_column=text_column,
    )

    unique_texts = get_unique_texts(
        df,
        text_column=text_column,
    )

    n_classifiable = len(classifiable)
    n_unique = len(unique_texts)

    duplicate_text_occurrences = (
        n_classifiable - n_unique
    )

    reduction_pct = (
        100 * duplicate_text_occurrences / n_classifiable
        if n_classifiable > 0
        else 0.0
    )

    return {
        "total_posts": total_posts,
        "classifiable_posts": n_classifiable,
        "posts_without_classifiable_text": (
            total_posts - n_classifiable
        ),
        "unique_texts": n_unique,
        "duplicate_text_occurrences": (
            duplicate_text_occurrences
        ),
        "inference_reduction_pct": round(
            reduction_pct,
            2,
        ),
    }


# ============================================================
# 11. PIPELINE GENERICA
# ============================================================

def preprocess_dataset(
    df,
    text_column="text",
    timestamp_column="created_time",
    normalize_timestamps=True,
    assume_utc=False,
):
    """
    Esegue la preparazione generale del dataset.

    Operazioni:
    1. Normalizzazione dei testi.
    2. Preparazione delle reaction.
    3. Calcolo del totale delle reaction.
    4. Normalizzazione dei timestamp, se richiesta.

    Non vengono eseguite:
    - deduplicazione dei post;
    - assegnazione delle fasi;
    - classificazione NLI;
    - costruzione della rete.

    Queste operazioni appartengono a moduli separati.
    """

    result = normalize_text_column(
        df,
        text_column=text_column,
        output_column="text_normalized",
    )

    result = prepare_reactions(
        result,
    )

    result = add_total_reactions(
        result,
    )

    if normalize_timestamps:

        result = normalize_datetime_column(
            result,
            timestamp_column=timestamp_column,
            output_column="dt",
            assume_utc=assume_utc,
        )

    return result
