"""
temporal_analysis.py
====================

Analisi temporale dei post pubblicati durante le emergenze.

Il modulo implementa la segmentazione in quattro fasi
utilizzata nella ricerca:

    Pre     : T0 - 7  / T0 - 1
    Evento  : T0      / T0 + 1
    Post1   : T0 + 2  / T0 + 7
    Post2   : T0 + 8  / T0 + 14

Le fasi sono definite in giorni di calendario UTC.

Il codice è indipendente dall'evento analizzato
e non richiede l'accesso alla Meta Content Library.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""

import numpy as np
import pandas as pd

try:
    from config.events import (
        PHASE_ORDER,
        PHASE_OFFSETS,
        normalize_t0,
    )
except ModuleNotFoundError as exc:
    if exc.name not in ("config", "config.events"):
        raise

    # Consente anche l'importazione diretta del modulo
    # dalla directory src, senza modificare sys.path.
    from pathlib import Path
    import sys

    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

    from config.events import (
        PHASE_ORDER,
        PHASE_OFFSETS,
        normalize_t0,
    )


# ============================================================
# 1. CONFIGURAZIONE TEMPORALE
# ============================================================

PHASE_DURATIONS = {
    phase: end - start + 1
    for phase, (start, end) in PHASE_OFFSETS.items()
}

OBSERVATION_DAYS = sum(PHASE_DURATIONS.values())


# ============================================================
# 2. NORMALIZZAZIONE DEI TIMESTAMP
# ============================================================

def prepare_timestamps(
    df,
    timestamp_column="dt",
    assume_utc=False,
):
    """
    Prepara i timestamp per l'analisi temporale.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset dei post.

    timestamp_column : str
        Nome della colonna temporale.

    assume_utc : bool
        Se True, interpreta come UTC anche i timestamp
        privi di fuso orario.

        Utilizzare questa opzione soltanto quando
        l'origine UTC dei dati è documentata.

    Returns
    -------
    pandas.Series
        Timestamp convertiti in UTC.

    Notes
    -----
    I timestamp mancanti sono mantenuti come NaT.
    I timestamp non validi generano un errore.
    """

    if timestamp_column not in df.columns:
        raise KeyError(
            f"Colonna temporale non trovata: "
            f"{timestamp_column}"
        )

    values = df[timestamp_column]

    if not assume_utc:

        for value in values.dropna():

            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Timestamp non valido: {value}"
                ) from exc

            if timestamp.tzinfo is None:
                raise ValueError(
                    "Sono presenti timestamp senza "
                    "fuso orario. Specificare "
                    "assume_utc=True soltanto se "
                    "i dati originali sono in UTC."
                )

    timestamps = pd.to_datetime(
        values,
        utc=True,
        errors="raise",
    )

    return timestamps


# ============================================================
# 3. ASSEGNAZIONE VETTORIZZATA DELLE FASI
# ============================================================

def assign_temporal_phases(
    df,
    t0,
    timestamp_column="dt",
    phase_column="fase",
    assume_utc=False,
    keep_outside_window=True,
):
    """
    Assegna i post alle quattro fasi dell'emergenza.

    La procedura utilizza operazioni vettorizzate
    pandas, evitando iterazioni sui singoli post.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset originale.

    t0 : str, date, datetime
        Data di riferimento dell'emergenza.

    timestamp_column : str
        Colonna dei timestamp.

    phase_column : str
        Nome della colonna delle fasi.

    assume_utc : bool
        Consente di interpretare come UTC
        timestamp privi di fuso orario.

    keep_outside_window : bool
        Se True, conserva anche i post esterni
        alla finestra di osservazione, assegnando
        loro una fase mancante.

        Se False, restituisce soltanto i post
        appartenenti alle quattro fasi.

    Returns
    -------
    pandas.DataFrame
        Dataset con la colonna temporale delle fasi.
    """

    result = df.copy()

    timestamps = prepare_timestamps(
        result,
        timestamp_column=timestamp_column,
        assume_utc=assume_utc,
    )

    reference_date = pd.Timestamp(
        normalize_t0(t0),
        tz="UTC",
    )

    # Giorni di calendario UTC rispetto a T0.
    day_offsets = (
        timestamps.dt.normalize() - reference_date
    ).dt.days

    # Inizialmente nessuna fase assegnata.
    phases = pd.Series(
        pd.NA,
        index=result.index,
        dtype="object",
    )

    for phase, (start, end) in PHASE_OFFSETS.items():

        mask = day_offsets.between(
            start,
            end,
            inclusive="both",
        )

        phases.loc[mask] = phase

    result[phase_column] = pd.Categorical(
        phases,
        categories=list(PHASE_ORDER),
        ordered=True,
    )

    if not keep_outside_window:

        result = result.loc[
            result[phase_column].notna()
        ].copy()

    return result


# ============================================================
# 4. CONTEGGIO DEI POST PER FASE
# ============================================================

def posts_by_phase(
    df,
    phase_column="fase",
):
    """
    Calcola il numero di post pubblicati
    in ciascuna fase dell'emergenza.

    Le quattro fasi vengono sempre riportate,
    anche quando non contengono post.

    Returns
    -------
    pandas.DataFrame
        Tabella con conteggi e percentuali.
    """

    if phase_column not in df.columns:
        raise KeyError(
            f"Colonna non trovata: {phase_column}"
        )

    counts = (
        df[phase_column]
        .value_counts()
        .reindex(PHASE_ORDER, fill_value=0)
    )

    counts = counts.astype(int)

    total = counts.sum()

    percentages = (
        counts / total * 100
        if total > 0
        else counts.astype(float)
    )

    result = pd.DataFrame({
        "n_posts": counts,
        "percentage": percentages,
    })

    result.index.name = "fase"

    return result


# ============================================================
# 5. ATTIVITÀ MEDIA GIORNALIERA PER FASE
# ============================================================

def daily_activity_by_phase(
    df,
    phase_column="fase",
):
    """
    Calcola il numero medio di post al giorno
    in ciascuna fase.

    La normalizzazione è necessaria perché
    le quattro fasi hanno durate differenti.

    Durate:
    - Pre: 7 giorni
    - Evento: 2 giorni
    - Post1: 6 giorni
    - Post2: 7 giorni

    Returns
    -------
    pandas.DataFrame
        Conteggi, durata e attività media giornaliera.
    """

    result = posts_by_phase(
        df,
        phase_column=phase_column,
    )

    result["n_days"] = [
        PHASE_DURATIONS[phase]
        for phase in result.index
    ]

    result["posts_per_day"] = (
        result["n_posts"] / result["n_days"]
    )

    return result


# ============================================================
# 6. ANDAMENTO GIORNALIERO DEI POST
# ============================================================

def posts_by_day(
    df,
    t0,
    timestamp_column="dt",
    assume_utc=False,
):
    """
    Calcola il numero di post per ciascun giorno
    della finestra temporale [-7, +14].

    I giorni senza pubblicazioni vengono
    mantenuti con conteggio zero.

    Returns
    -------
    pandas.DataFrame
        Data, giorno relativo a T0,
        fase temporale e numero di post.
    """

    timestamps = prepare_timestamps(
        df,
        timestamp_column=timestamp_column,
        assume_utc=assume_utc,
    )

    reference_date = pd.Timestamp(
        normalize_t0(t0),
        tz="UTC",
    )

    start_offset = min(
        start for start, _ in PHASE_OFFSETS.values()
    )

    end_offset = max(
        end for _, end in PHASE_OFFSETS.values()
    )

    date_range = pd.date_range(
        start=reference_date + pd.Timedelta(
            days=start_offset
        ),
        end=reference_date + pd.Timedelta(
            days=end_offset
        ),
        freq="D",
        tz="UTC",
    )

    counts = (
        timestamps
        .dt.normalize()
        .value_counts()
        .reindex(date_range, fill_value=0)
        .sort_index()
    )

    result = pd.DataFrame({
        "date": date_range,
        "day_from_t0": range(
            start_offset,
            end_offset + 1,
        ),
        "n_posts": counts.to_numpy(dtype=int),
    })

    def phase_from_offset(offset):

        for phase, (start, end) in PHASE_OFFSETS.items():
            if start <= offset <= end:
                return phase

        return None

    result["fase"] = pd.Categorical(
        result["day_from_t0"].map(
            phase_from_offset
        ),
        categories=list(PHASE_ORDER),
        ordered=True,
    )

    return result[
        [
            "date",
            "day_from_t0",
            "fase",
            "n_posts",
        ]
    ]


# ============================================================
# 7. VARIAZIONE DELL'ATTIVITÀ TRA LE FASI
# ============================================================

def phase_activity_changes(
    df,
    phase_column="fase",
):
    """
    Calcola le variazioni dell'attività media
    giornaliera tra fasi consecutive.

    La variazione percentuale è calcolata come:

        (attività_attuale / attività_precedente - 1) * 100

    Quando l'attività precedente è zero,
    la variazione viene lasciata mancante.

    Returns
    -------
    pandas.DataFrame
        Attività media e variazione percentuale.
    """

    activity = daily_activity_by_phase(
        df,
        phase_column=phase_column,
    )

    previous = activity[
        "posts_per_day"
    ].shift(1)

    activity["change_pct"] = (
        (
            activity["posts_per_day"] / previous
            - 1
        ) * 100
    )

    activity.loc[
        previous.isna() | previous.eq(0),
        "change_pct",
    ] = np.nan

    return activity


# ============================================================
# 8. RIEPILOGO TEMPORALE
# ============================================================

def temporal_summary(
    df,
    t0,
    timestamp_column="dt",
    phase_column="fase",
    assume_utc=False,
):
    """
    Esegue l'analisi temporale descrittiva.

    Restituisce:
    - dataset con fasi assegnate;
    - conteggi per fase;
    - attività media giornaliera;
    - andamento giornaliero;
    - variazioni tra fasi.

    Non modifica il DataFrame originale.
    """

    prepared = assign_temporal_phases(
        df,
        t0=t0,
        timestamp_column=timestamp_column,
        phase_column=phase_column,
        assume_utc=assume_utc,
        keep_outside_window=True,
    )

    phase_counts = posts_by_phase(
        prepared,
        phase_column=phase_column,
    )

    daily_activity = daily_activity_by_phase(
        prepared,
        phase_column=phase_column,
    )

    daily_counts = posts_by_day(
        prepared,
        t0=t0,
        timestamp_column=timestamp_column,
        assume_utc=assume_utc,
    )

    changes = phase_activity_changes(
        prepared,
        phase_column=phase_column,
    )

    return {
        "data": prepared,
        "phase_counts": phase_counts,
        "daily_activity": daily_activity,
        "daily_counts": daily_counts,
        "phase_changes": changes,
    }
