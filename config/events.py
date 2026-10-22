"""
events.py
=========

Configurazione temporale generica per l'analisi delle emergenze.

Il modulo implementa il protocollo temporale utilizzato nella
ricerca sulle dinamiche comunicative dei gruppi Facebook.

La procedura è indipendente dall'evento analizzato.

Protocollo temporale
--------------------
Pre     : giorni -7 / -1
Evento  : giorni  0 / +1
Post1   : giorni +2 / +7
Post2   : giorni +8 / +14

Le fasi sono definite in giorni di calendario UTC rispetto
alla data di riferimento T0.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""


from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Union

import pandas as pd


# ============================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================

PHASE_ORDER = (
    "Pre",
    "Evento",
    "Post1",
    "Post2",
)

PHASE_OFFSETS = {
    "Pre": (-7, -1),
    "Evento": (0, 1),
    "Post1": (2, 7),
    "Post2": (8, 14),
}

OBSERVATION_START = -7
OBSERVATION_END = 14

DATE_FORMAT = "%Y-%m-%d"


# ============================================================
# 2. STRUTTURA DI CONFIGURAZIONE DI UN EVENTO
# ============================================================

@dataclass(frozen=True)
class EventConfig:
    """
    Configurazione generica di un evento.

    Parameters
    ----------
    name : str
        Nome identificativo dell'evento.

    t0 : str
        Data di riferimento dell'evento in formato YYYY-MM-DD,
        espressa come giorno di calendario UTC.

    warning_time : str, optional
        Timestamp della prima allerta ufficiale, comprensivo
        di fuso orario.

    impact_time : str, optional
        Timestamp dell'impatto utilizzato per calcolare
        il preavviso, comprensivo di fuso orario.

        Può differire da T0 negli eventi a evoluzione
        graduale.

    Notes
    -----
    T0 determina la segmentazione temporale.

    warning_time e impact_time servono esclusivamente
    al calcolo del lead time.
    """

    name: str
    t0: str

    warning_time: Optional[str] = None
    impact_time: Optional[str] = None

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Il nome dell'evento non può essere vuoto.")

        try:
            parsed = date.fromisoformat(self.t0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "T0 deve essere una data valida nel formato YYYY-MM-DD."
            ) from exc

        if parsed.isoformat() != self.t0:
            raise ValueError(
                "T0 deve essere nel formato YYYY-MM-DD."
            )


# ============================================================
# 3. NORMALIZZAZIONE DELLE DATE
# ============================================================

def normalize_t0(t0: Union[str, date, datetime]) -> date:
    """
    Converte T0 in una data di calendario UTC.

    Sono accettati:
    - stringhe YYYY-MM-DD;
    - oggetti date;
    - datetime con fuso orario.

    I datetime senza fuso orario non sono accettati.
    """

    if isinstance(t0, datetime):

        if t0.tzinfo is None or t0.utcoffset() is None:
            raise ValueError(
                "Un T0 datetime deve includere il fuso orario."
            )

        return t0.astimezone(timezone.utc).date()

    if isinstance(t0, date):
        return t0

    if isinstance(t0, str):
        try:
            parsed = date.fromisoformat(t0)

            if parsed.isoformat() != t0:
                raise ValueError

            return parsed

        except ValueError as exc:
            raise ValueError(
                "T0 deve essere nel formato YYYY-MM-DD."
            ) from exc

    raise TypeError(
        "T0 deve essere una stringa, date o datetime."
    )


def normalize_timestamp(timestamp) -> pd.Timestamp:
    """
    Converte un timestamp in UTC.

    Sono accettati timestamp ISO 8601, datetime
    e pandas.Timestamp con fuso orario.

    I timestamp senza fuso orario vengono rifiutati
    per evitare conversioni temporali ambigue.
    """

    if timestamp is None or pd.isna(timestamp):
        raise ValueError("Timestamp mancante.")

    try:
        ts = pd.Timestamp(timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError("Timestamp non valido.") from exc

    if pd.isna(ts):
        raise ValueError("Timestamp mancante.")

    if ts.tzinfo is None:
        raise ValueError(
            "Il timestamp deve includere il fuso orario."
        )

    return ts.tz_convert("UTC")


# ============================================================
# 4. CALCOLO DELLE FINESTRE TEMPORALI
# ============================================================

def get_phase_windows(t0):
    """
    Restituisce le quattro finestre temporali.

    Gli estremi iniziali e finali sono inclusivi.

    Returns
    -------
    dict
        Dizionario contenente le date di inizio
        e fine di ciascuna fase.
    """

    reference_date = normalize_t0(t0)

    windows = {}

    for phase, (start_offset, end_offset) in PHASE_OFFSETS.items():

        windows[phase] = {
            "start": reference_date + timedelta(days=start_offset),
            "end": reference_date + timedelta(days=end_offset),
        }

    return windows


def get_observation_window(t0):
    """
    Restituisce la finestra completa di osservazione.

    L'inizio è incluso.
    La fine è esclusa.

    La fine esclusiva corrisponde a T0 + 15 giorni,
    così da includere interamente il giorno +14.
    """

    reference_date = normalize_t0(t0)

    start_date = (
        reference_date
        + timedelta(days=OBSERVATION_START)
    )

    end_exclusive = (
        reference_date
        + timedelta(days=OBSERVATION_END + 1)
    )

    return {
        "start": start_date,
        "end_exclusive": end_exclusive,
    }


# ============================================================
# 5. ASSEGNAZIONE DELLA FASE TEMPORALE
# ============================================================

def assign_phase(timestamp, t0):
    """
    Assegna un timestamp alla fase temporale corrispondente.

    Parameters
    ----------
    timestamp : str, datetime, pandas.Timestamp
        Data e ora di pubblicazione del post.

    t0 : str, date, datetime
        Data di riferimento dell'emergenza.

    Returns
    -------
    str or None
        Pre, Evento, Post1, Post2 oppure None.

    Notes
    -----
    Il confronto avviene sui giorni di calendario UTC,
    non sulle ore trascorse dall'impatto.
    """

    if timestamp is None or pd.isna(timestamp):
        return None

    ts = normalize_timestamp(timestamp)
    reference_date = normalize_t0(t0)

    days_from_t0 = (
        ts.date() - reference_date
    ).days

    for phase, (start, end) in PHASE_OFFSETS.items():

        if start <= days_from_t0 <= end:
            return phase

    return None


# ============================================================
# 6. ASSEGNAZIONE DELLE FASI A UN DATAFRAME
# ============================================================

def assign_phases_dataframe(
    df,
    t0,
    timestamp_column="dt",
    phase_column="fase",
):
    """
    Assegna le fasi temporali ai post di un DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset contenente i post.

    t0 : str, date, datetime
        Data di riferimento dell'emergenza.

    timestamp_column : str
        Nome della colonna contenente i timestamp.

    phase_column : str
        Nome della colonna da creare.

    Returns
    -------
    pandas.DataFrame
        Copia del dataset con la colonna delle fasi.

    Notes
    -----
    I timestamp validi devono includere il fuso orario.
    I valori mancanti rimangono senza fase.
    """

    if timestamp_column not in df.columns:
        raise KeyError(
            f"Colonna non trovata: {timestamp_column}"
        )

    result = df.copy()

    timestamps = pd.to_datetime(
        result[timestamp_column],
        utc=False,
        errors="raise",
    )

    # Verifica esplicita del fuso orario.
    # Non assumiamo che timestamp privi di timezone siano UTC.
    valid_timestamps = timestamps.dropna()

    if not valid_timestamps.empty:
        for value in valid_timestamps:
            normalize_timestamp(value)

    result[phase_column] = result[timestamp_column].apply(
        lambda value: assign_phase(value, t0)
    )

    result[phase_column] = pd.Categorical(
        result[phase_column],
        categories=list(PHASE_ORDER),
        ordered=True,
    )

    return result


# ============================================================
# 7. CALCOLO DEL LEAD TIME
# ============================================================

def calculate_lead_time(
    warning_time,
    impact_time,
):
    """
    Calcola il preavviso ufficiale in ore.

    Parameters
    ----------
    warning_time : str or datetime
        Momento della prima allerta ufficiale.

    impact_time : str or datetime
        Momento dell'impatto di riferimento.

    Returns
    -------
    float
        Intervallo temporale espresso in ore.

    Notes
    -----
    La funzione calcola una differenza temporale.
    Non stabilisce autonomamente quale allerta o
    quale impatto siano metodologicamente appropriati.

    Per eventi con incertezza temporale, il lead time
    deve essere documentato come intervallo.
    """

    warning = normalize_timestamp(warning_time)
    impact = normalize_timestamp(impact_time)

    lead_time = (
        impact - warning
    ).total_seconds() / 3600

    if lead_time < 0:
        raise ValueError(
            "L'allerta non può essere successiva "
            "all'impatto di riferimento."
        )

    return lead_time


def calculate_lead_time_range(
    warning_start,
    warning_end,
    impact_start,
    impact_end,
):
    """
    Calcola un intervallo di lead time in ore.

    Da utilizzare quando le fonti non permettono
    di stabilire con precisione l'ora dell'allerta
    o dell'impatto.

    Returns
    -------
    tuple
        (lead_time_min, lead_time_max)
    """

    warning_start = normalize_timestamp(warning_start)
    warning_end = normalize_timestamp(warning_end)

    impact_start = normalize_timestamp(impact_start)
    impact_end = normalize_timestamp(impact_end)

    if warning_start > warning_end:
        raise ValueError(
            "Intervallo di allerta non valido."
        )

    if impact_start > impact_end:
        raise ValueError(
            "Intervallo di impatto non valido."
        )

    lead_time_min = (
        impact_start - warning_end
    ).total_seconds() / 3600

    lead_time_max = (
        impact_end - warning_start
    ).total_seconds() / 3600

    if lead_time_min < 0:
        raise ValueError(
            "L'intervallo di allerta deve precedere "
            "quello dell'impatto."
        )

    return lead_time_min, lead_time_max


# ============================================================
# 8. RIEPILOGO DELLA CONFIGURAZIONE
# ============================================================

def describe_event(event: EventConfig):
    """
    Restituisce un riepilogo della configurazione
    temporale dell'evento.
    """

    windows = get_phase_windows(event.t0)
    observation = get_observation_window(event.t0)

    summary = {
        "name": event.name,
        "t0": event.t0,
        "observation_start": observation["start"],
        "observation_end_exclusive": observation["end_exclusive"],
        "phases": windows,
        "lead_time_hours": None,
    }

    if event.warning_time and event.impact_time:

        summary["lead_time_hours"] = calculate_lead_time(
            event.warning_time,
            event.impact_time,
        )

    return summary


# ============================================================
# 9. ESEMPIO GENERICO DI UTILIZZO
# ============================================================

if __name__ == "__main__":

    example_event = EventConfig(
        name="Example event",
        t0="2025-01-15",
    )

    summary = describe_event(example_event)

    print("=" * 60)
    print("EVENT CONFIGURATION")
    print("=" * 60)

    print("Event:", summary["name"])
    print("T0:", summary["t0"])

    print("\nObservation window:")
    print(
        summary["observation_start"],
        "->",
        summary["observation_end_exclusive"],
        "(exclusive)",
    )

    print("\nTemporal phases:")

    for phase, window in summary["phases"].items():

        print(
            f"{phase:7s}: "
            f"{window['start']} -> {window['end']}"
        )

    print("\nExample post classification:")

    example_timestamp = "2025-01-17T12:00:00Z"

    print(
        example_timestamp,
        "->",
        assign_phase(
            example_timestamp,
            example_event.t0,
        ),
    )
