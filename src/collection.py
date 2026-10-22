"""
collection.py
=============

Documentazione e validazione delle configurazioni
di raccolta dati dalla Meta Content Library.

Progetto:
Dinamiche emotive e reti di condivisione nei gruppi
Facebook durante le emergenze.

Metodologia
-----------
La raccolta originale è stata effettuata mediante:

1. Individuazione delle parole chiave pertinenti
   agli eventi, anche attraverso Google Trends.

2. Definizione delle interrogazioni nella
   Meta Content Library.

3. Selezione dei gruppi Facebook pubblici.

4. Definizione degli intervalli temporali
   relativi agli eventi.

5. Esecuzione delle interrogazioni nel
   Meta Secure Research Environment (SRE).

6. Elaborazione dei risultati nell'ambiente
   di ricerca autorizzato.

Questo modulo NON esegue interrogazioni Meta.

Non contiene credenziali, token di accesso,
dati personali o contenuti soggetti alle
restrizioni di accesso della piattaforma.

Le funzioni permettono di documentare e
validare le configurazioni delle interrogazioni.

Non sostituiscono il codice di raccolta
generato dalla Meta Content Library.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional


# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

DEFAULT_PLATFORM = "Facebook"

DEFAULT_CONTENT_TYPE = "public_groups"

DEFAULT_TIMEZONE = "UTC"

QUERY_SEPARATOR = "|"


# ============================================================
# 2. CONFIGURAZIONE DI UNA RACCOLTA
# ============================================================

@dataclass
class CollectionConfig:
    """
    Configurazione documentale di una raccolta.

    Parameters
    ----------
    event_name : str
        Nome dell'evento.

    query : str
        Espressione di ricerca utilizzata
        nella Meta Content Library.

    start_date : str
        Inizio dell'intervallo temporale.

    end_date : str
        Fine dell'intervallo temporale.

    platform : str
        Piattaforma analizzata.

    content_type : str
        Tipologia di contenuti selezionati.

    timezone : str
        Fuso orario di riferimento.

    notes : str, optional
        Informazioni metodologiche aggiuntive.
    """

    event_name: str
    query: str
    start_date: str
    end_date: str

    platform: str = DEFAULT_PLATFORM
    content_type: str = DEFAULT_CONTENT_TYPE
    timezone: str = DEFAULT_TIMEZONE

    notes: Optional[str] = None


# ============================================================
# 3. NORMALIZZAZIONE DEI TERMINI DI RICERCA
# ============================================================

def normalize_query_terms(terms):
    """
    Prepara una lista di termini di ricerca.

    Operazioni:
    - rimozione degli spazi esterni;
    - eliminazione dei termini vuoti;
    - eliminazione dei duplicati esatti;
    - conservazione dell'ordine originale.

    Non modifica:
    - lingua;
    - accenti;
    - punteggiatura interna;
    - struttura delle espressioni.

    Returns
    -------
    list[str]
        Termini normalizzati.
    """

    if isinstance(terms, str):
        raise TypeError(
            "terms deve essere una sequenza "
            "di stringhe, non una singola stringa."
        )

    if terms is None:
        raise ValueError(
            "La lista dei termini non può essere None."
        )

    normalized = []
    seen = set()

    for term in terms:

        if not isinstance(term, str):
            raise TypeError(
                "Tutti i termini devono essere stringhe."
            )

        value = term.strip()

        if not value:
            continue

        if QUERY_SEPARATOR in value:
            raise ValueError(
                "Ogni elemento deve contenere un solo "
                "termine o espressione di ricerca. "
                "Il separatore '|' viene aggiunto "
                "automaticamente."
            )

        if value not in seen:
            normalized.append(value)
            seen.add(value)

    return normalized


# ============================================================
# 4. COSTRUZIONE DOCUMENTALE DELLA QUERY
# ============================================================

def build_query(terms):
    """
    Costruisce una query separando i termini con '|'.

    La funzione riproduce la convenzione testuale
    utilizzata per documentare le interrogazioni
    della ricerca.

    Non verifica la sintassi corrente delle API Meta.

    Non esegue interrogazioni.

    Parameters
    ----------
    terms : iterable[str]
        Termini o espressioni di ricerca.

    Returns
    -------
    str
        Query composta.
    """

    normalized = normalize_query_terms(terms)

    if not normalized:
        raise ValueError(
            "È necessario specificare almeno "
            "un termine di ricerca."
        )

    return QUERY_SEPARATOR.join(normalized)


# ============================================================
# 5. ESTRAZIONE DEI TERMINI DA UNA QUERY
# ============================================================

def split_query(query):
    """
    Divide una query nei termini separati da '|'.

    Questa funzione è destinata alla documentazione
    delle query della ricerca.

    Non interpreta operatori avanzati o parentesi.
    """

    if not isinstance(query, str):
        raise TypeError(
            "La query deve essere una stringa."
        )

    terms = query.split(QUERY_SEPARATOR)

    return normalize_query_terms(terms)


# ============================================================
# 6. VALIDAZIONE DELLE DATE
# ============================================================

def parse_collection_date(value):
    """
    Converte una data ISO 8601 in datetime.

    Formati supportati:
    - YYYY-MM-DD
    - YYYY-MM-DDTHH:MM:SS
    - YYYY-MM-DDTHH:MM:SS+HH:MM
    - YYYY-MM-DDTHH:MM:SSZ

    Le date senza orario rappresentano
    una data di calendario, non un istante
    temporale già normalizzato in UTC.
    """

    if not isinstance(value, str):
        raise TypeError(
            "La data deve essere una stringa ISO 8601."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            "La data non può essere vuota."
        )

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

    except ValueError as exc:
        raise ValueError(
            f"Data non valida: {value}"
        ) from exc


# ============================================================
# 7. VALIDAZIONE DELLA CONFIGURAZIONE
# ============================================================

def validate_collection_config(config):
    """
    Verifica la completezza e la coerenza
    di una configurazione di raccolta.

    Controlla:
    - nome dell'evento;
    - query non vuota;
    - date valide;
    - ordine cronologico;
    - presenza dei metadati principali.

    Non verifica:
    - disponibilità dei dati Meta;
    - autorizzazioni dell'utente;
    - numero di risultati;
    - sintassi API della piattaforma.
    """

    if not isinstance(config, CollectionConfig):
        raise TypeError(
            "config deve essere un'istanza "
            "di CollectionConfig."
        )

    if not isinstance(config.event_name, str):
        raise TypeError(
            "event_name deve essere una stringa."
        )

    if not config.event_name.strip():
        raise ValueError(
            "Il nome dell'evento è obbligatorio."
        )

    if not isinstance(config.query, str):
        raise TypeError(
            "La query deve essere una stringa."
        )

    if not config.query.strip():
        raise ValueError(
            "La query non può essere vuota."
        )

    terms = split_query(config.query)

    if not terms:
        raise ValueError(
            "La query non contiene termini validi."
        )

    start = parse_collection_date(
        config.start_date
    )

    end = parse_collection_date(
        config.end_date
    )

    if (start.tzinfo is None) != (end.tzinfo is None):
        raise ValueError(
            "Le date devono essere entrambe "
            "con fuso orario oppure entrambe "
            "senza fuso orario."
        )

    if start > end:
        raise ValueError(
            "La data iniziale deve precedere "
            "o coincidere con quella finale."
        )

    for field_name in (
        "platform",
        "content_type",
        "timezone",
    ):

        value = getattr(config, field_name)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Il campo '{field_name}' "
                "deve contenere una stringa non vuota."
            )

    return True


# ============================================================
# 8. CREAZIONE DI UNA CONFIGURAZIONE
# ============================================================

def create_collection_config(
    event_name,
    terms,
    start_date,
    end_date,
    platform=DEFAULT_PLATFORM,
    content_type=DEFAULT_CONTENT_TYPE,
    timezone=DEFAULT_TIMEZONE,
    notes=None,
):
    """
    Crea e valida una configurazione di raccolta.

    La query viene costruita dai termini forniti.

    La configurazione risultante è un documento
    metodologico, non una richiesta API.
    """

    query = build_query(terms)

    config = CollectionConfig(
        event_name=event_name,
        query=query,
        start_date=start_date,
        end_date=end_date,
        platform=platform,
        content_type=content_type,
        timezone=timezone,
        notes=notes,
    )

    validate_collection_config(config)

    return config


# ============================================================
# 9. ESPORTAZIONE DEI METADATI
# ============================================================

def collection_config_to_dict(config):
    """
    Restituisce la configurazione come dizionario.

    Il risultato può essere utilizzato per
    documentare la procedura di raccolta.

    Non contiene dati raccolti dalla piattaforma.
    """

    validate_collection_config(config)

    return asdict(config)


# ============================================================
# 10. RIEPILOGO DELLA CONFIGURAZIONE
# ============================================================

def collection_summary(config):
    """
    Produce un riepilogo della configurazione.

    Returns
    -------
    dict
        Metadati e numero di termini di ricerca.
    """

    validate_collection_config(config)

    terms = split_query(config.query)

    return {
        "event_name": config.event_name,
        "platform": config.platform,
        "content_type": config.content_type,
        "query": config.query,
        "n_query_terms": len(terms),
        "start_date": config.start_date,
        "end_date": config.end_date,
        "timezone": config.timezone,
        "notes": config.notes,
        "collection_executed": False,
    }


# ============================================================
# 11. LIMITI OPERATIVI
# ============================================================

def explain_collection_requirements():
    """
    Descrive i requisiti per ripetere
    la raccolta originale.

    Non avvia alcuna operazione sulla piattaforma.
    """

    return {
        "data_source": "Meta Content Library",
        "execution_environment": (
            "Meta Secure Research Environment"
        ),
        "target_content": "Public Facebook groups",
        "query_definition": (
            "Meta Content Library interface"
        ),
        "execution": (
            "Platform-generated code executed "
            "in the authorized research environment"
        ),
        "access_required": True,
        "api_calls_implemented": False,
        "restricted_data_included": False,
    }
