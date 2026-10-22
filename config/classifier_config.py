"""
classifier_config.py
====================

Configurazione del classificatore zero-shot NLI
utilizzato per l'analisi degli intenti comunicativi.

Il modulo contiene le impostazioni generali condivise
dalle analisi, senza vincolarle a uno specifico evento.

Le ipotesi NLI possono essere definite separatamente
in fase di esecuzione, preservando le formulazioni
effettivamente adottate nei diversi dataset.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""

from dataclasses import dataclass
from typing import Mapping, Optional


# ============================================================
# 1. MODELLO NLI
# ============================================================

MODEL_NAME = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)

TASK = "zero-shot-classification"

# Il modello è utilizzato senza fine-tuning.
FINE_TUNING = False


# ============================================================
# 2. FUNZIONI COMUNICATIVE
# ============================================================

INTENT_ORDER = (
    "INFORM",
    "INTERPRET",
    "INSPIRE",
    "INVESTIGATE",
)

INTENT_DEFINITIONS = {
    "INFORM": (
        "Riportare fatti, aggiornamenti, avvisi, dati "
        "o indicazioni pratiche sull'emergenza."
    ),
    "INTERPRET": (
        "Spiegare o commentare cause, conseguenze, "
        "contesto e significato dell'emergenza."
    ),
    "INSPIRE": (
        "Esprimere solidarietà, sostegno, speranza, "
        "gratitudine o ricordo, oppure invitare "
        "ad aiutare e a mobilitarsi."
    ),
    "INVESTIGATE": (
        "Sollevare domande, critiche o richieste "
        "di responsabilità su decisioni, problemi "
        "o mancanze legati all'emergenza."
    ),
}


# ============================================================
# 3. PARAMETRI GENERALI
# ============================================================

DEFAULT_MAX_LENGTH = 256

# Valore utilizzato in alcune elaborazioni.
ALTERNATIVE_MAX_LENGTH = 128

# Classificazione a etichetta singola.
MULTI_LABEL = False

# Il modello opera sui testi nella lingua originale.
TRANSLATE_TEXT = False

# I testi identici possono essere classificati una sola volta.
DEDUPLICATE_TEXTS = True

# La classificazione viene eseguita per blocchi.
ENABLE_BATCH_PROCESSING = True

# Possibilità di riprendere elaborazioni interrotte.
ENABLE_CHECKPOINTS = True


# ============================================================
# 4. CONFIGURAZIONE DEL CLASSIFICATORE
# ============================================================

@dataclass(frozen=True)
class ClassifierConfig:
    """
    Configurazione di un'esecuzione del classificatore.

    Parameters
    ----------
    hypotheses : Mapping[str, str]
        Associazione tra le quattro funzioni comunicative
        e le ipotesi testuali utilizzate dal modello.

    max_length : int
        Lunghezza massima degli input in token.

    batch_size : int
        Numero di testi elaborati per blocco.

    device : str
        Dispositivo di esecuzione.

    Notes
    -----
    Le ipotesi non vengono definite automaticamente.

    Devono essere fornite esplicitamente per garantire
    la tracciabilità della configurazione utilizzata.
    """

    hypotheses: Mapping[str, str]

    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = 8
    device: str = "cpu"

    def __post_init__(self):

        if set(self.hypotheses) != set(INTENT_ORDER):
            raise ValueError(
                "Le ipotesi devono essere definite "
                "per tutte e quattro le funzioni: "
                f"{INTENT_ORDER}"
            )

        for intent, hypothesis in self.hypotheses.items():

            if not isinstance(hypothesis, str):
                raise TypeError(
                    f"L'ipotesi per {intent} deve essere una stringa."
                )

            if not hypothesis.strip():
                raise ValueError(
                    f"L'ipotesi per {intent} non può essere vuota."
                )

        if self.max_length <= 0:
            raise ValueError(
                "max_length deve essere positivo."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size deve essere positivo."
            )

        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError(
                "Dispositivo non supportato."
            )


# ============================================================
# 5. VALIDAZIONE DELLE IPOTESI
# ============================================================

def validate_hypotheses(hypotheses):
    """
    Verifica la presenza delle quattro ipotesi NLI.

    Returns
    -------
    bool
        True se la configurazione è valida.

    Raises
    ------
    ValueError
        Se mancano categorie o sono presenti
        categorie aggiuntive.
    """

    expected = set(INTENT_ORDER)
    provided = set(hypotheses)

    missing = expected - provided
    extra = provided - expected

    if missing or extra:
        raise ValueError(
            f"Ipotesi non valide. "
            f"Mancanti: {sorted(missing)}. "
            f"Aggiuntive: {sorted(extra)}."
        )

    for intent in INTENT_ORDER:
        value = hypotheses[intent]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Ipotesi non valida per {intent}."
            )

    return True


# ============================================================
# 6. ORDINAMENTO DEI RISULTATI
# ============================================================

def order_intent_scores(scores):
    """
    Ordina i punteggi secondo INTENT_ORDER.

    Parameters
    ----------
    scores : Mapping[str, float]
        Punteggi associati alle quattro funzioni.

    Returns
    -------
    dict
        Punteggi ordinati.
    """

    missing = set(INTENT_ORDER) - set(scores)

    if missing:
        raise ValueError(
            f"Punteggi mancanti: {sorted(missing)}"
        )

    return {
        intent: float(scores[intent])
        for intent in INTENT_ORDER
    }


# ============================================================
# 7. SELEZIONE DELL'INTENTO
# ============================================================

def select_intent(scores):
    """
    Seleziona la funzione con il punteggio maggiore.

    In caso di parità, viene utilizzato l'ordine
    definito in INTENT_ORDER.

    Returns
    -------
    dict
        Etichetta, punteggio massimo e margine
        rispetto al secondo punteggio.
    """

    ordered = order_intent_scores(scores)

    ranked = sorted(
        ordered.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_intent, best_score = ranked[0]
    second_score = ranked[1][1]

    return {
        "intent": best_intent,
        "score": best_score,
        "margin": best_score - second_score,
    }
