"""
intent_classification.py
========================

Classificazione zero-shot degli intenti comunicativi
mediante inferenza testuale (NLI).

Modello:
MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7

Funzioni:
- INFORM
- INTERPRET
- INSPIRE
- INVESTIGATE

Metodologia:
1. Normalizzazione dei testi.
2. Deduplicazione dei testi.
3. Costruzione delle coppie premessa-ipotesi.
4. Inferenza NLI in batch.
5. Estrazione dei punteggi di entailment.
6. Selezione dell'intento con punteggio massimo.
7. Calcolo del margine tra primo e secondo punteggio.
8. Salvataggio e ripresa tramite checkpoint.
9. Propagazione delle etichette ai post originali.

Le ipotesi NLI vengono fornite dall'esterno.
Non sono vincolate a uno specifico evento.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from tqdm.auto import tqdm


# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

MODEL_NAME = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)

LABELS = (
    "INFORM",
    "INTERPRET",
    "INSPIRE",
    "INVESTIGATE",
)

DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 256

CHECKPOINT_EVERY = 100


# ============================================================
# 2. VALIDAZIONE DELLE IPOTESI
# ============================================================

def validate_hypotheses(hypotheses):
    """
    Verifica che siano presenti le quattro
    ipotesi NLI richieste.

    Le formulazioni devono essere fornite
    esplicitamente dall'utilizzatore.
    """

    if not isinstance(hypotheses, dict):
        raise TypeError(
            "hypotheses deve essere un dizionario."
        )

    expected = set(LABELS)
    provided = set(hypotheses)

    if expected != provided:
        raise ValueError(
            "Le ipotesi devono corrispondere "
            "alle quattro categorie: "
            f"{LABELS}"
        )

    for label in LABELS:

        value = hypotheses[label]

        if not isinstance(value, str):
            raise TypeError(
                f"Ipotesi non valida per {label}."
            )

        if not value.strip():
            raise ValueError(
                f"Ipotesi vuota per {label}."
            )

    return True


# ============================================================
# 3. NORMALIZZAZIONE DEI TESTI
# ============================================================

def normalize_text(text):
    """
    Normalizza gli spazi senza modificare
    il contenuto linguistico del testo.
    """

    if not isinstance(text, str):
        return None

    normalized = " ".join(text.split())

    return normalized if normalized else None


# ============================================================
# 4. PREPARAZIONE DEI TESTI DISTINTI
# ============================================================

def prepare_unique_texts(
    df,
    text_column="text",
):
    """
    Prepara i testi distinti da classificare.

    Non elimina né modifica i post originali.

    Returns
    -------
    pandas.DataFrame
        Colonne:
        - text_id
        - text
    """

    if text_column not in df.columns:
        raise KeyError(
            f"Colonna mancante: {text_column}"
        )

    texts = (
        df[text_column]
        .map(normalize_text)
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    result = pd.DataFrame({
        "text_id": np.arange(
            len(texts),
            dtype=np.int64,
        ),
        "text": texts,
    })

    return result


# ============================================================
# 5. CARICAMENTO DEL MODELLO
# ============================================================

def load_nli_model(
    model_name=MODEL_NAME,
    device="cpu",
    local_files_only=False,
):
    """
    Carica tokenizer e modello NLI.

    Parameters
    ----------
    model_name : str
        Identificativo Hugging Face o percorso locale.

    device : str
        cpu, cuda oppure mps.

    local_files_only : bool
        Se True, utilizza esclusivamente
        file già disponibili localmente.

        Impostazione appropriata per ambienti
        protetti senza accesso alla rete.

    Returns
    -------
    tuple
        tokenizer, model, entailment_id
    """

    if device not in ("cpu", "cuda", "mps"):
        raise ValueError(
            "Dispositivo non supportato."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
    )

    model.eval()
    model.to(device)

    label2id = {
        str(label).lower(): int(index)
        for label, index
        in model.config.label2id.items()
    }

    entailment_ids = [
        index
        for label, index in label2id.items()
        if "entail" in label
    ]

    if len(entailment_ids) != 1:
        raise ValueError(
            "Impossibile identificare univocamente "
            "la classe entailment del modello."
        )

    entailment_id = entailment_ids[0]

    return tokenizer, model, entailment_id


# ============================================================
# 6. INFERENZA NLI SU UN BATCH
# ============================================================

def predict_batch(
    texts,
    hypotheses,
    tokenizer,
    model,
    entailment_id,
    max_length=DEFAULT_MAX_LENGTH,
):
    """
    Classifica un batch di testi.

    Per ogni testo vengono costruite quattro
    coppie premessa-ipotesi.

    Il punteggio associato a ciascuna funzione
    è la probabilità di entailment ottenuta
    mediante softmax sui logits NLI.

    Returns
    -------
    list[dict]
        Una predizione per ciascun testo.
    """

    validate_hypotheses(hypotheses)

    if not texts:
        return []

    premises = []
    hypothesis_texts = []

    for text in texts:

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Il batch contiene testi non validi."
            )

        for label in LABELS:

            premises.append(text)

            hypothesis_texts.append(
                hypotheses[label]
            )

    encoded = tokenizer(
        premises,
        hypothesis_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    device = next(model.parameters()).device

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():

        logits = model(
            **encoded
        ).logits

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[:, entailment_id]

    scores = (
        probabilities
        .reshape(len(texts), len(LABELS))
        .cpu()
        .numpy()
    )

    results = []

    for row in scores:

        # Ordinamento stabile per gestire eventuali parità.
        ranking = np.argsort(
            -row,
            kind="stable",
        )

        top1 = int(ranking[0])
        top2 = int(ranking[1])

        result = {
            "intent": LABELS[top1],
            "score_top1": float(row[top1]),
            "score_top2": float(row[top2]),
            "margin": float(
                row[top1] - row[top2]
            ),
        }

        results.append(result)

    return results


# ============================================================
# 7. GESTIONE DEI CHECKPOINT
# ============================================================

def load_checkpoint(checkpoint_path):
    """
    Carica un checkpoint precedentemente salvato.

    Se il file non esiste, restituisce
    un DataFrame vuoto.
    """

    columns = [
        "text_id",
        "text",
        "intent",
        "score_top1",
        "score_top2",
        "margin",
    ]

    if checkpoint_path is None:
        return pd.DataFrame(columns=columns)

    path = Path(checkpoint_path)

    if not path.exists():
        return pd.DataFrame(columns=columns)

    checkpoint = pd.read_pickle(path)

    missing = set(columns) - set(checkpoint.columns)

    if missing:
        raise ValueError(
            "Checkpoint incompleto. "
            f"Colonne mancanti: {sorted(missing)}"
        )

    if checkpoint["text_id"].duplicated().any():
        raise ValueError(
            "Il checkpoint contiene text_id duplicati."
        )

    return checkpoint[columns].copy()


def save_checkpoint(
    results,
    checkpoint_path,
):
    """
    Salva le predizioni in formato pickle.

    Il salvataggio utilizza un file temporaneo
    per ridurre il rischio di checkpoint parziali.
    """

    if checkpoint_path is None:
        return

    path = Path(checkpoint_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    results.to_pickle(temporary_path)

    temporary_path.replace(path)


# ============================================================
# 8. CLASSIFICAZIONE DEI TESTI DISTINTI
# ============================================================

def classify_unique_texts(
    unique_texts,
    hypotheses,
    tokenizer,
    model,
    entailment_id,
    batch_size=DEFAULT_BATCH_SIZE,
    max_length=DEFAULT_MAX_LENGTH,
    checkpoint_path=None,
    checkpoint_every=CHECKPOINT_EVERY,
):
    """
    Classifica un DataFrame di testi distinti.

    Parametri
    ---------
    unique_texts : pandas.DataFrame
        Deve contenere:
        - text_id
        - text

    hypotheses : dict
        Ipotesi NLI per le quattro funzioni.

    checkpoint_path : str or Path, optional
        Percorso del checkpoint.

    checkpoint_every : int
        Frequenza del salvataggio, in batch.

    Returns
    -------
    pandas.DataFrame
        Predizioni dei testi distinti.

    Notes
    -----
    I checkpoint devono essere riutilizzati
    esclusivamente con lo stesso modello,
    le stesse ipotesi e gli stessi parametri.

    Per una nuova configurazione utilizzare
    un nuovo percorso di checkpoint.
    """

    validate_hypotheses(hypotheses)

    required = {"text_id", "text"}

    if not required.issubset(unique_texts.columns):
        raise ValueError(
            "Sono richieste le colonne "
            "'text_id' e 'text'."
        )

    if unique_texts["text_id"].duplicated().any():
        raise ValueError(
            "Sono presenti text_id duplicati."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size deve essere positivo."
        )

    if max_length <= 0:
        raise ValueError(
            "max_length deve essere positivo."
        )

    if checkpoint_every <= 0:
        raise ValueError(
            "checkpoint_every deve essere positivo."
        )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    # Verifica che gli ID già elaborati corrispondano
    # agli stessi testi del dataset corrente.
    if not checkpoint.empty:

        reference = unique_texts[
            ["text_id", "text"]
        ]

        verification = checkpoint[
            ["text_id", "text"]
        ].merge(
            reference,
            on="text_id",
            how="left",
            suffixes=("_saved", "_current"),
            validate="one_to_one",
            indicator=True,
        )

        mismatch = (
            verification["_merge"].ne("both")
            | verification["text_saved"].ne(
                verification["text_current"]
            )
        )

        if mismatch.any():
            raise ValueError(
                "Il checkpoint non corrisponde "
                "ai testi correnti."
            )

    completed_ids = set(
        checkpoint["text_id"].tolist()
    )

    pending = unique_texts.loc[
        ~unique_texts["text_id"].isin(
            completed_ids
        )
    ].copy()

    new_results = []

    total_batches = (
        len(pending) + batch_size - 1
    ) // batch_size

    for batch_number, start in enumerate(
        tqdm(
            range(0, len(pending), batch_size),
            total=total_batches,
            desc="Classificazione NLI",
        ),
        start=1,
    ):

        batch = pending.iloc[
            start:start + batch_size
        ]

        predictions = predict_batch(
            texts=batch["text"].tolist(),
            hypotheses=hypotheses,
            tokenizer=tokenizer,
            model=model,
            entailment_id=entailment_id,
            max_length=max_length,
        )

        for (_, row), prediction in zip(
            batch.iterrows(),
            predictions,
        ):

            new_results.append({
                "text_id": row["text_id"],
                "text": row["text"],
                **prediction,
            })

        should_save = (
            checkpoint_path is not None
            and (
                batch_number % checkpoint_every == 0
                or batch_number == total_batches
            )
        )

        if should_save:

            partial = pd.concat(
                [
                    checkpoint,
                    pd.DataFrame(new_results),
                ],
                ignore_index=True,
            )

            save_checkpoint(
                partial,
                checkpoint_path,
            )

    results = pd.concat(
        [
            checkpoint,
            pd.DataFrame(new_results),
        ],
        ignore_index=True,
    )

    if not results.empty:

        results = results.sort_values(
            "text_id",
            kind="stable",
        ).reset_index(drop=True)

    return results


# ============================================================
# 9. PROPAGAZIONE DELLE ETICHETTE AI POST
# ============================================================

def propagate_predictions(
    df,
    predictions,
    text_column="text",
):
    """
    Associa le predizioni dei testi distinti
    a tutti i post originali.

    I post senza testo classificabile
    vengono conservati con valori mancanti.

    Il numero di righe del dataset
    originale rimane invariato.
    """

    if text_column not in df.columns:
        raise KeyError(
            f"Colonna mancante: {text_column}"
        )

    required = {
        "text",
        "intent",
        "score_top1",
        "score_top2",
        "margin",
    }

    if not required.issubset(predictions.columns):
        raise ValueError(
            "Il DataFrame delle predizioni "
            "non contiene tutte le colonne richieste."
        )

    if predictions["text"].duplicated().any():
        raise ValueError(
            "Le predizioni contengono testi duplicati."
        )

    result = df.copy()

    result["_normalized_text_nli"] = (
        result[text_column]
        .map(normalize_text)
    )

    existing_columns = [
        "intent",
        "score_top1",
        "score_top2",
        "margin",
    ]

    if any(
        column in result.columns
        for column in existing_columns
    ):
        raise ValueError(
            "Il dataset contiene già colonne "
            "di classificazione."
        )

    prediction_columns = [
        "text",
        "intent",
        "score_top1",
        "score_top2",
        "margin",
    ]

    lookup = predictions[
        prediction_columns
    ].rename(
        columns={
            "text": "_normalized_text_nli"
        }
    )

    original_length = len(result)

    result = result.merge(
        lookup,
        on="_normalized_text_nli",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if len(result) != original_length:
        raise RuntimeError(
            "La propagazione ha modificato "
            "il numero di post."
        )

    result = result.drop(
        columns=["_normalized_text_nli"]
    )

    return result


# ============================================================
# 10. PIPELINE COMPLETA
# ============================================================

def classify_posts(
    df,
    hypotheses,
    text_column="text",
    model_name=MODEL_NAME,
    device="cpu",
    batch_size=DEFAULT_BATCH_SIZE,
    max_length=DEFAULT_MAX_LENGTH,
    checkpoint_path=None,
    checkpoint_every=CHECKPOINT_EVERY,
    local_files_only=False,
):
    """
    Esegue l'intera pipeline NLI.

    Passaggi:
    1. Preparazione dei testi distinti.
    2. Caricamento del modello.
    3. Classificazione in batch.
    4. Salvataggio/ripresa da checkpoint.
    5. Propagazione delle etichette ai post.

    Returns
    -------
    dict
        - data: dataset originale con intenti
        - unique_texts: testi distinti
        - predictions: classificazioni dei testi
        - summary: statistiche di elaborazione
    """

    validate_hypotheses(hypotheses)

    unique_texts = prepare_unique_texts(
        df,
        text_column=text_column,
    )

    if unique_texts.empty:

        empty_predictions = pd.DataFrame(
            columns=[
                "text_id",
                "text",
                "intent",
                "score_top1",
                "score_top2",
                "margin",
            ]
        )

        classified_data = propagate_predictions(
            df,
            empty_predictions,
            text_column=text_column,
        )

        return {
            "data": classified_data,
            "unique_texts": unique_texts,
            "predictions": empty_predictions,
            "summary": {
                "total_posts": len(df),
                "unique_texts": 0,
                "classified_texts": 0,
            },
        }

    tokenizer, model, entailment_id = (
        load_nli_model(
            model_name=model_name,
            device=device,
            local_files_only=local_files_only,
        )
    )

    predictions = classify_unique_texts(
        unique_texts=unique_texts,
        hypotheses=hypotheses,
        tokenizer=tokenizer,
        model=model,
        entailment_id=entailment_id,
        batch_size=batch_size,
        max_length=max_length,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )

    classified_data = propagate_predictions(
        df,
        predictions,
        text_column=text_column,
    )

    summary = {
        "total_posts": len(df),
        "unique_texts": len(unique_texts),
        "classified_texts": len(predictions),
        "model": model_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "device": device,
    }

    return {
        "data": classified_data,
        "unique_texts": unique_texts,
        "predictions": predictions,
        "summary": summary,
    }
