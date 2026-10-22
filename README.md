# Dinamiche emotive e reti di condivisione nei gruppi Facebook durante le emergenze

### Un'analisi comparata tra eventi a insorgenza improvvisa e prevedibile

**Tesi di laurea triennale in Informatica**  
Università degli Studi di Torino — A.A. 2025/2026

**Autore:** Leonardo Cosmin Francu

---

## 1. Descrizione del progetto

Questo repository raccoglie il codice sorgente Python, le configurazioni e la documentazione tecnica sviluppati nell'ambito della tesi:

*Dinamiche emotive e reti di condivisione nei gruppi Facebook durante le emergenze: un'analisi comparata tra eventi a insorgenza improvvisa e prevedibile.*

L'obiettivo del repository è documentare le procedure computazionali adottate, garantire la tracciabilità delle scelte implementative e favorire la riproducibilità metodologica della ricerca.

Lo studio esamina la comunicazione nei gruppi Facebook pubblici durante otto emergenze naturali, analizzando la risposta del pubblico attraverso le reaction, le funzioni comunicative dei contenuti e le relazioni di co-condivisione tra gruppi.

Le analisi sono state sviluppate in Python ed eseguite all'interno del **Meta Secure Research Environment (SRE)**, utilizzando dati acquisiti attraverso la **Meta Content Library API**.

---

## 2. Obiettivi della ricerca

La ricerca si articola in quattro principali dimensioni analitiche.

### Analisi temporale delle reaction

Esaminare come varia la distribuzione delle reaction associate ai contenuti pubblicati nelle diverse fasi di un'emergenza.

### Classificazione degli intenti comunicativi

Identificare le principali funzioni comunicative dei post mediante un modello multilingue di classificazione zero-shot.

### Analisi delle reti di co-condivisione

Ricostruire le relazioni tra gruppi Facebook sulla base della condivisione di contenuti e collegamenti comuni.

### Analisi comparativa

Confrontare i risultati ottenuti in emergenze caratterizzate da differenti tempi di preavviso (*lead time*).

---

## 3. Eventi analizzati

Il campione comprende otto emergenze naturali verificatesi tra il 2021 e il 2024.

| Evento | Area geografica | Anno |
|---|---|---|
| Eruzione del Cumbre Vieja | La Palma, Spagna | 2021 |
| Tifone Rai (Odette) | Filippine | 2021 |
| Terremoto di Turchia e Siria | Turchia e Siria | 2023 |
| Ciclone Freddy | Africa sudorientale | 2023 |
| Alluvione dell'Emilia-Romagna | Italia | 2023 |
| Terremoto del Marocco | Marocco | 2023 |
| Uragano Otis | Messico | 2023 |
| Alluvioni del Rio Grande do Sul | Brasile | 2024 |

Per ciascun evento è stata definita una data di riferimento (**T0**) e una finestra temporale comune di osservazione.

---

## 4. Metodologia

### 4.1 Raccolta e preparazione dei dati

I dati sono stati acquisiti mediante interrogazioni testuali della Meta Content Library API, limitando la ricerca ai post pubblicati nei gruppi Facebook pubblici.

Le interrogazioni sono state definite individualmente per ciascun evento, utilizzando termini di ricerca pertinenti al fenomeno e all'area geografica interessata.

La preparazione dei dati comprende:

- Unificazione dei risultati delle interrogazioni.
- Rimozione dei post duplicati.
- Normalizzazione dei campi utilizzati nelle analisi.
- Conversione e gestione dei riferimenti temporali.
- Assegnazione delle fasi dell'emergenza.

### 4.2 Segmentazione temporale

Per garantire la comparabilità delle analisi, è stata adottata una finestra di osservazione compresa tra 7 giorni precedenti e 14 giorni successivi a T0.

I post sono suddivisi in quattro fasi sulla base del giorno di pubblicazione rispetto a T0.

| Fase | Giorni rispetto a T0 | Descrizione |
|---|---|---|
| Pre | −7 / −1 | Periodo precedente all'evento |
| Evento | 0 / +1 | Fase iniziale dell'emergenza |
| Post1 | +2 / +7 | Prima fase successiva |
| Post2 | +8 / +14 | Seconda fase successiva |

La segmentazione è basata sui giorni di calendario, utilizzando riferimenti temporali in UTC.

### 4.3 Analisi delle reaction

L'analisi considera sei tipologie di reaction:

`like` · `love` · `haha` · `wow` · `sad` · `angry`

La reaction `care` è esclusa dalle elaborazioni poiché presenta conteggi nulli nei dataset raccolti.

Per ciascun evento e fase temporale vengono calcolati:

- Numero di post.
- Conteggio complessivo delle reaction.
- Distribuzione percentuale delle singole reaction.
- Numero medio di reaction per post.

Le distribuzioni percentuali sono calcolate sul totale delle reaction, e non come media delle percentuali dei singoli post.

**Nota metodologica:** i conteggi delle reaction si riferiscono al momento dell'estrazione dei dati. Le fasi temporali identificano quindi il periodo di pubblicazione dei post, non necessariamente il momento in cui le reaction sono state espresse.

### 4.4 Classificazione degli intenti comunicativi

La classificazione automatica dei post utilizza un approccio zero-shot basato sulla **Natural Language Inference (NLI)**.

Il modello impiegato è:

`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`

La classificazione distingue quattro funzioni comunicative:

| Funzione | Definizione operativa |
|---|---|
| **INFORM** | Riportare fatti, aggiornamenti, avvisi, dati e indicazioni pratiche. |
| **INTERPRET** | Spiegare o commentare cause, conseguenze, contesto e significato dell'emergenza. |
| **INSPIRE** | Esprimere solidarietà, sostegno e speranza oppure promuovere aiuto e mobilitazione. |
| **INVESTIGATE** | Sollevare domande, critiche e richieste di responsabilità. |

La procedura comprende:

1. Normalizzazione dei testi.
2. Identificazione dei testi distinti.
3. Inferenza NLI sulle quattro funzioni.
4. Assegnazione dell'etichetta con punteggio maggiore.
5. Propagazione dell'etichetta ai post corrispondenti.

L'elaborazione viene eseguita per blocchi, con salvataggi intermedi.

Le formulazioni delle ipotesi utilizzate dal classificatore possono differire tra eventi; pertanto, le distribuzioni assolute degli intenti non vengono considerate direttamente confrontabili tra eventi.

### 4.5 Analisi delle reti di co-condivisione

Le reti rappresentano le relazioni tra gruppi Facebook che condividono contenuti o collegamenti comuni.

La rappresentazione adottata prevede:

- **Nodi:** gruppi Facebook pubblici.
- **Archi:** relazioni di co-condivisione.
- **Profili dei nodi:** distribuzioni aggregate delle reaction.

L'analisi comprende:

1. Costruzione dei grafi di co-condivisione.
2. Calcolo dei profili di reaction dei gruppi.
3. Misurazione della similarità mediante similarità del coseno.
4. Confronto tra gruppi collegati e non collegati.
5. Valutazione statistica attraverso test di permutazione.

---

## 5. Struttura del repository

Il repository è organizzato in moduli Python che rispecchiano le principali fasi della procedura analitica sviluppata nel Meta Secure Research Environment (SRE).

La struttura prevista è la seguente:

```text
tesi-gruppi-facebook-emergenze/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── README.md
│   ├── events.py
│   └── classifier_config.py
│
└── src/
    ├── collection.py
    ├── preprocessing.py
    ├── temporal_analysis.py
    ├── reaction_analysis.py
    ├── intent_classification.py
    ├── intent_analysis.py
    ├── network_analysis.py
    ├── network_statistics.py
    ├── statistical_analysis.py
    └── visualizations.py
```

### Organizzazione del codice

| Modulo | Funzione |
|---|---|
| `config/README.md` | Documentazione delle configurazioni della ricerca. |
| `config/events.py` | Definizione delle finestre temporali e dei parametri degli eventi. |
| `config/classifier_config.py` | Configurazione del modello NLI e delle funzioni comunicative. |
| `src/collection.py` | Procedure di raccolta dei dati mediante Meta Content Library API. |
| `src/preprocessing.py` | Normalizzazione, pulizia e preparazione dei dataset. |
| `src/temporal_analysis.py` | Assegnazione delle fasi temporali rispetto a T0. |
| `src/reaction_analysis.py` | Analisi delle distribuzioni e dell'evoluzione delle reaction. |
| `src/intent_classification.py` | Classificazione zero-shot delle funzioni comunicative. |
| `src/intent_analysis.py` | Analisi degli intenti nelle diverse fasi dell'emergenza. |
| `src/network_analysis.py` | Costruzione delle reti di co-condivisione tra gruppi Facebook. |
| `src/network_statistics.py` | Calcolo della similarità dei profili di reaction e test di permutazione. |
| `src/statistical_analysis.py` | Analisi statistiche delle reaction e confronto tra eventi. |
| `src/visualizations.py` | Produzione delle rappresentazioni grafiche dei risultati. |

Gli script seguono la sequenza metodologica della ricerca: raccolta dei dati, preprocessing, segmentazione temporale, analisi delle reaction, classificazione degli intenti, costruzione delle reti e confronto dei risultati.

I moduli sono progettati per essere riutilizzabili su eventi differenti, mediante configurazioni specifiche, senza incorporare nel codice i dati originali raccolti attraverso Meta Content Library.

**Stato di implementazione:** i file `collection.py`, `statistical_analysis.py` e `visualizations.py` sono previsti ma non risultano ancora presenti nella struttura GitHub verificata. Le funzionalità degli altri moduli sono in fase di revisione per verificarne la compatibilità con le procedure SRE.

---

## 6. Ambiente di esecuzione

**Linguaggio:** Python 3

**Ambiente di ricerca:** Meta Secure Research Environment (SRE)

### Strumenti e librerie principali

- **pandas** — Gestione e trasformazione dei dati.
- **numpy** — Elaborazione numerica.
- **scipy** — Analisi statistica.
- **networkx** — Costruzione e analisi delle reti.
- **scikit-learn** — Metriche di similarità.
- **torch** — Esecuzione del modello neurale.
- **transformers** — Inferenza NLI.
- **matplotlib** — Visualizzazione dei risultati.

Le dipendenze del progetto saranno documentate nel file `requirements.txt`.

---

## 7. Riproducibilità e accesso ai dati

Il repository è finalizzato alla trasparenza delle procedure computazionali e alla riproducibilità metodologica della ricerca.

I dati individuali acquisiti attraverso la Meta Content Library API non sono pubblicati né distribuiti nel repository.

L'esecuzione delle procedure sui dati originali richiede:

- Autorizzazione all'accesso alla Meta Content Library.
- Disponibilità del Meta Secure Research Environment.
- Rispetto delle condizioni di utilizzo e delle restrizioni previste da Meta.

La disponibilità pubblica del codice non implica la disponibilità pubblica dei dataset.

---

## 8. Limitazioni metodologiche

L'interpretazione dei risultati deve considerare alcune limitazioni:

- La selezione dei contenuti dipende dalle interrogazioni testuali utilizzate.
- I dataset comprendono soltanto i post disponibili al momento dell'estrazione.
- I conteggi delle reaction non consentono di ricostruire il momento esatto della risposta del pubblico.
- La classificazione zero-shot presenta margini di incertezza.
- Le differenze nelle ipotesi NLI limitano la comparabilità diretta delle distribuzioni degli intenti tra eventi.
- Le reti di co-condivisione rappresentano relazioni tra contenuti pubblicati, non necessariamente interazioni dirette tra utenti o gruppi.

Per la discussione completa delle scelte metodologiche e dei risultati si rimanda all'elaborato di tesi.

---

## 9. Licenza

Il codice sorgente è distribuito secondo i termini della **MIT License**.

Per maggiori informazioni, consultare il file `LICENSE`.

La licenza riguarda esclusivamente i materiali del repository e non si estende ai dati originali della Meta Content Library.

---

**Leonardo Cosmin Francu**  
Università degli Studi di Torino · 2026
