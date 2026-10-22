# Dinamiche emotive e reti di condivisione nei gruppi Facebook durante le emergenze

### Un'analisi comparata tra eventi a insorgenza improvvisa e prevedibile

**Tesi di laurea triennale in Informatica**  
Università degli Studi di Torino — A.A. 2025/2026

**Autore:** Leonardo Cosmin Francu

---

## 1. Descrizione del progetto

Questo repository raccoglie il codice sorgente Python, le configurazioni e la documentazione tecnica sviluppati nell'ambito della tesi *Dinamiche emotive e reti di condivisione nei gruppi Facebook durante le emergenze: un'analisi comparata tra eventi a insorgenza improvvisa e prevedibile*.

L'obiettivo è documentare le procedure computazionali adottate, garantire la tracciabilità delle scelte implementative e favorire la riproducibilità metodologica della ricerca.

Lo studio esamina la comunicazione nei gruppi Facebook pubblici durante otto emergenze naturali, analizzando:

- La distribuzione e l'evoluzione delle reaction nelle diverse fasi dell'emergenza.
- Le funzioni comunicative dei contenuti pubblicati.
- Le relazioni di co-condivisione tra gruppi Facebook.
- Le differenze tra emergenze caratterizzate da tempi di preavviso differenti.

Le analisi sono state sviluppate in Python ed eseguite all'interno del **Meta Secure Research Environment (SRE)**, utilizzando dati acquisiti attraverso la **Meta Content Library API**. Il repository contiene moduli riutilizzabili che rappresentano le principali operazioni della procedura analitica.

---

## 2. Obiettivi della ricerca

### 2.1 Analisi temporale delle reaction

Esaminare come varia la distribuzione delle reaction associate ai contenuti pubblicati nelle diverse fasi di un'emergenza, considerando la composizione percentuale e l'evoluzione temporale.

### 2.2 Classificazione degli intenti comunicativi

Identificare le principali funzioni comunicative dei post mediante un modello multilingue di classificazione zero-shot basato sulla **Natural Language Inference (NLI)**.

### 2.3 Analisi delle reti di co-condivisione

Ricostruire le relazioni tra gruppi Facebook sulla base della pubblicazione di contenuti e collegamenti comuni; esaminare la relazione tra struttura delle reti e similarità dei profili aggregati di reaction.

### 2.4 Analisi comparativa

Confrontare i risultati ottenuti in emergenze caratterizzate da differenti tempi di preavviso (*lead time*), valutando le possibili associazioni tra prevedibilità dell'evento e dinamiche osservate.

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

Per ciascun evento sono stati definiti un momento di riferimento (**T0**), un eventuale avviso operativo ufficiale, il tempo di preavviso quando determinabile, una finestra temporale di osservazione e interrogazioni testuali specifiche. La definizione di T0 e degli eventuali avvisi precede l'analisi dei contenuti social.

---

## 4. Metodologia

### 4.1 Raccolta e preparazione dei dati

I dati sono stati acquisiti mediante interrogazioni testuali della Meta Content Library, limitando la ricerca ai post pubblicati nei gruppi Facebook pubblici. Le interrogazioni sono state definite individualmente per ciascun evento, utilizzando termini pertinenti al fenomeno e all'area geografica interessata; la selezione delle parole chiave è stata supportata anche dall'esame delle tendenze di ricerca disponibili attraverso Google Trends.

Le interrogazioni sono state configurate nella Meta Content Library ed eseguite nel Meta Secure Research Environment mediante gli strumenti autorizzati della piattaforma.

La preparazione dei dati comprende:

1. Unificazione dei risultati delle interrogazioni.
2. Identificazione e rimozione dei post duplicati nella preparazione dei dataset di analisi.
3. Normalizzazione dei campi utilizzati nelle elaborazioni.
4. Conversione e gestione dei riferimenti temporali.
5. Preparazione dei conteggi delle reaction.
6. Assegnazione delle fasi dell'emergenza.

La deduplicazione dei post nella preparazione dei dataset è distinta dalla deduplicazione dei testi effettuata per ottimizzare la classificazione NLI.

Il modulo `src/collection.py` **documenta e valida le configurazioni delle interrogazioni, ma non esegue direttamente chiamate alla Meta Content Library API**.

### 4.2 Segmentazione temporale

Per garantire la comparabilità delle analisi, è stata adottata una finestra di osservazione compresa tra 7 giorni precedenti e 14 giorni successivi a T0. I post sono suddivisi in quattro fasi sulla base del giorno di pubblicazione rispetto a T0.

| Fase | Giorni rispetto a T0 | Descrizione |
|---|---|---|
| Pre | −7 / −1 | Periodo precedente all'evento |
| Evento | 0 / +1 | Fase iniziale dell'emergenza |
| Post1 | +2 / +7 | Prima fase successiva |
| Post2 | +8 / +14 | Seconda fase successiva |

La segmentazione è basata sui giorni di calendario, utilizzando riferimenti temporali in UTC. Le quattro fasi consentono di analizzare l'evoluzione delle distribuzioni delle reaction e delle funzioni comunicative durante l'emergenza.

### 4.3 Analisi delle reaction

L'analisi considera sei tipologie di reaction:

`like` · `love` · `haha` · `wow` · `sad` · `angry`

La reaction `care` non è inclusa nelle elaborazioni comparative definitive, a causa della disponibilità non uniforme dei relativi conteggi nei dataset analizzati.

Per ciascun evento e fase temporale vengono calcolati:

- Numero di post.
- Conteggio complessivo delle reaction.
- Distribuzione percentuale delle singole reaction.
- Numero medio di reaction per post.

Le distribuzioni percentuali sono calcolate sul totale delle reaction aggregate, e non come media delle percentuali dei singoli post:

**Quota reaction (%) = (Conteggio della reaction / Totale delle reaction considerate) × 100**

> **Nota metodologica:** i conteggi delle reaction si riferiscono al momento dell'estrazione dei dati. Le fasi temporali identificano quindi il periodo di pubblicazione dei post, non necessariamente il momento in cui le reaction sono state espresse.

### 4.4 Classificazione degli intenti comunicativi

La classificazione automatica dei post utilizza un approccio zero-shot basato sulla **Natural Language Inference (NLI)**.

**Modello:** `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`

| Funzione | Definizione operativa |
|---|---|
| **INFORM** | Riportare fatti, aggiornamenti, avvisi, dati e indicazioni pratiche. |
| **INTERPRET** | Spiegare o commentare cause, conseguenze, contesto e significato dell'emergenza. |
| **INSPIRE** | Esprimere solidarietà, sostegno e speranza oppure promuovere aiuto e mobilitazione. |
| **INVESTIGATE** | Sollevare domande, critiche e richieste di responsabilità. |

La procedura comprende:

1. Normalizzazione dei testi.
2. Identificazione dei testi distinti.
3. Inferenza NLI sulle quattro funzioni comunicative.
4. Assegnazione dell'etichetta con punteggio maggiore.
5. Propagazione dell'etichetta ai post corrispondenti.

L'elaborazione viene eseguita per blocchi, con salvataggi intermedi, per ridurre i tempi di esecuzione e limitare la perdita di risultati in caso di interruzione. La deduplicazione dei testi è utilizzata esclusivamente per ottimizzare l'inferenza: i post originali vengono conservati per le analisi successive.

Le formulazioni delle ipotesi utilizzate dal classificatore possono differire tra eventi. Pertanto, le distribuzioni assolute degli intenti non vengono considerate direttamente confrontabili tra eventi senza ulteriori verifiche di omogeneità metodologica. Le configurazioni documentate del classificatore sono raccolte in `config/classifier_config.py`.

### 4.5 Analisi delle reti di co-condivisione

Le reti rappresentano le relazioni tra gruppi Facebook che pubblicano contenuti o collegamenti comuni.

- **Nodi:** gruppi Facebook pubblici identificati mediante identificativi univoci.
- **Archi:** relazioni di co-condivisione tra coppie di gruppi.
- **Pesi degli archi:** numero di contenuti distinti condivisi, secondo la procedura implementata nel modulo di costruzione della rete.
- **Profili dei nodi:** distribuzioni aggregate delle reaction associate ai post dei gruppi.

Il modulo `src/network_analysis.py` implementa una rete semplice, non orientata e pesata. Per impostazione predefinita, il grafo comprende soltanto i gruppi con almeno un collegamento.

L'analisi comprende:

1. Costruzione dei grafi di co-condivisione.
2. Calcolo dei profili aggregati di reaction dei gruppi.
3. Misurazione della similarità mediante similarità del coseno.
4. Confronto tra gruppi collegati e non collegati.
5. Valutazione statistica attraverso test di permutazione.

La similarità del coseno misura la somiglianza tra i profili aggregati delle reaction. I test di permutazione permettono di valutare se le differenze osservate tra gruppi collegati e non collegati siano compatibili con l'ipotesi nulla adottata.

> **Precisazione metodologica:** il modulo pubblico di costruzione della rete identifica le co-condivisioni attraverso identificativi di contenuti e collegamenti comuni. Non applica automaticamente una soglia di sincronia temporale. La corrispondenza completa con le diverse implementazioni utilizzate nelle analisi originali richiede la verifica degli script SRE definitivi.

---

## 5. Struttura del repository

Il repository è organizzato in moduli Python che rappresentano le principali fasi della procedura analitica sviluppata nel Meta Secure Research Environment (SRE).

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

### 5.1 Organizzazione del codice

| Modulo | Funzione |
|---|---|
| `config/README.md` | Documentazione delle configurazioni della ricerca. |
| `config/events.py` | Definizione delle finestre temporali e dei parametri degli eventi. |
| `config/classifier_config.py` | Configurazione del modello NLI e delle funzioni comunicative. |
| `src/collection.py` | Documentazione, costruzione e validazione delle configurazioni delle interrogazioni Meta; non esegue richieste API. |
| `src/preprocessing.py` | Normalizzazione dei testi, preparazione delle reaction e gestione dei timestamp. |
| `src/temporal_analysis.py` | Assegnazione delle fasi temporali rispetto a T0. |
| `src/reaction_analysis.py` | Analisi delle distribuzioni e dell'evoluzione delle reaction. |
| `src/intent_classification.py` | Classificazione zero-shot delle funzioni comunicative. |
| `src/intent_analysis.py` | Analisi degli intenti nelle diverse fasi dell'emergenza. |
| `src/network_analysis.py` | Costruzione delle reti di co-condivisione tra gruppi Facebook. |
| `src/network_statistics.py` | Calcolo della similarità dei profili di reaction e test di permutazione. |
| `src/statistical_analysis.py` | Analisi statistiche delle reaction e confronto tra eventi. |
| `src/visualizations.py` | Produzione delle rappresentazioni grafiche dei risultati. |

I moduli seguono la sequenza metodologica della ricerca:

**Raccolta → Preprocessing → Segmentazione temporale → Analisi delle reaction → Classificazione NLI → Analisi degli intenti → Costruzione delle reti → Analisi statistica → Visualizzazione dei risultati.**

I moduli sono progettati per essere riutilizzabili su eventi differenti mediante configurazioni specifiche, senza incorporare nel codice i dati originali raccolti attraverso Meta Content Library.

### 5.2 Stato di implementazione

Tutti i moduli Python previsti dalla struttura del progetto sono presenti nel repository. Il repository documenta e implementa le principali procedure computazionali della ricerca.

La corrispondenza integrale tra gli script pubblicati e tutte le elaborazioni originali svolte nel Meta Secure Research Environment rimane soggetta a verifica metodologica. La presenza dei moduli non costituisce, da sola, una garanzia di riproduzione numericamente identica dei risultati riportati nella tesi.

---

## 6. Ambiente di esecuzione

**Linguaggio:** Python 3  
**Ambiente di ricerca originale:** Meta Secure Research Environment (SRE)

### 6.1 Strumenti e librerie principali

- **pandas** — Gestione e trasformazione dei dati.
- **numpy** — Elaborazione numerica.
- **scipy** — Analisi statistica.
- **networkx** — Costruzione e analisi delle reti.
- **torch** — Esecuzione del modello neurale.
- **transformers** — Inferenza NLI.
- **sentencepiece** — Supporto alla tokenizzazione del modello multilingue.
- **matplotlib** — Visualizzazione dei risultati.
- **tqdm** — Monitoraggio dell'avanzamento delle elaborazioni.

Le dipendenze Python del progetto sono elencate nel file `requirements.txt`, presente nella directory principale del repository.

### 6.2 Installazione delle dipendenze

In un ambiente Python compatibile, è possibile installare le dipendenze dichiarate eseguendo:

```bash
python -m pip install -r requirements.txt
```

L'installazione delle librerie non garantisce automaticamente la compatibilità con tutte le configurazioni del Meta Secure Research Environment. L'esecuzione del modello NLI può richiedere risorse computazionali significative, in particolare per dataset di grandi dimensioni.

### 6.3 Esecuzione delle analisi

I moduli contenuti in `src/` possono essere importati in script Python o notebook Jupyter. Il repository non include un unico comando che riproduca automaticamente l'intera ricerca.

Per eseguire le procedure è necessario disporre di:

1. Dataset compatibili con le strutture attese dai moduli.
2. Configurazioni degli eventi e delle finestre temporali.
3. Dipendenze Python installate.
4. Risorse computazionali adeguate alle operazioni richieste.

L'accesso ai dati originali della Meta Content Library rimane soggetto alle autorizzazioni previste dalla piattaforma.

---

## 7. Riproducibilità e accesso ai dati

Il repository è finalizzato alla trasparenza delle procedure computazionali, alla tracciabilità delle scelte metodologiche e al riutilizzo del codice sviluppato nell'ambito della ricerca.

I moduli Python implementano le principali operazioni di preparazione, trasformazione e analisi dei dati, consentendo di applicare le procedure a dataset compatibili.

La raccolta originale è stata effettuata attraverso la Meta Content Library, con esecuzione delle interrogazioni nel Meta Secure Research Environment (SRE).

Il modulo `src/collection.py` consente di documentare e validare le configurazioni delle interrogazioni, ma non implementa direttamente le chiamate alla Meta Content Library API.

### 7.1 Accesso ai dati

I dati originali acquisiti attraverso la Meta Content Library API non sono pubblicati né distribuiti nel repository.

L'esecuzione delle procedure sui dati originali richiede:

- Autorizzazione all'accesso alla Meta Content Library.
- Disponibilità del Meta Secure Research Environment.
- Rispetto delle condizioni di utilizzo e delle restrizioni previste da Meta.

La disponibilità pubblica del codice non implica la disponibilità pubblica dei dataset.

### 7.2 Livelli di riproducibilità

**Riproducibilità metodologica:** il repository documenta le principali procedure di elaborazione e analisi, consentendo di comprenderne le scelte implementative.

**Riproducibilità computazionale:** i moduli possono essere riutilizzati su dataset compatibili, previa verifica delle dipendenze, delle configurazioni e delle condizioni di esecuzione.

**Riproducibilità dei risultati originali:** richiede l'accesso ai dati utilizzati nella ricerca, alle configurazioni definitive delle interrogazioni e alle versioni effettivamente impiegate degli script analitici.

Pertanto, il repository non garantisce, da solo, la riproduzione numericamente identica di tutti i risultati riportati nell'elaborato di tesi.

### 7.3 Limitazioni dell'ambiente pubblico

Il repository non contiene credenziali, token di autenticazione o dataset soggetti alle restrizioni della Meta Content Library.

Le operazioni di acquisizione dei dati devono essere eseguite esclusivamente attraverso gli strumenti e gli ambienti autorizzati dalla piattaforma. Il codice pubblico costituisce quindi un riferimento metodologico e computazionale, non un sistema autonomo di accesso ai dati Meta.

---

## 8. Limitazioni metodologiche

### 8.1 Selezione dei contenuti

La selezione dei post dipende dalle interrogazioni testuali utilizzate. I contenuti che non corrispondono ai termini selezionati potrebbero non essere rappresentati nei dataset analizzati.

### 8.2 Disponibilità dei dati

I dataset comprendono soltanto i contenuti disponibili e accessibili attraverso la piattaforma al momento dell'estrazione. Eventuali variazioni nella disponibilità dei contenuti possono influire sulla riproduzione delle raccolte.

### 8.3 Interpretazione delle reaction

Le reaction costituiscono indicatori osservabili della risposta del pubblico, ma non rappresentano misure dirette e univoche degli stati emotivi individuali. Inoltre, i conteggi disponibili non permettono di ricostruire il momento esatto in cui ciascuna reaction è stata espressa.

### 8.4 Classificazione automatica

La classificazione zero-shot presenta margini di incertezza. Le etichette assegnate dipendono dal modello, dalle formulazioni delle ipotesi NLI e dalle caratteristiche linguistiche dei contenuti. Le differenze nelle ipotesi utilizzate possono limitare la comparabilità diretta delle distribuzioni degli intenti tra eventi.

### 8.5 Interpretazione delle reti

Le reti di co-condivisione rappresentano relazioni tra contenuti pubblicati da gruppi Facebook. Un collegamento tra due gruppi non implica necessariamente:

- Un'interazione diretta tra gli amministratori.
- Un rapporto organizzativo.
- Una relazione causale.
- Un coordinamento intenzionale.

La similarità dei profili di reaction non dimostra che la connessione tra gruppi determini risposte emotive simili.

### 8.6 Comparabilità tra eventi

Gli eventi analizzati differiscono per caratteristiche geografiche, linguistiche, temporali e per disponibilità dei contenuti. I risultati comparativi devono pertanto essere interpretati considerando tali differenze e le specifiche condizioni di raccolta e analisi.

Per la discussione completa delle scelte metodologiche, dei risultati e delle limitazioni si rimanda all'elaborato di tesi.

---

## 9. Licenza

Il codice sorgente è distribuito secondo i termini della **MIT License**. Per maggiori informazioni, consultare il file `LICENSE`.

La licenza riguarda esclusivamente i materiali del repository e non si estende ai dati originali della Meta Content Library.

---

**Leonardo Cosmin Francu**  
Università degli Studi di Torino · 2026
