"""
visualizations.py
=================

Visualizzazioni per l'analisi dei gruppi Facebook
durante le emergenze naturali.

Grafici disponibili:
1. Distribuzione percentuale delle reaction per fase.
2. Evoluzione delle reaction per fase.
3. Distribuzione degli intenti per fase.
4. Heatmap intenti × fasi.
5. Reaction per intento comunicativo.
6. Rete di co-condivisione tra gruppi.
7. Similarità dei gruppi collegati e non collegati.
8. Relazione tra lead time e indicatori quantitativi.

Le funzioni utilizzano esclusivamente dati aggregati
o strutture di rete già elaborate.

Non sono necessarie interrogazioni alla Meta Content Library.

Autore: Leonardo Cosmin Francu
Università degli Studi di Torino
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAZIONE
# ============================================================

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

INTENT_ORDER = (
    "INFORM",
    "INTERPRET",
    "INSPIRE",
    "INVESTIGATE",
)

REACTION_COLORS = {
    "like": "#457B9D",
    "love": "#E63946",
    "wow": "#F4A261",
    "haha": "#E9C46A",
    "sad": "#6C757D",
    "angry": "#9D0208",
}

INTENT_COLORS = {
    "INFORM": "#264653",
    "INTERPRET": "#2A9D8F",
    "INSPIRE": "#E9C46A",
    "INVESTIGATE": "#E76F51",
}


# ============================================================
# FUNZIONI AUSILIARIE
# ============================================================

def _require_columns(df, columns):
    """Verifica la presenza delle colonne richieste."""

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonne mancanti: {missing}"
        )


def _save_figure(fig, save_path=None):
    """
    Salva una figura se viene specificato un percorso.

    Non salva automaticamente dati o grafici.
    """

    if save_path is not None:

        path = Path(save_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig


def _prepare_phase_table(
    df,
    phase_column="fase",
):
    """
    Converte una tabella con colonna fase
    in una tabella indicizzata per fase.

    Accetta anche una tabella già indicizzata.
    """

    table = df.copy()

    if phase_column in table.columns:

        table = table.set_index(
            phase_column
        )

    table = table.reindex(
        PHASE_ORDER
    )

    return table


def _numeric_table(
    table,
    columns,
):
    """
    Converte le colonne selezionate in valori numerici.

    Le categorie mancanti vengono rappresentate
    con zero, senza modificare la tabella originale.
    """

    table = table.reindex(
        columns=list(columns),
        fill_value=0,
    )

    return (
        table.apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0)
    )


# ============================================================
# 1. DISTRIBUZIONE DELLE REACTION
# ============================================================

def plot_reaction_distribution(
    df,
    phase_column="fase",
    reaction_columns=REACTION_COLUMNS,
    title="Distribuzione delle reaction per fase",
    save_path=None,
):
    """
    Grafico a barre impilate al 100%.

    Input:
    - tabella aggregata fase × reaction;
    - conteggi assoluti oppure percentuali.

    Le percentuali vengono calcolate
    a partire dai conteggi ricevuti.

    Se l'input contiene già percentuali
    che sommano a 100, il risultato resta invariato.
    """

    table = _prepare_phase_table(
        df,
        phase_column,
    )

    table = _numeric_table(
        table,
        reaction_columns,
    )

    totals = table.sum(axis=1)

    percentages = (
        table.div(
            totals.replace(0, np.nan),
            axis=0,
        )
        .fillna(0)
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    bottom = np.zeros(
        len(percentages)
    )

    for reaction in reaction_columns:

        values = percentages[
            reaction
        ].to_numpy()

        ax.bar(
            percentages.index,
            values,
            bottom=bottom,
            label=reaction.capitalize(),
            color=REACTION_COLORS.get(
                reaction
            ),
        )

        bottom += values

    ax.set_title(title)
    ax.set_ylabel("Percentuale delle reaction (%)")
    ax.set_xlabel("Fase dell'emergenza")
    ax.set_ylim(0, 100)

    ax.legend(
        title="Reaction",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return _save_figure(
        fig,
        save_path,
    )


# ============================================================
# 2. EVOLUZIONE DELLE REACTION
# ============================================================

def plot_reaction_evolution(
    df,
    phase_column="fase",
    reaction_columns=REACTION_COLUMNS,
    normalize=True,
    title="Evoluzione delle reaction",
    save_path=None,
):
    """
    Grafico a linee delle reaction nelle quattro fasi.

    normalize=True:
        rappresenta la composizione percentuale
        delle reaction per fase.

    normalize=False:
        rappresenta i conteggi aggregati.

    Le fasi hanno durate differenti:
    il grafico non rappresenta tassi giornalieri.
    """

    table = _prepare_phase_table(
        df,
        phase_column,
    )

    table = _numeric_table(
        table,
        reaction_columns,
    )

    if normalize:

        totals = table.sum(axis=1)

        table = (
            table.div(
                totals.replace(0, np.nan),
                axis=0,
            )
            * 100
        )

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    for reaction in reaction_columns:

        ax.plot(
            PHASE_ORDER,
            table[reaction].to_numpy(),
            marker="o",
            linewidth=2,
            label=reaction.capitalize(),
            color=REACTION_COLORS.get(
                reaction
            ),
        )

    ax.set_title(title)
    ax.set_xlabel("Fase dell'emergenza")

    if normalize:
        ax.set_ylabel(
            "Percentuale delle reaction (%)"
        )
    else:
        ax.set_ylabel(
            "Numero di reaction"
        )

    ax.legend(
        title="Reaction",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return _save_figure(
        fig,
        save_path,
    )


# ============================================================
# 3. DISTRIBUZIONE DEGLI INTENTI
# ============================================================

def plot_intent_distribution(
    df,
    phase_column="fase",
    intent_column="intent",
    title="Distribuzione degli intenti per fase",
    save_path=None,
):
    """
    Grafico a barre impilate degli intenti.

    Input:
        DataFrame contenente una riga per post
        e le colonne fase e intento.

    Le percentuali sono calcolate sul numero
    di post classificati in ciascuna fase.

    I post senza classificazione sono esclusi.
    """

    _require_columns(
        df,
        [
            phase_column,
            intent_column,
        ],
    )

    data = df.dropna(
        subset=[
            phase_column,
            intent_column,
        ]
    ).copy()

    counts = pd.crosstab(
        data[phase_column],
        data[intent_column],
    )

    counts = counts.reindex(
        index=PHASE_ORDER,
        columns=INTENT_ORDER,
        fill_value=0,
    )

    totals = counts.sum(axis=1)

    percentages = (
        counts.div(
            totals.replace(0, np.nan),
            axis=0,
        )
        .fillna(0)
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    bottom = np.zeros(
        len(PHASE_ORDER)
    )

    for intent in INTENT_ORDER:

        values = percentages[
            intent
        ].to_numpy()

        ax.bar(
            PHASE_ORDER,
            values,
            bottom=bottom,
            label=intent,
            color=INTENT_COLORS.get(
                intent
            ),
        )

        bottom += values

    ax.set_title(title)
    ax.set_xlabel("Fase dell'emergenza")
    ax.set_ylabel("Post classificati (%)")
    ax.set_ylim(0, 100)

    ax.legend(
        title="Intento",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return _save_figure(
        fig,
        save_path,
    )


# ============================================================
# 4. HEATMAP INTENTI × FASI
# ============================================================

def plot_intent_heatmap(
    df,
    phase_column="fase",
    intent_column="intent",
    title="Distribuzione degli intenti nelle fasi",
    save_path=None,
):
    """
    Heatmap della distribuzione percentuale
    degli intenti nelle quattro fasi.

    Righe:
        intenti comunicativi.

    Colonne:
        fasi dell'emergenza.

    Ogni colonna somma a 100%, se sono
    presenti post classificati nella fase.
    """

    _require_columns(
        df,
        [
            phase_column,
            intent_column,
        ],
    )

    data = df.dropna(
        subset=[
            phase_column,
            intent_column,
        ]
    )

    counts = pd.crosstab(
        data[phase_column],
        data[intent_column],
    )

    counts = counts.reindex(
        index=PHASE_ORDER,
        columns=INTENT_ORDER,
        fill_value=0,
    )

    totals = counts.sum(axis=1)

    percentages = (
        counts.div(
            totals.replace(0, np.nan),
            axis=0,
        )
        .fillna(0)
        * 100
    )

    matrix = percentages.T

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=100,
    )

    ax.set_xticks(
        range(len(PHASE_ORDER))
    )

    ax.set_xticklabels(
        PHASE_ORDER
    )

    ax.set_yticks(
        range(len(INTENT_ORDER))
    )

    ax.set_yticklabels(
        INTENT_ORDER
    )

    for i in range(len(INTENT_ORDER)):

        for j in range(len(PHASE_ORDER)):

            value = matrix.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color=(
                    "white"
                    if value >= 50
                    else "black"
                ),
            )

    ax.set_title(title)
    ax.set_xlabel("Fase dell'emergenza")
    ax.set_ylabel("Intento comunicativo")

    fig.colorbar(
        image,
        ax=ax,
        label="Percentuale dei post (%)",
    )

    fig.tight_layout()

    return _save_figure(
        fig,
        save_path,
    )


# ============================================================
# 5. REACTION PER INTENTO
# ============================================================

def plot_reactions_by_intent(
    df,
    intent_column="intent",
    reaction_columns=REACTION_COLUMNS,
    title="Distribuzione delle reaction per intento",
    save_path=None,
):
    """
    Visualizza la composizione delle reaction
    associate ai post di ciascun intento.

    Input:
        DataFrame con una riga per post,
        etichetta di intento e conteggi reaction.

    Le percentuali sono calcolate sui
    conteggi aggregati delle reaction.
    """

    _require_columns(
        df,
        [
            intent_column,
            *reaction_columns,
        ],
    )

    data = df.loc[
        df[intent_column].isin(
            INTENT_ORDER
        )
    ].copy()

    counts = (
        data.groupby(
            intent_column
        )[list(reaction_columns)]
        .sum()
        .reindex(
            INTENT_ORDER,
            fill_value=0,
        )
    )

    totals = counts.sum(axis=1)

    percentages = (
        counts.div(
            totals.replace(0, np.nan),
            axis=0,
        )
        .fillna(0)
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    bottom = np.zeros(
        len(INTENT_ORDER)
    )

    for reaction in reaction_columns:

        values = percentages[
            reaction
        ].to_numpy()

        ax.bar(
            INTENT_ORDER,
            values,
            bottom=bottom,
            label=reaction.capitalize(),
            color=REACTION_COLORS.get(
                reaction
            ),
        )

        bottom += values

    ax.set_title(title)
    ax.set_xlabel("Intento comunicativo")
    ax.set_ylabel("Reaction (%)")
    ax.set_ylim(0, 100)

    ax.legend(
        title="Reaction",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return _save_figure(
        fig,
        save_path,
    )


# ============================================================
# 6. RETE DI CO-CONDIVISIONE
# ============================================================

def plot_cosharing_network(
    graph,
    node_values=None,
    title="Rete di co-condivisione",
    save_path=None,
    seed=42,
):
    """
    Visualizza una rete NetworkX.

    Parametri
    ---------
    graph:
        Grafo dei gruppi Facebook.

    node_values:
        Dizionario facoltativo:
        {nodo: valore numerico}

        Se fornito, determina la colorazione
        dei nodi.

    seed:
        Seme per rendere riproducibile
        il layout della rete.

    Notes
    -----
    Il grafico rappresenta i gruppi e gli archi
    presenti nel grafo ricevuto.

    Non crea nuove relazioni.
    """

    if not isinstance(
        graph,
        nx.Graph,
    ):
        raise TypeError(
            "È richiesto un grafo NetworkX."
        )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    if graph.number_of_nodes() == 0:

        ax.text(
            0.5,
            0.5,
            "Nessun nodo disponibile",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.axis("off")

        return _save_figure(
            fig,
            save_path,
        )

    positions = nx.spring_layout(
        graph,
        seed=seed,
    )

    degrees = dict(
        graph.degree()
    )

    node_sizes = [
        80 + 25 * degrees[node]
        for node in graph.nodes()
    ]

    if node_values is None:

        node_colors = [
            degrees[node]
            for node in graph.nodes()
        ]

    else:

        node_colors = [
            node_values.get(node, 0)
            for node in graph.nodes()
        ]

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        alpha=0.2,
        width=0.8,
    )

    nodes = nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.9,
    )

    fig.colorbar(
        nodes,
        ax=ax,
        label=(
            "Valore del nodo"
            if node_values is not None
            else "Grado del nodo"
        ),
        shrink=0.7,
    )

    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()

    return _save_figure(
        fig,
        save_path,
    )


# ============================================================
# 7. SIMILARITÀ TRA GRUPPI
# ============================================================

def plot_similarity_comparison(
    connected_similarities,
    unconnected_similarities,
    title="Similarità dei profili di reaction",
    save_path=None,
):
    """
    Boxplot della similarità del coseno.

    Confronta:
    - coppie di gruppi collegati;
    - coppie di gruppi non collegati.

    Le distribuzioni devono essere state
    calcolate precedentemente.

    Non esegue test statistici.
    """

    connected = np.asarray(
        connected_similarities,
        dtype=float,
    )

    unconnected = np.asarray(
        unconnected_similarities,
        dtype=float,
    )

    connected = connected[
        np.isfinite(connected)
    ]

    unconnected = unconnected[
        np.isfinite(unconnected)
    ]

    if (
        connected.size == 0
        or unconnected.size == 0
    ):
        raise ValueError(
            "Entrambi i gruppi devono contenere "
            "almeno un valore valido."
        )

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.boxplot(
        [
            connected,
            unconnected,
        ],
        tick_labels=[
            "Collegati",
            "Non collegati",
        ],
        showmeans=True,
    )

    ax.set_title(title)
    ax.set_ylabel("Similarità del coseno")
    ax.set_xlabel("Coppie di gruppi")

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return _save_figure(
        fig,
        save_path,
    )


# ============================================================
# 8. LEAD TIME E INDICATORI
# ============================================================

def plot_lead_time_relationship(
    df,
    lead_time_column="lead_time_hours",
    indicator_column="indicator",
    event_column="event",
    title="Lead time e indicatore quantitativo",
    save_path=None,
):
    """
    Scatter plot del confronto tra eventi.

    Ogni punto rappresenta un evento.

    Non viene aggiunta una regressione lineare,
    poiché il confronto statistico previsto
    utilizza la correlazione di Spearman.

    Il grafico non implica causalità.
    """

    _require_columns(
        df,
        [
            event_column,
            lead_time_column,
            indicator_column,
        ],
    )

    data = df[
        [
            event_column,
            lead_time_column,
            indicator_column,
        ]
    ].copy()

    for column in (
        lead_time_column,
        indicator_column,
    ):

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=[
            lead_time_column,
            indicator_column,
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        data[lead_time_column],
        data[indicator_column],
        s=65,
        alpha=0.85,
    )

    for _, row in data.iterrows():

        ax.annotate(
            str(row[event_column]),
            (
                row[lead_time_column],
                row[indicator_column],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel("Lead time (ore)")
    ax.set_ylabel(indicator_column)

    ax.grid(
        alpha=0.2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return _save_figure(
        fig,
        save_path,
    )
