from typing import Optional
from matplotlib import pyplot as plt
import numpy as np


def plot_segmentClasses(
    correct: int,
    partial: int,
    missed: int,
    false_positives: int,
    coverage: float,  # Annahme: abgedeckte LÄNGE (z.B. Meter)
    missed_length: float,
    false_positive_length: float,
    out_png: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Zeichnet:
    - Oben: gestapelter horizontaler Balken nach Anzahl (Korrekt / Teilweise / Falsch-Negativ / Falsch-Positiv)
    - Unten: gestapelter horizontaler Balken nach LÄNGEN (abgedeckt, verpasst, falsch-positiv) mit eigener Skalierung

    Hinweis: `coverage` wird hier als absolute Länge interpretiert.
    """

    # ---------- Vorbereitung Counts ----------
    values = [correct, partial, missed, false_positives]
    labels = ["Korrekt", "Teilweise", "Falsch-Negativ", "Falsch-Positiv"]
    colors = ["#556B2F", "#C6D870", "#F75270", "#DC143C"]

    total_counts = sum(values)
    if total_counts == 0:
        raise ValueError("Keine Daten zum Plotten gefunden (Counts).")

    # ---------- Vorbereitung Lengths ----------
    len_values = [coverage, missed_length, false_positive_length]
    len_labels = [
        "Gültige Abschnitte",
        "Falsch-Negative",
        "Falsch-Positive",
    ]
    len_colors = ["#556B2F", "#F75270", "#DC143C"]

    total_length = sum(len_values)
    if total_length == 0:
        # Wenn keine Längen vorhanden sind, zeichne nur den Count-Plot.
        total_length = 0.0

    # ---------- Figure mit zwei Zeilen (Counts oben, Lengths unten) ----------
    fig, (ax_counts, ax_lengths) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 1.0},
    )

    # ---- Counts (oben) ----
    cumulative = 0
    for value, label, color in zip(values, labels, colors):
        if value > 0:
            pct = value / total_counts * 100
            ax_counts.barh(
                0,
                value,
                left=cumulative,
                color=color,
                label=f"{label}: {value} ({pct:.1f}%)",
                height=0.6,
            )
            # Text im Segment nur wenn breit genug (relativ)
            if pct > 8:
                ax_counts.text(
                    cumulative + value / 2,
                    0,
                    f"{value}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                    fontsize=12,
                )
            cumulative += value

    ax_counts.set_xlim(0, total_counts)
    ax_counts.set_ylim(-0.6, 0.6)
    ax_counts.set_xlabel("Anzahl Rohre", fontsize=11)
    ax_counts.set_title("Erkennung der Rohre (pro Rohr)", fontsize=13)
    ax_counts.set_yticks([])
    ax_counts.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax_counts.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # sekundäre x-Achse, die Prozent der Gesamtanzahl anzeigt
    ax_percent = ax_counts.twiny()
    ax_percent.set_xlim(0, 100)
    ax_percent.set_xticks(np.arange(0, 101, 10))  # Ticks alle 10%
    ax_percent.set_xticklabels([f"{t}%" for t in np.arange(0, 101, 10)])
    ax_percent.set_xlabel("Anteil an gesamter Anzahl [%]", fontsize=10)

    # ---- Lengths (unten) ----
    if total_length > 0:
        cumulative = 0.0
        for value, label, color in zip(len_values, len_labels, len_colors):
            if value > 0:
                pct = value / total_length * 100
                ax_lengths.barh(
                    0,
                    value,
                    left=cumulative,
                    color=color,
                    label=f"{label}: {value:.2f} ({pct:.1f}%)",
                    height=0.6,
                )
                # Text im Segment: zeige absolute Länge; nur wenn Segment relativ groß genug
                if pct > 6:
                    ax_lengths.text(
                        cumulative + value / 2,
                        0,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        color="white",
                        fontsize=12,
                    )
                cumulative += value

        ax_lengths.set_xlim(0, total_length)
        ax_lengths.set_ylim(-0.6, 0.6)
        ax_lengths.set_xlabel("Länge [m]", fontsize=11)
        ax_lengths.set_title("Erkennung der Rohre (Länge)", fontsize=13)
        ax_lengths.set_yticks([])
        ax_lengths.grid(True, axis="x", linestyle="--", alpha=0.3)
        ax_lengths.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

        # sekundäre x-Achse, die Prozent des Gesamtlängen-Stacks anzeigt
        ax_percent = ax_lengths.twiny()
        ax_percent.set_xlim(0, 100)
        ax_percent.set_xticks(np.arange(0, 101, 10))
        ax_percent.set_xticklabels([f"{t}%" for t in np.arange(0, 101, 10)])
        ax_percent.set_xlabel("Anteil an gesamter Länge [%]", fontsize=10)
    else:
        ax_lengths.text(
            0.5,
            0.5,
            "Keine Längen-Daten vorhanden",
            ha="center",
            va="center",
            transform=ax_lengths.transAxes,
            fontsize=11,
            color="gray",
        )
        ax_lengths.set_xticks([])
        ax_lengths.set_yticks([])

    plt.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Gespeichert: {out_png}")

    if show:
        plt.show()
    plt.close(fig)
