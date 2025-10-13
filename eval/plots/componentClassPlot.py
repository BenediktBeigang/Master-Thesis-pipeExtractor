from typing import Optional
from matplotlib import pyplot as plt
import numpy as np


def plot_componentClasses(
    found: int,
    missed: int,
    false_positives: int,
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
    values = [found, missed, false_positives]
    labels = ["Gefunden", "Falsch-Negativ", "Falsch-Positiv"]
    colors = ["#556B2F", "#F75270", "#DC143C"]

    total_counts = sum(values)
    if total_counts == 0:
        raise ValueError("Keine Daten zum Plotten gefunden (Counts).")

    # ---------- Figure (nur Counts) ----------
    fig, ax_counts = plt.subplots(
        figsize=(12, 4),
        constrained_layout=True,
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
    ax_counts.set_xlabel("Anzahl Rohrbauteile", fontsize=11)
    ax_counts.set_title("Erkennung der Rohrbauteile", fontsize=13)
    ax_counts.set_yticks([])
    ax_counts.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax_counts.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # sekundäre x-Achse, die Prozent der Gesamtanzahl anzeigt
    ax_percent = ax_counts.twiny()
    ax_percent.set_xlim(0, 100)
    ax_percent.set_xticks(np.arange(0, 101, 10))  # Ticks alle 10%
    ax_percent.set_xticklabels([f"{t}%" for t in np.arange(0, 101, 10)])
    ax_percent.set_xlabel("Anteil an gesamter Anzahl [%]", fontsize=10)

    plt.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Gespeichert: {out_png}")

    if show:
        plt.show()
    plt.close(fig)
