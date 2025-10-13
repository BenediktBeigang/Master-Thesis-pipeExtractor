from typing import Optional, Sequence
from matplotlib import pyplot as plt
import numpy as np


def plot_boxplots_lineDistances(
    line_dist_xy_samples: Optional[Sequence[float]],
    line_dist_z_samples: Optional[Sequence[float]],
    out_png: Optional[str],
    part: str,
    title: str,
    show: bool = False,
) -> None:
    """
    Erstellt Boxplots für lineare Abstände und (optional) Partial-Überlappungen.
    - line_dist_samples: Liste/Array von Punkt-zu-Linie-Distanzen (float)
    - partial_overlap_samples: Liste/Array mit Überlappungsanteilen [0..1]
    """

    # In saubere Arrays konvertieren, NaNs/Inf filtern
    def _clean(x):
        if x is None:
            return np.array([], dtype=float)
        arr = np.asarray(list(x), dtype=float)
        return arr[np.isfinite(arr)]

    line_xy_arr = _clean(line_dist_xy_samples)
    line_z_arr = _clean(line_dist_z_samples)

    data = []
    labels = []

    if line_z_arr.size > 0:
        data.append(line_z_arr)
        labels.append(f"Abstände der\n{part} z")

    if line_xy_arr.size > 0:
        data.append(line_xy_arr)
        labels.append(f"Abstände der\n{part} xy")

    if not data:
        raise ValueError("Keine Daten zum Plotten gefunden.")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 3))
    bp = ax.boxplot(
        data,
        vert=False,
        labels=labels,
        showmeans=True,
        meanline=False,
        whis=1.5,
        widths=0.6,
    )
    ax.set_title(title)
    ax.set_xlabel("Abstand [m]")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    max_dist = max(max(line_dist_xy_samples), max(line_dist_z_samples))

    # X-Achsen-Range und Ticks setzen
    ax.set_xlim(-0.05, max_dist + 0.05)
    ax.set_xticks(np.arange(-0.05, max_dist + 0.05, 0.05))
    ax.set_xticklabels(
        [
            f"{x:.2f}" if round(x, 2) % 0.25 == 0 else ""
            for x in np.arange(-0.05, max_dist + 0.05, 0.05)
        ]
    )  # Labels nur alle 0.25

    if "Overlap" in " ".join(labels):
        ax.set_ylim(bottom=0.0)

    # Legende für Boxplot-Elemente hinzufügen
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color="orange",
            # linestyle="None",
            markersize=8,
            label="Median",
        ),
        plt.Line2D(
            [0],
            [0],
            color="green",
            marker="^",
            linestyle="None",
            markersize=8,
            label="Mittelwert",
        ),
        plt.Line2D([0], [0], color="black", linewidth=1, label="Box (Q1-Q3)"),
        plt.Line2D([0], [0], color="black", linewidth=1, label="Whiskers (1.5×IQR)"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor="black",
            label="Ausreißer",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    if out_png:
        plt.tight_layout()
        fig.savefig(out_png, dpi=300)
        print(f"Gespeichert: {out_png}")

    if show:
        plt.show()
    plt.close(fig)
