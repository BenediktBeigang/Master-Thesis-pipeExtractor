#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D-Liniendetektion in Z-Slices einer LAS-Punktwolke per Rasterisierung + Hough,
Export aller Linien als OBJ (Polylines) für CloudCompare.

Abhängigkeiten:
    numpy
    laspy        (>=2.0 empfohlen; Fallback für 1.7.0 integriert)
    scikit-image (skimage)
    matplotlib   (für Bildexport)

Beispiel:
    python slice_hough_obj.py \
        --input in.las \
        --thickness 0.10 \
        --cell-size 0.03 \
        --min-count 3 \
        --use-canny \
        --canny-sigma 1.2 \
        --min-line-length-m 0.40 \
        --max-line-gap-m 0.10 \
        --top-k 300 \
        --output lines.obj \
        --save-images \
        --image-output-dir slice_images
"""

from itertools import repeat
import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import math
from multiprocessing import get_context
import os
import time
import sys

from calcSlice import get_z_slices, process_single_slice
from clustering_hough import (
    cluster_segments,
    cluster_segments_strict,
    subcluster_with_segement_z,
)
from export import write_clusters_as_json, write_obj_lines, write_clusters_as_obj
from merge_segments import merge_segments_in_clusters
from parallel_slices import share_xyz_array, _init_shm, worker_process_slice
from util import load_las, prepare_output_directory


def main():
    startTime = time.time()

    ap = argparse.ArgumentParser(
        description="Z-Slice Hough-Liniendetektion aus LAS → OBJ (CloudCompare)"
    )
    ap.add_argument("--input", required=True, help="Pfad zur LAS/LAZ-Datei")
    ap.add_argument(
        "--thickness",
        type=float,
        default=0.1,
        help="Dicke der Z-Slices (Meter)",
    )
    ap.add_argument(
        "--cell-size",
        type=float,
        default=0.01,
        help="Raster-Zellgröße in m (Default: 0.05)",
    )
    ap.add_argument(
        "--canny-sigma",
        type=float,
        default=1.6,
        help="Canny Sigma (bei --use-canny)",
    )
    ap.add_argument(
        "--min-line-length-m",
        type=float,
        default=1.5,
        help="Mindest-Linienlänge (Meter)",
    )
    ap.add_argument(
        "--max-line-gap-m",
        type=float,
        default=0.5,
        help="Max. Lückenspanne zwischen Segmenten (Meter)",
    )
    ap.add_argument(
        "--output",
        default="houghOutput.obj",
        help="Ausgabe-OBJ",
    )
    args = ap.parse_args()

    print(f"Lade Punktwolke: {args.input}")
    xyz = load_las(args.input)
    if xyz.size == 0:
        print("Leere Punktwolke.", file=sys.stderr)
        sys.exit(1)

    print(f"Punktwolke geladen: {xyz.shape[0]} Punkte")

    # Berechne alle Z-Slices
    slices = get_z_slices(xyz, args.thickness)

    # Shared Memory vorbereiten
    shm, shape, dtype_str = share_xyz_array(xyz)

    # Klein gehaltene, picklbare Argumente für den Worker
    args_dict = dict(
        cell_size=args.cell_size,
        canny_sigma=args.canny_sigma,
        min_line_length_m=args.min_line_length_m,
        max_line_gap_m=args.max_line_gap_m,
        local_eps_euclid=0.35,
        local_min_samples=3,
        local_rho_scale=1.0,
        local_preserve_noise=False,
        local_gap_threshold=2.0,
        local_min_length=1.0,
        local_z_max=False,
        merge_segments=False,
    )

    # Tasks in der Reihenfolge der Slices bauen (bewahrt Sortierung)
    tasks = [(i, zc, zmin, zmax) for i, (zc, zmin, zmax) in enumerate(slices)]

    # Verarbeite alle Slices
    all_segments = []
    total_processed = 0

    prepare_output_directory("./output/")

    # robustes Startverfahren wählen:
    # - Linux: 'fork' nutzt COW, spart anfänglich RAM, ist ok wenn du SharedMemory sowieso nutzt
    # - Windows/macOS: 'spawn' ist Standard, SHM funktioniert dort genau für diesen Use-Case
    ctx = get_context()  # Standard-Startmethode des OS
    max_workers = os.cpu_count() or 4
    chunksize = 8  # feinabstimmen bei vielen Slices

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_shm,
            initargs=(shm.name, shape, dtype_str),
        ) as ex:
            # worker_process_slice ist TOP-LEVEL und nimmt (task, args_dict) an
            for slice_idx, segments in ex.map(
                worker_process_slice,
                tasks,  # iterable 1: task tuples
                repeat(args_dict),  # iterable 2: args_dict für jeden Task
                chunksize=chunksize,
            ):
                if segments:
                    all_segments.extend(segments)
                total_processed += 1
                if total_processed % 10 == 0:
                    print(f"Verarbeitet: {total_processed}/{len(slices)} Slices")
    finally:
        # SHM nur im Hauptprozess schließen/unlinken
        shm.close()
        shm.unlink()
        print()

    if len(all_segments) == 0:
        print("Keine Linien in keinem einzigen Slice gefunden.", file=sys.stderr)
        sys.exit(0)

    distance_tolerance_m = 1
    angle_tolerance_deg = 5
    angle_tolerance_radians = math.radians(angle_tolerance_deg)
    rho_scale = (distance_tolerance_m) / (2 * math.sin(angle_tolerance_radians))
    epsilon = math.sqrt(2) * 2 * math.sin(angle_tolerance_radians)
    print(f"rho_scale: {rho_scale:.2f}, eps_euclid: {epsilon:.2f}")

    # Phase 2: Cluster über alle Slices
    print("Phase 2: Clustering über alle Segmente...")
    result_phase2_clustering = cluster_segments(
        all_segments,
        eps_euclid=0.5,
        min_samples=1,
        rho_scale=1.3,
        preserve_noise=True,
    )

    # result_phase2_clustering = cluster_segments_strict(
    #     all_segments,
    #     delta_r_eq=1.0,
    #     delta_deg=5.0,
    #     min_samples=3,
    # )

    # write_clusters_as_obj(
    #     slice_idx=-1,
    #     segments=all_segments,
    #     clusters=result_phase2_clustering["clusters"],
    #     output_dir="./output/obj",
    # )

    # Phase 3: Cluster weiter unterteilen mit Z-Segmentierung
    phase_3_enabled = True
    if phase_3_enabled:
        print("Phase 3: Subclustering mit Z-Segmentierung...")
        result_phase3_clustering = subcluster_with_segement_z(
            segments=all_segments,
            clusters=result_phase2_clustering["clusters"],
            gap=0.2,
        )
    else:
        result_phase3_clustering = result_phase2_clustering["clusters"]

    write_clusters_as_obj(
        slice_idx=-1,
        segments=all_segments,
        clusters=result_phase3_clustering,
        output_dir="./output/obj",
    )

    # write_clusters_as_json(
    #     slice_idx=-1,
    #     segments=all_segments,
    #     clusters=result_phase3_clustering,
    #     output_dir="./output/json",
    # )

    all_segments = merge_segments_in_clusters(
        all_segments,
        result_phase3_clustering,
        gap_threshold=2.0,
        min_length=1.0,
        z_max=True,
    )

    write_obj_lines(
        all_segments,
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.output}",
    )

    print(f"\nFertig!")
    print(f"Verarbeitete Slices: {len(slices)}")
    print(f"Gefundene Linien gesamt: {len(all_segments)}")
    print(f"Ausgabedatei: {args.output}")
    endTime = time.time()
    print(f"Benötigte Zeit: {endTime - startTime:.2f} Sekunden")


if __name__ == "__main__":
    main()
