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

import argparse
import datetime
import time
import sys

from calcSlice import get_z_slices, process_single_slice
from clustering_hough import (
    cluster_segments,
    merge_segments_in_slice,
    merge_segments_global,
    subcluster_with_segement_z,
)
from export import write_obj_lines, write_clusters_as_obj
from util import load_las_xyz, prepare_output_directory


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
        default=0.02,
        help="Raster-Zellgröße in m (Default: 0.05)",
    )
    ap.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Schwellwert für Counts (ohne Canny)",
    )
    ap.add_argument(
        "--use-canny",
        action="store_true",
        default=True,
        help="Kanten via Canny (empfohlen)",
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
        "--top-k-total",
        type=int,
        default=0,
        help="Nur die längsten K Segmente insgesamt behalten (0 = alle)",
    )
    ap.add_argument(
        "--output",
        default="houghOutput.obj",
        help="Ausgabe-OBJ",
    )
    ap.add_argument(
        "--slices-range",
        type=str,
        default="-1",
        help="Nur Slices in diesem Z-Bereich speichern (min;max), z.B. 17;19",
    )
    args = ap.parse_args()

    print(f"Lade Punktwolke: {args.input}")
    xyz = load_las_xyz(args.input)
    if xyz.size == 0:
        print("Leere Punktwolke.", file=sys.stderr)
        sys.exit(1)

    print(f"Punktwolke geladen: {xyz.shape[0]} Punkte")

    # Berechne alle Z-Slices
    slices = get_z_slices(xyz, args.thickness)

    # Verarbeite alle Slices
    all_segments = []
    total_processed = 0

    if args.slices_range == "-1":
        minSliceId = 0
        maxSliceId = len(slices) - 1
    else:
        minSliceId = int(args.slices_range.split(";")[0])
        maxSliceId = int(args.slices_range.split(";")[1])
    print(f"Speichere nur Slices im Bereich {minSliceId} .. {maxSliceId}")

    prepare_output_directory("./output/")

    for i, (z_center, zmin, zmax) in enumerate(slices):
        try:
            slice_in_range = i >= minSliceId and i <= maxSliceId
            if not slice_in_range:
                continue

            # Calc Segements with Hough
            segments = process_single_slice(
                xyz, z_center, zmin, zmax, args, i, slice_in_range
            )

            # Cluster and save debug obj
            result = cluster_segments(
                segments,
                eps_euclid=0.35,
                min_samples=3,
                rho_scale=1.0,
                preserve_noise=False,
            )
            # write_clusters_as_obj(
            #     slice_idx=i,
            #     segments=segments,
            #     clusters=result["clusters"],
            #     output_dir="./output/obj",
            #     z_value=z_center,
            # )
            print(f"  → Gefundene Cluster: {len(result['clusters'])}")

            if "clusters" not in result or not result["clusters"]:
                continue

            concatenated_segments = merge_segments_in_slice(
                segments,
                result["clusters"],
                gap_threshold=2.0,
                min_length=1.0,
            )
            write_obj_lines(
                concatenated_segments, f"./output/obj/slice_{i:03d}_concat.obj"
            )
            print(f"  → Zusammengeführt Segemente: {len(concatenated_segments)}")
            all_segments.extend(concatenated_segments)

        except Exception as e:
            print(f"Fehler bei Slice {i}: {e}", file=sys.stderr)
        finally:
            total_processed += 1
            if total_processed % 10 == 0:
                print(f"Verarbeitet: {total_processed}/{len(slices)} Slices")

    if len(all_segments) == 0:
        print("Keine Linien in keinem einzigen Slice gefunden.", file=sys.stderr)
        sys.exit(0)

    result_phase2_clustering = cluster_segments(
        all_segments,
        eps_euclid=0.5,
        min_samples=1,
        rho_scale=1.3,
        preserve_noise=True,
    )

    # write_clusters_as_obj(
    #     slice_idx=-1,
    #     segments=all_segments,
    #     clusters=result_phase2_clustering["clusters"],
    #     output_dir="./output/obj",
    # )

    result_phase3_clustering = subcluster_with_segement_z(
        segments=all_segments,
        clusters=result_phase2_clustering["clusters"],
        gap=0.3,
    )

    write_clusters_as_obj(
        slice_idx=-1,
        segments=all_segments,
        clusters=result_phase3_clustering,
        output_dir="./output/obj",
    )

    all_segments = merge_segments_global(
        all_segments,
        result_phase3_clustering,
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
