import pdal
import json
import argparse
import os
import time
import glob

radius = 0.02


def process(input_file, output_file):
    start = time.time()

    if not os.path.exists(input_file):
        print(f"Fehler: Datei {input_file} nicht gefunden")
        return

    pipeline = {
        "pipeline": [
            input_file,
            {
                "type": "filters.expression",
                "expression": "Classification == 1 || Classification == 2",
            },
            {
                "type": "filters.sample",
                "radius": radius,
            },
            {
                "type": "writers.las",
                "filename": output_file,
            },
        ]
    }

    pipe = pdal.Pipeline(json.dumps(pipeline))
    count = pipe.execute()

    end = time.time()
    print(f"{end - start:.2f} sec - {output_file}")


if __name__ == "__main__":
    las_files = [
        "/mnt/c/Users/bened/Downloads/0904/ontras_0/ontras_0_predicted_0904_t1.las",
        "/mnt/c/Users/bened/Downloads/0904/ontras_1/ontras_1_predicted_0904_t1.las",
        "/mnt/c/Users/bened/Downloads/0904/ontras_2/ontras_2_predicted_0904_t1.las",
        "/mnt/c/Users/bened/Downloads/0904/ontras_3/ontras_3_predicted_0904_t1.las",
        "/mnt/c/Users/bened/Downloads/0904/ontras_4/ontras_4_predicted_0904_t1.las",
    ]
    target_folder = "/mnt/c/Users/bened/Downloads/0904/"

    for las_file in las_files:
        las_file_without_extension = os.path.splitext(os.path.basename(las_file))[0]
        output_file = os.path.join(
            target_folder, f"{las_file_without_extension}_sampled.las"
        )
        process(las_file, output_file)
