"""
Downloads a small labeled subset of COCO val2017 for accuracy testing.

Uses FiftyOne's dataset zoo, which pulls only the images/annotations you
ask for (not the full 6GB val2017 set) and exports them in a simple
folder + labels format to use in check_accuracy.py.

Install:
    pip install fiftyone

Usage:
    python fetch_coco_subset.py
    (edit CLASSES / NUM_IMAGES below to taste)
"""

import fiftyone as fo
import fiftyone.zoo as foz
import os


# Which COCO classes to pull. 
CLASSES = ["person", "car", "truck", "bicycle", "dog"]

NUM_IMAGES = 40       
SPLIT = "validation"     # val2017
OUTPUT_DIR = "./coco_subset" 


def main():
    print(f"Fetching {NUM_IMAGES} COCO val2017 images "
          f"(classes: {CLASSES or 'all'})...")

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=SPLIT,
        label_types=["detections"],
        classes=CLASSES,
        max_samples=NUM_IMAGES,
        shuffle=True,
        seed=42,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Export as COCO-format JSON + image folder 
    dataset.export(
        export_dir=OUTPUT_DIR,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
    )

    print(f"Done. Images + annotations saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("Structure:")
    print("  coco_subset/data/      <- images")
    print("  coco_subset/labels.json <- COCO-format ground truth annotations")

    # Optional: clean up FiftyOne's local DB entry for this dataset
    # fo.delete_dataset(dataset.name)


if __name__ == "__main__":
    main()