import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from pipeline import *

DATASET_NAME: str = "PhonemeSpectra"

diversity_pipeline(dataset_name=DATASET_NAME, dataset_path=f"/scratch/luisfeliperamos/time-series-classification/data/{DATASET_NAME}/{DATASET_NAME}_TRAIN.arff")
