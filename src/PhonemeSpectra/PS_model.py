import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from pipeline import *

DATASET_NAME: str = "PhonemeSpectra"

model_pipeline(prefix="results/rocket/{DATASET_NAME}/all", model_name="Rocket_XGBoost", train_data_path="/scratch/luisfeliperamos/time-series-classification/data/{DATASET_NAME}/{DATSET_NAME}_TRAIN.arff", test_data_path="/scratch/luisfeliperamos/time-series-classification/data/{DATASET_NAME}/{DATASET_NAME}_TEST.arff", dimensions_to_use = None)