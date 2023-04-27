import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from pipeline import *

model_pipeline(prefix="results/rocket/NATOPS/8_10_11_12_16_17_21_22", model_name="Rocket_XGBoost", train_data_path="/scratch/luisfeliperamos/time-series-classification/data/NATOPS/NATOPS_TRAIN.arff", test_data_path="/scratch/luisfeliperamos/time-series-classification/data/NATOPS/NATOPS_TEST.arff", dimensions_to_use = np.array([8, 10, 21, 22]))