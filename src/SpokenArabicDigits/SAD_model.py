import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from pipeline import *

model_pipeline(prefix="results/rocket/SAD/all", model_name="Rocket_XGBoost", train_data_path="/scratch/luisfeliperamos/time-series-classification/data/SpokenArabicDigits/SpokenArabicDigits_TRAIN.arff", test_data_path="/scratch/luisfeliperamos/time-series-classification/data/SpokenArabicDigits/SpokenArabicDigits_TEST.arff", dimensions_to_use = None)