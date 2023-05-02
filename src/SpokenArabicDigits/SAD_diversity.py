import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from pipeline import *

diversity_pipeline(dataset_name="SpokenArabicDigits", dataset_path="/scratch/luisfeliperamos/time-series-classification/data/SpokenArabicDigits/SpokenArabicDigits_TRAIN.arff")
