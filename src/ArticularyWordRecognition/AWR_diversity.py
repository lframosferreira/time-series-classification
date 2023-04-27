import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from pipeline import *

diversity_pipeline(dataset_name="ArticularyWordRecognition", dataset_path="/scratch/luisfeliperamos/time-series-classification/data/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.arff")
