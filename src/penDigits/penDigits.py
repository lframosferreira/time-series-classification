import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from pipeline import *

diversity_pipeline(dataset_name="PenDigits", dataset_path="/home/araju/Área de Trabalho/Projetos/IC/datasets/PenDigits/PenDigits_TRAIN.arff")
