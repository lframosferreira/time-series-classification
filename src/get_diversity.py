from pipeline import *
import argparse

parser = argparse.ArgumentParser(description="Input for diversity script")
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--train_path", type=str)
parser.add_argument("--test_path", type=str)

args = parser.parse_args()

dataset_name: str = args.dataset_name
train_path: str = args.train_path
test_path: str = args.test_path

diversity_pipeline(dataset_name=dataset_name, train_path=train_path, test_path=test_path)
