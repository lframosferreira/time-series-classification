from src import pipeline

pipeline(prefix="results/rocket/NATOPS", model_name="Rocket_XGBoost", train_data_path="../../data/NATOPS/NATOPS_TRAIN.arff", test_data_path="../../data/NATOPS/NATOPS_TEST.arff")