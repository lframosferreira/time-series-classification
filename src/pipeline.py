from datetime import datetime
import json
import numpy as np
import pandas as pd
import pickle
import scipy
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import Rocket
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def classifier_pipeline(dataset_name: str, train_path: str, test_path: str, dimensions_to_use: np.array = None) -> None:

    MODEL_NAME: str = "Rocket_XGBoost"

    RANDOM_STATE: np.int_ = 0
    TEST_SIZE: np.float_ = 0.3

    def save_pickle(data, path):
        with open(path, "wb") as file:
            pickle.dump(data, file)

    train_data: np.array = scipy.io.arff.loadarff(f"{train_path}")
    train_data = train_data[0]

    test_data: np.array = scipy.io.arff.loadarff(f"{test_path}")
    test_data = test_data[0]

    all_data: np.array = np.concatenate([train_data, test_data])

    input_data: np.array = np.array([np.array(e[0]) for e in all_data])
    labels: np.array = np.array([np.array(e[1]) for e in all_data])

    ORIGINAL_NUMBER_OF_DIMENSIONS: np.int_ = all_data[0][0].shape[0]

    X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train = np.array(X_train.tolist(), dtype=np.float_)

    if dimensions_to_use is not None:
        X_train = np.delete(X_train, np.setdiff1d(np.arange(X_train.shape[1]), np.array(dimensions_to_use)), axis=1)

    y_train = le.fit_transform(y_train)

    X_test = np.array(X_test.tolist(), dtype=np.float_)

    if dimensions_to_use is not None:
        X_test = np.delete(X_test, np.setdiff1d(np.arange(X_test.shape[1]), dimensions_to_use), axis=1)

    y_test = le.fit_transform(y_test)

    # cleaning up RAM
    del train_data
    del test_data

    ROCKET_KERNELS: np.int_ = 10_000

    trf = Rocket(num_kernels=ROCKET_KERNELS, random_state=RANDOM_STATE, n_jobs=-1)
    trf.fit(X_train)
    trf.save("rocket_transform")
    
    X_train = trf.transform(X_train) 
    X_test = trf.transform(X_test) 

    classifier = XGBClassifier()

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Evaluate the performance
    accuracy = classifier.score(X_test, y_test)

    recall_score = metrics.recall_score(y_test, y_pred, average=None)
    precision_score = metrics.precision_score(y_test, y_pred, average=None)
    f1_score = metrics.f1_score(y_test, y_pred, average=None)

    now = datetime.now()
    ts = now.strftime("%m-%d-%Y_%H:%M:%S")
    confusion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)

    # classifier.save_model(f"models/{MODEL_NAME}/{dataset_name}.pickle")

    run_info = {
        "model_name": MODEL_NAME,
        "dataset_name": dataset_name,
        "model_params": {},
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state_seed": RANDOM_STATE,
        "timestamp": str(now),
        "dimensions_used": dimensions_to_use if dimensions_to_use is not None else list(range(ORIGINAL_NUMBER_OF_DIMENSIONS)),
        "accuracy": accuracy,
        "f1_avg": np.average(f1_score),
        "recall_avg": np.average(recall_score),
        "precision_avg": np.average(precision_score),
        "f1_score": f1_score.tolist(),
        "recall_score": recall_score.tolist(),
        "precision_score": precision_score.tolist(),
        "confusion_matrix": confusion_matrix.tolist()
    }

    results: pd.DataFrame = pd.read_json("results/results.json")
    results = pd.concat([results, pd.DataFrame.from_dict(run_info, orient="index").transpose()])
    results.to_json("results/results.json", orient="records")


import ts2vg
import h5sparse
from multiprocessing import Pool
from diversity_measure import *

def run_multiprocessing(func, iterable, n_processors):
        with Pool(processes=n_processors) as pool:
            return pool.map(func, iterable)
        
def calculate_node_distribution(graph: np.array) -> np.array:
    return node_distribution(graph)

def calculate_transition_matrix(graph: np.array) -> np.array:    
    return transition_matrix(graph)

def diversity_pipeline(dataset_name: str, train_path: str, test_path: str) -> None:

    train_data: np.array = scipy.io.arff.loadarff(f"{train_path}")
    train_data = train_data[0]

    test_data: np.array = scipy.io.arff.loadarff(f"{test_path}")
    test_data = test_data[0]

    data: np.array = np.concatenate([train_data, test_data])
    
    print("Iniciando calculos de diversidade")
    for i in range(len(data)):
        node_distributions: list = []
        transition_matrices: list = []
        visibility_graphs: list = []
        time_series_set, _ = data[i]
        print(f"Calculando grafos do exemplo {i}")
        for j in range(len(time_series_set)):
            graph: np.array = ts2vg.NaturalVG().build(time_series_set[j].tolist()).adjacency_matrix()
            visibility_graphs.append(graph)
        node_distributions = run_multiprocessing(calculate_node_distribution, visibility_graphs, n_processors=4)
        transition_matrices = run_multiprocessing(calculate_transition_matrix, visibility_graphs, n_processors=4)

        sparse_visibility_graphs = np.array([scipy.sparse.csr_matrix(graph, dtype=np.int8) for graph in visibility_graphs])
        less_contribute_rank: np.array = less_contribute(node_distributions, transition_matrices)
        k = 0
        with h5sparse.File(f"{dataset_name}.hdf5", "a") as f:
            for graph in sparse_visibility_graphs:
                f.create_dataset(f"visibility_graphs/{i}/{k}", data=graph)
                k += 1
            f.create_dataset(f"diversity_rank/{i}", data=less_contribute_rank)