from datetime import datetime
import json
import numpy as np
import pandas as pd
import pickle
import time
import scipy
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import Rocket
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def model_pipeline(prefix: str, model_name: str, train_data_path: str, test_data_path: str, dimensions_to_use: np.array = None) -> None:

    PREFIX: str = prefix

    MODEL_NAME: str = model_name

    RANDOM_STATE: np.int_ = 0
    TEST_SIZE: np.float_ = 0.3

    def save_pickle(data, path):
        with open(path, "wb") as file:
            pickle.dump(data, file)

    train_data: np.array = scipy.io.arff.loadarff(f"{train_data_path}")
    train_data = train_data[0]

    test_data: np.array = scipy.io.arff.loadarff(f"{test_data_path}")
    test_data = test_data[0]

    all_data: np.array = np.concatenate([train_data, test_data])

    input_data: np.array = np.array([np.array(e[0]) for e in all_data])
    labels: np.array = np.array([np.array(e[1]) for e in all_data])

    ORIGINAL_NUMBER_OF_DIMENSIONS: np.int_ = all_data[0][0].shape

    X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train = np.array(X_train.tolist(), dtype=np.float_)

    if dimensions_to_use is not None:
        X_train = np.delete(X_train, np.setdiff1d(np.arange(X_train.shape[1]), dimensions_to_use), axis=1)

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

    t0 = time.time()
    classifier.fit(X_train, y_train)
    t1 = time.time()

    y_pred = classifier.predict(X_test)

    # Evaluate the performance
    accuracy = classifier.score(X_test, y_test)

    recall = recall_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    now = datetime.now()
    ts = now.strftime("%m-%d-%Y_%H:%M:%S")
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    save_pickle(conf_matrix, F"{PREFIX}/confusion_matrix.pickle")

    classifier.save_model(f"{PREFIX}/{MODEL_NAME}_{ts}.pickle")

    cr = classification_report(y_test, y_pred, output_dict=True)

    run_info = {
        'model_name': MODEL_NAME, 
        'model_params': {},
        'train_size': len(X_train),
        'test_size': len(X_test),
        'classification_report': cr,
        'random_state_seed': RANDOM_STATE,
        'timestamp': str(now),
        'dimensios_used': dimensions_to_use.tolist() if dimensions_to_use is not None else list(range(ORIGINAL_NUMBER_OF_DIMENSIONS))
    }

    with open(f"{PREFIX}/run_info_{ts}.json", "w") as file:
        json.dump(run_info, file, indent = 4)

import ts2vg
import matplotlib.pyplot as plt
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

def diversity_pipeline(dataset_name: str, dataset_path: str) -> None:

    data: np.array = scipy.io.arff.loadarff(f"{dataset_path}")
    data = data[0]
    
    print("Iniciando calculos de diversidade")
    for i in range(len(data)):
        node_distributions: list = []
        transition_matrices: list = []
        visibility_graphs: list = []
        time_series_set, label = data[i]
        print(f"Calculando grafo do exemplo {i}")
        for j in range(len(time_series_set)):
            graph: np.array = ts2vg.NaturalVG().build(time_series_set[j].tolist()).adjacency_matrix()
            visibility_graphs.append(graph)
        node_distributions = run_multiprocessing(calculate_node_distribution, visibility_graphs, n_processors=2)
        transition_matrices = run_multiprocessing(calculate_transition_matrix, visibility_graphs, n_processors=2)

        sparse_visibility_graphs = np.array([scipy.sparse.csr_matrix(graph, dtype=np.int8) for graph in visibility_graphs])
        less_contribute_rank: np.array = less_contribute(node_distributions, transition_matrices)
        k = 0
        with h5sparse.File(f"{dataset_name}.hdf5", "a") as f:
            for graph in sparse_visibility_graphs:
                f.create_dataset(f"visibility_graphs/{i}/{k}", data=graph)
                k += 1
            f.create_dataset(f"diversity_rank/{i}", data=less_contribute_rank)
