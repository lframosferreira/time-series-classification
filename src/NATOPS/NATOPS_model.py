from datetime import datetime
import json
import numpy as np
import pandas as pd
import pickle
import time
import scipy
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sktime.transformations.panel.rocket import Rocket
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def pipeline(prefix: str, model_name: str, train_data_path: str, test_data_path: str, dimensions_to_use: np.array = None) -> None:

    PREFIX: str = prefix

    MODEL_NAME: str = model_name

    def save_pickle(data, path):
        with open(path, "wb") as file:
            pickle.dump(data, file)

    train_data: np.array = scipy.io.arff.loadarff(f"{train_data_path}")
    train_data = train_data[0]

    test_data: np.array = scipy.io.arff.loadarff(f"{test_data_path}")
    test_data = test_data[0]

    X_train = np.array([np.array(e[0]) for e in train_data])
    X_train = np.array(X_train.tolist(), dtype=np.float_)

    ORIGINAL_NUMBER_OF_DIMENSIONS: int = X_train.shape[1]

    if dimensions_to_use is not None:
        X_train = np.delete(X_train, np.setdiff1d(np.arange(X_train.shape[1]), dimensions_to_use), axis=1)

    y_train = np.array([int(float(e[1])) for e in train_data])
    y_train = le.fit_transform(y_train)

    X_test = np.array([np.array(e[0]) for e in test_data])
    X_test = np.array(X_test.tolist(), dtype=np.float_)

    if dimensions_to_use is not None:
        X_test = np.delete(X_test, np.setdiff1d(np.arange(X_test.shape[1]), dimensions_to_use), axis=1)

    y_test = np.array([int(float(e[1])) for e in test_data])
    y_test = le.fit_transform(y_test)

    # cleaning up RAM
    del train_data
    del test_data

    ROCKET_KERNELS: np.int_ = 10_000
    RANDOM_STATE: np.int_ = 0

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

    run_info = {
        'model_name': MODEL_NAME, 
        'model_params': {},
        'train_size': len(X_train),
        'test_size': len(X_test),
        'accuracy': accuracy,
        'recall': recall.tolist(),
        'precision': precision.tolist(),
        'f1_score': f1.tolist(),
        'random_state_seed': RANDOM_STATE,
        'timestamp': str(now),
        'dimensios_used': dimensions_to_use.tolist() if dimensions_to_use is not None else list(range(ORIGINAL_NUMBER_OF_DIMENSIONS))
    }

    with open(f"{PREFIX}/run_info_{ts}.json", "w") as file:
        json.dump(run_info, file, indent = 4)


pipeline(prefix="results/rocket/NATOPS", model_name="Rocket_XGBoost", train_data_path="/scratch/luisfeliperamos/time-series-classification/data/NATOPS/NATOPS_TRAIN.arff", test_data_path="/scratch/luisfeliperamos/time-series-classification/data/NATOPS/NATOPS_TEST.arff")