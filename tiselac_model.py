from datetime import datetime
import json
import numpy as np
import pandas as pd
import pickle
import time
import scipy
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sktime.transformations.panel.rocket import Rocket
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

MODEL_NAME: str = "Rocket_XGBoost"

def save_pickle(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)

train_data: np.array = scipy.io.arff.loadarff("data/Tiselac/Tiselac_TRAIN.arff")
train_data = train_data[0]

test_data: np.array = scipy.io.arff.loadarff("data/Tiselac/Tiselac_TEST.arff")
test_data = test_data[0]

X_train = np.array([np.array(e[0]) for e in train_data])
X_train = np.array(X_train.tolist(), dtype=np.float_)

# Removendo dimensões (deixando 7 e 8)
X_train = np.delete(X_train, [0, 1, 2, 3, 4, 5, 6, 9] , axis=1)

y_train = np.array([int(e[1]) for e in train_data])
y_train = le.fit_transform(y_train)

X_test = np.array([np.array(e[0]) for e in test_data])
X_test = np.array(X_test.tolist(), dtype=np.float_)

# Removendo dimensões (deixando 7 e 8)
X_test = np.delete(X_test, [0, 1, 2, 3, 4, 5, 6, 9] , axis=1)

y_test = np.array([int(e[1]) for e in test_data])
y_test = le.fit_transform(y_test)

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

f1 = f1_score(y_test, y_pred, average=None)

now = datetime.now()
ts = now.strftime("%m-%d-%Y_%H:%M:%S")
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

save_pickle(conf_matrix, "confusion_matrix.pickle")

classifier.save_model(f"./{MODEL_NAME}_{ts}.pickle")

run_info = {
    'model_name': MODEL_NAME, 
    'model_params': {},
    'train_size': len(train_data),
    'test_size': len(test_data),
    'accuracy': accuracy,
    'f1_score': f1.tolist(),
    'random_state_seed': RANDOM_STATE,
    'timestamp': str(now)
}

with open(f"./run_info_{ts}.json", "w") as file:
    json.dump(run_info, file, indent = 4)
