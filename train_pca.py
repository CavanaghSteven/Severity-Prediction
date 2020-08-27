
import numpy as np
from util import load_data
from util import create_logfile
import csv
import pandas as pd
from sklearn.decomposition import PCA
from vanilla_ann import DataGenerator, Prediction_Model, train_model
import keras
from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import RandomUnderSampler


if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_data()
    y_train_onehot = keras.utils.to_categorical(y_train, len(np.unique(y_train.values.astype('float32'))))

    num_features = 6
    layers = [64, 64, 64]
    model_name = 'ann-pca'
    num_outputs = len(np.unique(y_train.astype(int)))
    batch_size = 128

    pca = PCA(n_components=num_features)
    pca.fit(X_train)
    comp_columns = [f'comp {i}' for i in range(len(pca.components_))]

    X_train = pd.DataFrame(pca.transform(X_train), columns=comp_columns)
    X_val = pd.DataFrame(pca.transform(X_val), columns=comp_columns)
    X_test = pd.DataFrame(pca.transform(X_test), columns=comp_columns)

    model = Prediction_Model((num_features, layers, num_outputs, model_name))

    # Parameters
    params = {'num_features': num_features,
              'batch_size': batch_size,
              'shuffle': True,
              'num_outputs': num_outputs}

    # training_generator = DataGenerator(X_train, y_train, **params)
    sampler = RandomUnderSampler(random_state=42)
    training_generator = BalancedBatchGenerator(
        X_train, y_train_onehot, sampler=sampler, batch_size=batch_size, random_state=42)
    validation_generator = DataGenerator(X_val, y_val, **params)
    test_generator = DataGenerator(X_test, y_test, **params)

    train_model(
        model,
        training_generator,
        validation_generator,
        model_name,
        num_epochs=30
    )

    model.load_weights(f'log/{model_name}/weights.hdf5')
    log_fname = f'log/repres_losses.csv'

    create_logfile(log_fname)

    scores = model.evaluate_generator(test_generator)

    with open(log_fname, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([model_name] + scores)

    # print('test loss', model_name, scores)
